import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as linalg
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import SEM
import openmdao.api as om
import time


class NavierStokes(om.ImplicitComponent):
    """
    Implicit component to solve the steady-state NAVIER-STOKES equation for u(x,y) and v(x,y) given the
    temperature T(x,y)
    Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v] + Gr/Re [0, T] ∀(x,y)∈[0,L_x]×[0,L_y]
    ∇∘[u, v] = 0 ∀(x,y)∈[0,L_x]×[0,L_y]
    with no normal flow and tangential DIRICHLET conditions
    v(0,y)   = v_W y∈[0,L_y]
    v(L_x,y) = v_E y∈[0,L_y]
    u(x,0)   = u_S x∈[0,L_x]
    u(x,L_y) = u_N x∈[0,L_x]
    Artificial NEUMANN boundary condition for the pressure
    ∂ₙp = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])
    """

    def initialize(self):
        # declare parameters
        self.options.declare('L_x', types=(float, int), desc='length in x direction')
        self.options.declare('L_y', types=(float, int), desc='length in y direction')
        self.options.declare('Re', types=(float, int), desc='REYNOLDS number')
        self.options.declare('Gr', types=(float, int), default=0, desc='GRASHOF number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('points', types=np.ndarray, desc='points as global vectors [x, y]')
        self.options.declare('u_N', types=(float, int), default=0, desc='tangential velocity at y=L_y')
        self.options.declare('u_S', types=(float, int), default=0, desc='tangential velocity at y=0')
        self.options.declare('v_E', types=(float, int), default=0, desc='tangential velocity at x=0')
        self.options.declare('v_W', types=(float, int), default=0, desc='tangential velocity at x=L_x')
        self.options.declare('mtol', types=(float, int), default=1e-7, desc='tolerance on mean square residual')
        self.options.declare('solver_type', types=str, default='lu', desc='solver: lu/qmr')
        self.options.declare('iprecon_type', types=str, default='jac', desc='inner preconditioner (qmr): jac/ilu/no')
        self.options.declare('drop_tol', types=(float, int), default=1e-3, desc='ILU drop tolerance')
        self.options.declare('fill_factor', types=(float, int), default=2, desc='ILU fill factor')

    def setup(self):
        # load parameters
        self.Re = self.options['Re']
        self.Gr = self.options['Gr']
        self.points = self.options['points']
        self.L_x = self.options['L_x']
        self.L_y = self.options['L_y']
        P = self.options['P']
        N_ex = self.options['N_ex']
        N_ey = self.options['N_ey']
        self.N = (N_ex*P+1)*(N_ey*P+1)

        # check singularity
        if self.Re == 0 and self.Gr != 0:
            raise ValueError('Cannot have Re == 0 and Gr != 0')
        self.Gr_over_Re = self.Gr/self.Re if self.Re != 0 else 0.

        # declare variables
        self.add_output('u', val=np.zeros(self.N), desc='u as global vector')
        self.add_output('v', val=np.zeros(self.N), desc='v as global vector')
        self.add_output('pressure', val=np.zeros(self.N), desc='pseudo pressure as global vector')
        self.add_input('T', val=np.zeros(self.N), desc='T as global vector')

        # global matrices
        dx = self.L_x / N_ex
        dy = self.L_y / N_ey
        self.M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
        self.K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
        self.G_x, self.G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)

        # masks
        self.mask_bound = np.isclose(self.points[0], 0) \
                        + np.isclose(self.points[0], self.L_x) \
                        + np.isclose(self.points[1], 0) \
                        + np.isclose(self.points[1], self.L_y)
        self.mask_dir_p = np.isclose(self.points[0], self.L_x/2) \
                        * np.isclose(self.points[1], self.L_y/2)

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        # load variables
        u = outputs['u']
        v = outputs['v']
        pressure = outputs['pressure']
        T = inputs['T']

        # left-hand-side multiplication of convection velocities, i.e. Re*(u @ C_x + v @ C_y)
        Conv = self.Re * (sparse.tensordot(self.C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                        + sparse.tensordot(self.C_y, v, (1, 0), return_type=sparse.COO).tocsr())

        # system matrix
        self.Sys = self.K + Conv

        # residuals
        residuals['u'] = self.Sys @ u + self.G_x @ pressure
        residuals['v'] = self.Sys @ v + self.G_y @ pressure - self.Gr_over_Re * self.M @ T
        residuals['pressure'] = self.G_x @ u + self.G_y @ v

        # apply DIRICHLET velocity conditions
        mask = np.isclose(self.points[0], 0)  # eastern points
        residuals['u'][mask] = u[mask] - 0
        residuals['v'][mask] = v[mask] - self.options['v_E']
        mask = np.isclose(self.points[0], self.L_x)  # western points
        residuals['u'][mask] = u[mask] - 0
        residuals['v'][mask] = v[mask] - self.options['v_W']
        mask = np.isclose(self.points[1], 0)  # south points
        residuals['u'][mask] = u[mask] - self.options['u_S']
        residuals['v'][mask] = v[mask] - 0
        mask = np.isclose(self.points[1], self.L_y)  # north points
        residuals['u'][mask] = u[mask] - self.options['u_N']
        residuals['v'][mask] = v[mask] - 0

        # apply artificial NEUMANN pressure condition
        residuals['pressure'][self.mask_bound] = self.K[self.mask_bound, :] @ pressure

        # apply reference DIRICHLET pressure condition
        residuals['pressure'][self.mask_dir_p] = pressure[self.mask_dir_p] - 0

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        u = outputs['u']
        v = outputs['v']

        # JACOBI matrices of the predictor equations ignoring the DIRICHLET BC but including the chain rule,
        # i.e. right-hand-side multiplication of the velocities, e.g. Re * C_x @ u
        self.Jac_u_u = self.Sys\
                     + self.Re * sparse.tensordot(self.C_x, u, (2, 0), return_type=sparse.COO).tocsr()
        self.Jac_v_v = self.Sys\
                     + self.Re * sparse.tensordot(self.C_y, v, (2, 0), return_type=sparse.COO).tocsr()
        self.Jac_u_v = self.Re * sparse.tensordot(self.C_y, u, (2, 0), return_type=sparse.COO).tocsr()
        self.Jac_v_u = self.Re * sparse.tensordot(self.C_x, v, (2, 0), return_type=sparse.COO).tocsr()

        self.Jac_u_u.eliminate_zeros()
        self.Jac_u_v.eliminate_zeros()
        self.Jac_v_u.eliminate_zeros()
        self.Jac_v_v.eliminate_zeros()

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_residuals['v'] = - self.Gr_over_Re * self.M @ d_inputs['T']
        d_residuals['v'][self.mask_bound] = 0  # apply DIRICHLET conditions
        d_residuals['v'] *= 5.e-1

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        atol = self.options['mtol']*np.sqrt(self.N)
        solver_type = self.options['solver_type']
        iprecon_type = self.options['iprecon_type']

        # == Jac_velo solver ==
        if solver_type == 'lu':  # LU
            print('NavierStokes LU: Started')
            tStart = time.perf_counter()
            mask = np.hstack((self.mask_bound,)*2)
            Jac_velo = sp_sparse.bmat([[self.Jac_u_u, self.Jac_u_v],
                                       [self.Jac_v_u, self.Jac_v_v]], format='lil')
            Jac_velo[mask, :] = 0  # apply DIRICHLET condition
            Jac_velo[mask, mask] = 1  # ...
            Jac_velo = Jac_velo.tocsc()
            Jac_velo_lu = linalg.splu(Jac_velo)
            print(f'NavierStokes LU: Succeeded in {time.perf_counter()-tStart:0.2f}sec '
                  f'with fill factor {Jac_velo_lu.nnz/Jac_velo.nnz:0.1f}')

            def solve_jac_velo(dr_u, dr_v):
                dr_uv = np.hstack((dr_u, dr_v))
                duv = Jac_velo_lu.solve(dr_uv)
                return np.split(duv, 2)

        elif solver_type == 'qmr':  # quasi minimal residual
            mask = np.hstack((self.mask_bound,)*2)
            Jac_velo = sp_sparse.bmat([[self.Jac_u_u, self.Jac_u_v],
                                       [self.Jac_v_u, self.Jac_v_v]], format='csr')
            A = Jac_velo[~mask, :][:, ~mask]  # principal submatrix

            # preconditioners
            if iprecon_type == 'jac':  # JACOBI
                def iprecon_mv(c):
                    z = c/A.diagonal()
                    return z
                iprecon_LO = linalg.LinearOperator(A.get_shape(), matvec=iprecon_mv, rmatvec=iprecon_mv)
                id_LO = linalg.aslinearoperator(sp_sparse.identity(A.get_shape()[0]))
            elif iprecon_type == 'ilu':  # ILU
                print('NavierStokes inner QMR precon ILU: Started')
                tStart = time.perf_counter()
                A_ilu = linalg.spilu(A.tocsc(),
                                     drop_tol=self.options['drop_tol'],
                                     fill_factor=self.options['fill_factor'])
                print(f'NavierStokes inner QMR precon ILU: Succeeded in {time.perf_counter()-tStart:0.2f}sec '
                      f'with fill factor {A_ilu.nnz/A.nnz:0.1f}')
                iprecon_LO = linalg.LinearOperator((A.get_shape()), matvec=lambda x: A_ilu.solve(x),
                                                                    rmatvec=lambda x: A_ilu.solve(x, 'T'))
                id_LO = linalg.aslinearoperator(sp_sparse.identity(A.get_shape()[0]))
            elif iprecon_type == 'no':
                iprecon_LO = None
                id_LO = None
            else:
                raise ValueError('not a valid inner preconditioner type')

            # solve
            def solve_jac_velo(dr_u, dr_v):
                def qmr_counter(xk):
                    qmr_counter.count += 1
                    # if qmr_counter.count % 10 == 0:
                    #     res = np.linalg.norm(A@xk - dr_uv[~mask])
                    #     print(f'NavierStokes inner QMR: {qmr_counter.count}\t{res}')
                qmr_counter.count = 0
                dr_uv = np.hstack((dr_u, dr_v))
                duv = np.zeros(self.N*2)
                duv[mask] = dr_uv[mask]  # apply DIRICHLET condition
                dr_uv[~mask] -= Jac_velo[~mask, :][:, mask] @ dr_uv[mask]
                duv[~mask], info = linalg.qmr(A, dr_uv[~mask],
                                              tol=0, atol=atol,
                                              M1=iprecon_LO,
                                              M2=id_LO,
                                              callback=qmr_counter)
                if info != 0:
                    raise RuntimeError(f'NavierStokes QMR: Failed to converge in {info} iterations')
                # else:
                #     res = np.linalg.norm(A@duv[~mask] - dr_uv[~mask], ord=np.inf)
                #     print(f'NavierStokes inner QMR: Converged in {qmr_counter.count} iterations with max-norm {res}')
                return np.split(duv, 2)
        else:
            raise ValueError('not a valid solver type')

        # == solve for pressure ==
        # RHS
        b_shur_u, b_shur_v = solve_jac_velo(d_residuals['u'], d_residuals['v'])
        b_shur = -(self.G_x @ b_shur_u + self.G_y @ b_shur_v)
        b_shur[self.mask_bound] = 0  # apply NEUMANN pressure conditions
        b_shur[self.mask_dir_p] = 0  # apply DIRICHLET pressure conditions
        b_shur += d_residuals['pressure']

        # LHS
        def shur_mv(dp):
            # apply gradient
            f_x = self.G_x @ dp
            f_y = self.G_y @ dp
            # apply DIRICHLET velocity condition
            f_x[self.mask_bound] = f_y[self.mask_bound] = 0
            # apply inverse
            f_x, f_y = solve_jac_velo(f_x, f_y)
            # apply divergence
            f = -(self.G_x @ f_x + self.G_y @ f_y)
            # apply artificial NEUMANN pressure condition
            f[self.mask_bound] = self.K[self.mask_bound, :] @ dp
            # apply DIRICHLET pressure condition
            f[self.mask_dir_p] = dp[self.mask_dir_p]
            return f
        shur_LO = linalg.LinearOperator((self.N,)*2, shur_mv)

        # preconditioner
        def precon_mv(c):  # mass precon
            z = c/self.M.diagonal()
            z[self.mask_dir_p] = c[self.mask_dir_p]
            return z
        precon_LO = linalg.LinearOperator((self.N,)*2, precon_mv)

        # solve
        def gmres_counter(res):
            gmres_counter.count += 1
            if gmres_counter.count % 10 == 0:
                print(f'NavierStokes GMRES: {gmres_counter.count}\t{res}')
        gmres_counter.count = 0

        d_outputs['pressure'], info = linalg.gmres(A=shur_LO, b=b_shur, M=precon_LO, x0=d_outputs['pressure'],
                                                   atol=atol, tol=0,
                                                   restart=np.infty,
                                                   callback=gmres_counter, callback_type='pr_norm')
        if info != 0:
            raise RuntimeError(f'NavierStokes GMRES: Failed to converge in {info} iterations')
        else:
            res = np.linalg.norm(shur_LO.matvec(d_outputs['pressure']) - b_shur, ord=np.inf)
            print(f'NavierStokes GMRES: Converged in {gmres_counter.count} iterations with max-norm {res}')

        # == solve for velocities ==
        # RHS
        b_u = -self.G_x @ d_outputs['pressure']
        b_v = -self.G_y @ d_outputs['pressure']
        b_u[self.mask_bound] = 0  # apply DIRICHLET velocity condition
        b_v[self.mask_bound] = 0  # ...
        b_u += d_residuals['u']
        b_v += d_residuals['v']

        d_outputs['u'], d_outputs['v'] = solve_jac_velo(b_u, b_v)
