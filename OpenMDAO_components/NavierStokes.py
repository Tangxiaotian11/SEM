import time
import numpy as np
import SEM
import scipy.sparse.linalg as linalg
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import openmdao.api as om


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
        self.options.declare('L_x', types=float, desc='length in x direction')
        self.options.declare('L_y', types=float, desc='length in y direction')
        self.options.declare('Re', types=float, desc='REYNOLDS number')
        self.options.declare('Gr', types=float, default=0., desc='GRASHOF number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('points', types=np.ndarray, desc='points as global vectors [x, y]')
        self.options.declare('u_N', types=float, default=0., desc='tangential velocity at y=L_y')
        self.options.declare('u_S', types=float, default=0., desc='tangential velocity at y=0')
        self.options.declare('v_E', types=float, default=0., desc='tangential velocity at x=0')
        self.options.declare('v_W', types=float, default=0., desc='tangential velocity at x=L_x')

    def setup(self):
        # load parameters
        self.Re = self.options['Re']
        self.Gr = self.options['Gr']
        self.points = self.options['points']
        self.L_x = self.options['L_x']
        self.L_y = self.options['L_y']
        self.P = self.options['P']
        self.N_ex = self.options['N_ex']
        self.N_ey = self.options['N_ey']
        self.N = (self.N_ex*self.P+1)*(self.N_ey*self.P+1)

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
        dx = self.L_x / self.N_ex
        dy = self.L_y / self.N_ey
        self.M = SEM.global_mass_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.M_inv = sp_sparse.diags(1/self.M.diagonal()).tocsr()
        self.K = SEM.global_stiffness_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.G_x, self.G_y = SEM.global_gradient_matrices(self.P, self.N_ex, self.N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(self.P, self.N_ex, self.N_ey, dx, dy)

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

        # JACOBI matrices of the predictor equations including chain rule,
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

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        # == LU decomposition ==
        mask = np.hstack((self.mask_bound,)*2)
        Jac_velo = sp_sparse.bmat([[self.Jac_u_u, self.Jac_u_v],
                                   [self.Jac_v_u, self.Jac_v_v]]).tolil()
        Jac_velo[mask, :] = 0
        Jac_velo[mask, mask] = 1
        print('NavierStokes LU: Started')
        tStart = time.perf_counter()
        Jac_velo_inv = linalg.splu(Jac_velo.tocsc())
        print(f'NavierStokes LU: Succeeded in {time.perf_counter()-tStart:0.2f}sec '
              f'with fill factor {Jac_velo_inv.nnz/Jac_velo.nnz:0.1f}')

        def solve_jac_velo(dr_u, dr_v):
            dr_uv = np.hstack((dr_u, dr_v))
            duv = Jac_velo_inv.solve(dr_uv)
            return np.split(duv, 2)

        # == solve for pressure ==
        # RHS
        b_shur_u, b_shur_v = solve_jac_velo(d_residuals['u'], d_residuals['v'])
        b_shur = -(self.G_x @ b_shur_u + self.G_y @ b_shur_v)
        b_shur[self.mask_bound + self.mask_dir_p] = 0
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

        # solve
        def print_res(res):
            print_res.count += 1
            if print_res.count % 10 == 0:
                print(f'NavierStokes GMRES: {print_res.count}\t{res}')
        print_res.count = 0

        d_outputs['pressure'], info = linalg.gmres(A=shur_LO, b=b_shur, M=self.M_inv, x0=d_outputs['pressure'],
                                                   atol=1e-6, tol=0, restart=np.infty,
                                                   callback=print_res, callback_type='pr_norm')
        if info != 0:
            raise RuntimeError(f'NavierStokes GMRES: Failed to converge in {info} iterations')
        else:
            res = np.linalg.norm(shur_LO.matvec(d_outputs['pressure']) - b_shur, ord=np.inf)
            print(f'NavierStokes GMRES: Converged in {print_res.count} iterations with max-norm {res}')

        # == solve for velocities ==
        # RHS
        b_u = -self.G_x @ d_outputs['pressure']
        b_u[self.mask_bound] = 0
        b_u += d_residuals['u']
        b_v = -self.G_y @ d_outputs['pressure']
        b_v[self.mask_bound] = 0
        b_v += d_residuals['v']

        d_outputs['u'], d_outputs['v'] = solve_jac_velo(b_u, b_v)
