import numpy as np
import SEM
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as linalg
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import openmdao.api as om


class ConvectionDiffusion(om.ImplicitComponent):
    """
    Implicit component to solve the steady-state convection-diffusion equation for T(x,y) given the convection
    velocities u(x,y) and v(x,y)
    Pe [u, v]∘∇T = ∇²T ∀(x,y)∈[0,L_x]×[0,L_y]
    with either DIRICHLET or NEUMANN conditions
    T(0,y)   = T_W or ∂ₙT(0,y)   = dT_W ∀y∈[0,L_y]
    T(L_x,y) = T_E or ∂ₙT(L_x,y) = dT_E ∀y∈[0,L_y]
    T(x,0)   = T_S or ∂ₙT(x,0)   = dT_S ∀x∈[0,L_x]
    T(x,L_y) = T_N or ∂ₙT(x,L_y) = dT_N ∀x∈[0,L_x]
    """

    def initialize(self):
        # declare parameters
        self.options.declare('L_x', types=float, desc='length in x direction')
        self.options.declare('L_y', types=float, desc='length in y direction')
        self.options.declare('Pe', types=float, default=1., desc='PECLET number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('points', types=np.ndarray, desc='points as global vectors [x, y]')
        self.options.declare('T_W', types=float, default=None, desc='T at x=0')
        self.options.declare('T_E', types=float, default=None, desc='T at x=L_x')
        self.options.declare('T_S', types=float, default=None, desc='T at y=0')
        self.options.declare('T_N', types=float, default=None, desc='T at y=L_y')
        self.options.declare('dT_W', types=float, default=None, desc='normal derivative of T at x=0')
        self.options.declare('dT_E', types=float, default=None, desc='normal derivative of T at x=L_x')
        self.options.declare('dT_S', types=float, default=None, desc='normal derivative of T at y=0')
        self.options.declare('dT_N', types=float, default=None, desc='normal derivative of T at y=L_y')

    def setup(self):
        # load parameters
        self.L_x = self.options['L_x']
        self.L_y = self.options['L_y']
        self.Pe = self.options['Pe']
        self.points = self.options['points']
        self.P = self.options['P']
        self.N_ex = self.options['N_ex']
        self.N_ey = self.options['N_ey']
        self.N = (self.N_ex*self.P+1)*(self.N_ey*self.P+1)

        # declare variables
        self.add_input('u', val=np.zeros(self.N), desc='u as global vector')
        self.add_input('v', val=np.zeros(self.N), desc='v as global vector')
        self.add_output('T', val=np.zeros(self.N), desc='T as global vector')

        # global matrices
        dx = self.L_x / self.N_ex
        dy = self.L_y / self.N_ey
        self.M = SEM.global_mass_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.K = SEM.global_stiffness_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(self.P, self.N_ex, self.N_ey, dx, dy)
        # TODO non-homogeneous NEUMANN conditions

        # masks
        self.mask_dir = np.isclose(self.points[0], 0) * (self.options['T_E'] is not None) \
                      + np.isclose(self.points[0], self.L_x) * (self.options['T_W'] is not None) \
                      + np.isclose(self.points[1], 0) * (self.options['T_S'] is not None) \
                      + np.isclose(self.points[1], self.L_y) * (self.options['T_N'] is not None)

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        # load variables
        u = inputs['u']
        v = inputs['v']
        T = outputs['T']

        # left-hand-side multiplication of convection velocities, i.e. Pe*(u @ C_x + v @ C_y)
        Conv = self.Pe * (sparse.tensordot(self.C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                        + sparse.tensordot(self.C_y, v, (1, 0), return_type=sparse.COO).tocsr())

        # system matrix
        self.Sys = Conv + self.K

        # residuals
        residuals['T'] = self.Sys @ T

        # apply DIRICHLET/NEUMANN conditions
        # TODO NEUMANN conditions
        if self.options['T_W'] is not None:
            mask = np.isclose(self.points[0], 0)  # western points
            residuals['T'][mask] = T[mask] - self.options['T_W']
        if self.options['T_E'] is not None:
            mask = np.isclose(self.points[0], self.L_x)  # eastern points
            residuals['T'][mask] = T[mask] - self.options['T_E']
        if self.options['T_S'] is not None:
            mask = np.isclose(self.points[1], 0)  # southern points
            residuals['T'][mask] = T[mask] - self.options['T_S']
        if self.options['T_N'] is not None:
            mask = np.isclose(self.points[1], self.L_y)  # northern points
            residuals['T'][mask] = T[mask] - self.options['T_N']

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        T = outputs['T']

        # right-hand-side multiplication of T, i.e. Pe*(C_x @ T)
        self.Jac_T_u = self.Pe * sparse.tensordot(self.C_x, T, (2, 0), return_type=sparse.COO).tocsr()
        self.Jac_T_v = self.Pe * sparse.tensordot(self.C_y, T, (2, 0), return_type=sparse.COO).tocsr()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        def precon_mv(y):  # pure diffusion as preconditioner
            x = np.zeros(self.N)
            x[self.mask_dir] = y[self.mask_dir]
            y[~self.mask_dir] -= self.K[~self.mask_dir, :][:, self.mask_dir] @ y[self.mask_dir]
            x[~self.mask_dir], info = linalg.cg(self.K[~self.mask_dir, :][:, ~self.mask_dir],
                                                y[~self.mask_dir], atol=1e-6, tol=0)
            if info != 0:
                raise RuntimeError(f'ConvectionDiffusion precon CG: Failed to converge in {info} iterations')
            return x
        precon_LO = linalg.LinearOperator((self.N,)*2, precon_mv)

        def jac_mv(dT):
            dr_T = self.Sys @ dT
            dr_T[self.mask_dir] = dT[self.mask_dir]
            return dr_T
        jac_LO = linalg.LinearOperator((self.N,)*2, jac_mv)

        def print_res(res):
            print_res.count += 1
            if print_res.count % 10 == 0:
                print(f'ConvectionDiffusion GMRES: {print_res.count}\t{res}')
        print_res.count = 0

        d_outputs['T'], info = linalg.gmres(A=jac_LO, b=d_residuals['T'], M=precon_LO, x0=d_outputs['T'],
                                            atol=1e-6, tol=0, restart=np.inf,
                                            callback=print_res, callback_type='pr_norm')
        if info != 0:
            raise RuntimeError(f'ConvectionDiffusion GMRES: Failed to converge in {info} iterations')
        else:
            res = np.linalg.norm(jac_LO.matvec(d_outputs['T']) - d_residuals['T'], ord=np.inf)
            print(f'ConvectionDiffusion GMRES: Converged in {print_res.count} iterations with max-norm {res}')
