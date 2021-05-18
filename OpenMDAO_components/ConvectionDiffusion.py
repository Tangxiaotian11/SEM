import numpy as np
import SEM
import scipy.sparse as sp_sparse
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

        # declare variables
        self.add_input('u', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='u as global vector')
        self.add_input('v', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='v as global vector')
        self.add_output('T', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='T as global vector')

        # global matrices
        dx = self.L_x / self.N_ex
        dy = self.L_y / self.N_ey
        self.M = SEM.global_mass_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.K = SEM.global_stiffness_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(self.P, self.N_ex, self.N_ey, dx, dy)
        # TODO non-homogeneous NEUMANN conditions

    def setup_partials(self):
        # indices in the global matrices with possible non-zero entries
        Full = SEM.assemble(np.ones((self.N_ex, self.N_ey, self.P+1, self.P+1, self.P+1, self.P+1))).tocoo()
        self.rows, self.cols = Full.row, Full.col
        self.declare_partials(of='*', wrt='*', rows=self.rows, cols=self.cols)

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

        # print('lg(max|res[T]|) = ', np.log10(np.max(np.abs(residuals['T']))))

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        T = outputs['T']

        # right-hand-side multiplication of T, i.e. Pe*(C_x @ T)
        jac_T_u = self.Pe * sparse.tensordot(self.C_x, T, (2, 0), return_type=sparse.COO).tocsr()
        jac_T_v = self.Pe * sparse.tensordot(self.C_y, T, (2, 0), return_type=sparse.COO).tocsr()

        # hand over values on the relevant indices; '.getA1()' to convert from np.matrix to np.array
        partials['T', 'T'] = self.Sys[self.rows, self.cols].getA1()
        partials['T', 'u'] = jac_T_u[self.rows, self.cols].getA1()
        partials['T', 'v'] = jac_T_v[self.rows, self.cols].getA1()

        # apply DIRICHLET conditions
        # subset of the relevant indices whose rows correspond to DIRICHLET boundary points
        mask_rows = np.isclose(self.points[0][self.rows], 0) * (self.options['T_E'] is not None) \
                 + np.isclose(self.points[0][self.rows], self.L_x) * (self.options['T_W'] is not None) \
                 + np.isclose(self.points[1][self.rows], 0) * (self.options['T_S'] is not None) \
                 + np.isclose(self.points[1][self.rows], self.L_y) * (self.options['T_N'] is not None)
        partials['T', 'T'][mask_rows] = 0  # clear rows
        partials['T', 'u'][mask_rows] = 0
        partials['T', 'v'][mask_rows] = 0
        partials['T', 'T'][(self.cols == self.rows) * mask_rows] = 1  # set 1 on main diagonal
