import numpy as np
import SEM
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import openmdao.api as om


class ConvectionDiffusion(om.ImplicitComponent):
    """
    Implicit component to solve the steady-state convection-diffusion equation for T(x,y) given the convection
    velocities u(x,y) and v(x,y) on (x,y)∈[0,L_x]×[0,L_y]
    Re [u, v]∘∇T = 1/Pr ∇²T
    with either DIRICHLET or NEUMANN conditions
    T(0,y)   = T_W(y) or ∂ₙT(0,y)   = dT_W(y) ∀y∈[0,L_y]
    T(L_x,y) = T_E(y) or ∂ₙT(L_x,y) = dT_E(y) ∀y∈[0,L_y]
    T(x,0)   = T_S(x) or ∂ₙT(x,0)   = dT_S(x) ∀x∈[0,L_x]
    T(x,L_y) = T_N(x) or ∂ₙT(x,L_y) = dT_N(x) ∀x∈[0,L_x]
    """
    # T_W = -0.5, T_E = 0.5, dT_S = dT_N = 0 only for now

    def initialize(self):
        # declare parameters
        self.options.declare('L_x', types=float, desc='length in x direction')
        self.options.declare('L_y', types=float, desc='length in y direction')
        self.options.declare('Re', types=float, desc='REYNOLDS number')
        self.options.declare('Pr', types=float, default=1., desc='PRANDTL number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('points', types=np.ndarray, desc='points as global vectors (x, y)')
        # TODO DIRICHLET conditions
        # TODO NEUMANN conditions

    def setup(self):
        # load parameters
        self.L_x = self.options['L_x']
        self.L_y = self.options['L_y']
        self.Re = self.options['Re']
        self.Pr = self.options['Pr']
        self.points = self.options['points']
        P = self.options['P']
        N_ex = self.options['N_ex']
        N_ey = self.options['N_ey']

        # check singularity
        if self.Pr == 0:
            raise ValueError('Cannot have Pr == 0')

        # declare variables
        self.add_input('u', val=np.zeros((N_ex*P+1)*(N_ey*P+1)), desc='u as global vector')
        self.add_input('v', val=np.zeros((N_ex*P+1)*(N_ey*P+1)), desc='v as global vector')
        self.add_output('T', val=np.zeros((N_ex*P+1)*(N_ey*P+1)), desc='T as global vector')

        # list of all indices in the global matrices with possible non-zero entries
        Full = SEM.assemble(np.ones((N_ex, N_ey, P+1, P+1, P+1, P+1))).tocoo()
        self.rows, self.cols = Full.row, Full.col
        del Full
        self.declare_partials(of='*', wrt='*', rows=self.rows, cols=self.cols)

        # global matrices
        dx = self.L_x / N_ex
        dy = self.L_x / N_ey
        self.M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
        self.K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)
        # TODO non-homogeneous NEUMANN conditions

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        # load variables
        u = inputs['u']
        v = inputs['v']
        T = outputs['T']

        # left-hand-side multiplication of convection velocities, i.e. Re*(u @ C_x + v @ C_y)
        Conv = self.Re * (sparse.tensordot(self.C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                        + sparse.tensordot(self.C_y, v, (1, 0), return_type=sparse.COO).tocsr())

        # system matrix
        self.Sys = Conv + 1/self.Pr * self.K

        # residuals
        residuals['T'] = self.Sys @ T

        # apply DIRICHLET conditions  TODO DIRICHLET conditions
        ind = np.isclose(self.points[0], 0)  # western points
        residuals['T'][ind] = T[ind] - -0.5
        ind = np.isclose(self.points[0], self.L_x)  # eastern points
        residuals['T'][ind] = T[ind] - 0.5

        # print('lg(max|res[T]|) = ', np.log10(np.max(np.abs(residuals['T']))))

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        T = outputs['T']

        # right-hand-side multiplication of T, i.e. Re C_x @ T
        jac_T_u = self.Re * sparse.tensordot(self.C_x, T, (2, 0), return_type=sparse.COO).tocsr()
        jac_T_v = self.Re * sparse.tensordot(self.C_y, T, (2, 0), return_type=sparse.COO).tocsr()

        # hand over values on the relevant indices; '.getA1()' to convert from np.matrix to np.array
        partials['T', 'T'] = self.Sys[self.rows, self.cols].getA1()
        partials['T', 'u'] = jac_T_u[self.rows, self.cols].getA1()
        partials['T', 'v'] = jac_T_v[self.rows, self.cols].getA1()

        # apply DIRICHLET conditions
        # subset of the relevant indices whose rows correspond to eastern or western points
        ind_rows = np.isclose(self.points[0][self.rows], 0) \
                 + np.isclose(self.points[0][self.rows], self.L_x)
        partials['T', 'T'][ind_rows] = 0  # clear rows
        partials['T', 'u'][ind_rows] = 0
        partials['T', 'v'][ind_rows] = 0
        partials['T', 'T'][(self.cols == self.rows) * ind_rows] = 1  # set 1 on main diagonal
