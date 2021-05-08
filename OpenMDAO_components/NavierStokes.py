import numpy as np
import SEM
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import openmdao.api as om


class NavierStokes(om.ImplicitComponent):
    """
    Implicit component to solve the steady-state NAVIER-STOKES equation for u(x,y) and v(x,y) given the
    temperature T(x,y) on (x,y)∈[0,L_x]×[0,L_y]
    Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v] + Gr/Re [0, T]
    ∇∘[u, v] = 0
    with no normal flow and tangential DIRICHLET conditions
    v(t,0,y)   = v_W y∈[0,L_y]
    v(t,L_x,y) = v_E y∈[0,L_y]
    u(t,x,0)   = u_S x∈[0,L_x]
    u(t,x,L_y) = u_N x∈[0,L_x]
    The steady-state is found by treating the time dependent equations as fixed-point problem.
    Temporal discretization is performed using the pressure-correction method from KIM-MOIN
    (doi.org/10.1016/0021-9991(85)90148-2).
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
        self.options.declare('dt', types=float, desc='step size in time')
        self.options.declare('u_N', types=float, default=0., desc='tangential velocity at y=L_y')
        self.options.declare('u_S', types=float, default=0., desc='tangential velocity at y=0')
        self.options.declare('v_E', types=float, default=0., desc='tangential velocity at x=0')
        self.options.declare('v_W', types=float, default=0., desc='tangential velocity at x=L_x')

    def setup(self):
        # load parameters
        self.Re = self.options['Re']
        self.Gr = self.options['Gr']
        self.points = self.options['points']
        self.dt = self.options['dt']
        self.L_x = self.options['L_x']
        self.L_y = self.options['L_y']
        self.P = self.options['P']
        self.N_ex = self.options['N_ex']
        self.N_ey = self.options['N_ey']

        # check singularity
        if self.Re == 0 and self.Gr != 0:
            raise ValueError('Cannot have Re == 0 and Gr != 0')
        self.Gr_over_Re = self.Gr/self.Re if self.Re != 0 else 0.

        # declare variables
        self.add_output('u_pre', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='u* as global vector')
        self.add_output('v_pre', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='v* as global vector')
        self.add_output('u', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='u as global vector')
        self.add_output('v', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='v as global vector')
        self.add_output('pressure', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='pseudo pressure as global vector')
        self.add_input('T', val=np.zeros((self.N_ex*self.P+1)*(self.N_ey*self.P+1)), desc='T as global vector')

        # global matrices
        dx = self.L_x / self.N_ex
        dy = self.L_y / self.N_ey
        self.M = SEM.global_mass_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.K = SEM.global_stiffness_matrix(self.P, self.N_ex, self.N_ey, dx, dy)
        self.G_x, self.G_y = SEM.global_gradient_matrices(self.P, self.N_ex, self.N_ey, dx, dy)
        self.C_x, self.C_y = SEM.global_convection_matrices(self.P, self.N_ex, self.N_ey, dx, dy)
        
    def setup_partials(self):
        # indices in the global matrices with possible non-zero entries
        Full = SEM.assemble(np.ones((self.N_ex, self.N_ey, self.P+1, self.P+1, self.P+1, self.P+1))).tocoo()
        self.rows, self.cols = Full.row, Full.col
        self.declare_partials(of='*', wrt='*', rows=self.rows, cols=self.cols)

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        # load variables
        u_pre = outputs['u_pre']
        v_pre = outputs['v_pre']
        u = outputs['u']
        v = outputs['v']
        pressure = outputs['pressure']
        T = inputs['T']

        # left-hand-side multiplication of convection velocities, i.e. Re*(u @ C_x + v @ C_y)
        Conv_pre = self.Re * (sparse.tensordot(self.C_x, u_pre, (1, 0), return_type=sparse.COO).tocsr()
                            + sparse.tensordot(self.C_y, v_pre, (1, 0), return_type=sparse.COO).tocsr())
        Conv_stat = self.Re * (sparse.tensordot(self.C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                             + sparse.tensordot(self.C_y, v, (1, 0), return_type=sparse.COO).tocsr())

        # system matrices for the predictor
        self.LHS = 1/self.dt * self.M + 0.5*self.K + 0.5*Conv_pre
        self.RHS = -1/self.dt * self.M + 0.5*self.K + 0.5*Conv_stat

        # residuals
        residuals['u_pre'] = self.LHS @ u_pre + self.RHS @ u
        residuals['v_pre'] = self.LHS @ v_pre + self.RHS @ v - self.Gr_over_Re * self.M @ T
        residuals['pressure'] = self.K @ pressure + 1/self.dt * (self.G_x @ u_pre + self.G_y @ v_pre)
        residuals['u'] = 1/self.dt * self.M @ (u - u_pre) + self.G_x @ pressure  # =(u - u_corr)/dt
        residuals['v'] = 1/self.dt * self.M @ (v - v_pre) + self.G_y @ pressure

        # apply DIRICHLET conditions
        ind = np.isclose(self.points[0], 0)  # eastern points
        residuals['u_pre'][ind] = u_pre[ind] - 0
        residuals['v_pre'][ind] = v_pre[ind] - self.options['v_E']
        ind = np.isclose(self.points[0], self.L_x)  # western points
        residuals['u_pre'][ind] = u_pre[ind] - 0
        residuals['v_pre'][ind] = v_pre[ind] - self.options['v_W']
        ind = np.isclose(self.points[1], 0)  # south points
        residuals['u_pre'][ind] = u_pre[ind] - self.options['u_S']
        residuals['v_pre'][ind] = v_pre[ind] - 0
        ind = np.isclose(self.points[1], self.L_y)  # north points
        residuals['u_pre'][ind] = u_pre[ind] - self.options['u_N']
        residuals['v_pre'][ind] = v_pre[ind] - 0

        # print('lg(max|res[u_pre]|) = ', np.log10(max(np.max(np.abs(residuals['u_pre'])),
        #                                              np.max(np.abs(residuals['v_pre'])))), '; '
        #       'lg(max|res[u]|) = ', np.log10(max(np.max(np.abs(residuals['u'])),
        #                                          np.max(np.abs(residuals['v'])))))

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        u_pre = outputs['u_pre']
        v_pre = outputs['v_pre']
        u = outputs['u']
        v = outputs['v']
        T = inputs['T']

        # JACOBI matrices of the predictor equations including chain rule,
        # i.e. right-hand-side multiplication of the velocities, e.g. Re * C_x @ u
        jac_u_pre_u_pre = self.LHS\
                        + 0.5*self.Re * sparse.tensordot(self.C_x, u_pre, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_pre_v_pre = self.LHS\
                        + 0.5*self.Re * sparse.tensordot(self.C_y, v_pre, (2, 0), return_type=sparse.COO).tocsr()
        jac_u_pre_v_pre = 0.5*self.Re * sparse.tensordot(self.C_y, u_pre, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_pre_u_pre = 0.5*self.Re * sparse.tensordot(self.C_x, v_pre, (2, 0), return_type=sparse.COO).tocsr()
        jac_u_pre_u = self.RHS\
                    + 0.5*self.Re * sparse.tensordot(self.C_x, u, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_pre_v = self.RHS\
                    + 0.5*self.Re * sparse.tensordot(self.C_y, v, (2, 0), return_type=sparse.COO).tocsr()
        jac_u_pre_v = 0.5*self.Re * sparse.tensordot(self.C_y, u, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_pre_u = 0.5*self.Re * sparse.tensordot(self.C_x, v, (2, 0), return_type=sparse.COO).tocsr()

        # hand over values on the relevant indices; '.getA1()' to convert from np.matrix to np.array
        partials['u_pre', 'u_pre'] = jac_u_pre_u_pre[self.rows, self.cols].getA1()
        partials['u_pre', 'v_pre'] = jac_u_pre_v_pre[self.rows, self.cols].getA1()
        partials['u_pre', 'u'] = jac_u_pre_u[self.rows, self.cols].getA1()
        partials['u_pre', 'v'] = jac_u_pre_v[self.rows, self.cols].getA1()
        partials['v_pre', 'u_pre'] = jac_v_pre_u_pre[self.rows, self.cols].getA1()
        partials['v_pre', 'v_pre'] = jac_v_pre_v_pre[self.rows, self.cols].getA1()
        partials['v_pre', 'u'] = jac_v_pre_u[self.rows, self.cols].getA1()
        partials['v_pre', 'v'] = jac_v_pre_v[self.rows, self.cols].getA1()
        partials['pressure', 'pressure'] = self.K[self.rows, self.cols].getA1()
        partials['pressure', 'u_pre'] = 1/self.dt * self.G_x[self.rows, self.cols].getA1()
        partials['pressure', 'v_pre'] = 1/self.dt * self.G_y[self.rows, self.cols].getA1()
        partials['u', 'u'] = 1/self.dt * self.M[self.rows, self.cols].getA1()
        partials['v', 'v'] = 1/self.dt * self.M[self.rows, self.cols].getA1()
        partials['u', 'u_pre'] = -1/self.dt * self.M[self.rows, self.cols].getA1()
        partials['v', 'v_pre'] = -1/self.dt * self.M[self.rows, self.cols].getA1()
        partials['u', 'pressure'] = self.G_x[self.rows, self.cols].getA1()
        partials['v', 'pressure'] = self.G_y[self.rows, self.cols].getA1()
        partials['v_pre', 'T'] = - self.Gr_over_Re * self.M[self.rows, self.cols].getA1()

        # apply DIRICHLET conditions
        # subset of the relevant indices whose rows correspond to boundary points
        ind_rows = np.isclose(self.points[0][self.rows], 0) \
                 + np.isclose(self.points[0][self.rows], self.L_x) \
                 + np.isclose(self.points[1][self.rows], 0) \
                 + np.isclose(self.points[1][self.rows], self.L_y)
        partials['u_pre', 'u_pre'][ind_rows] = 0  # clear rows
        partials['u_pre', 'v_pre'][ind_rows] = 0
        partials['u_pre', 'u'][ind_rows] = 0
        partials['u_pre', 'v'][ind_rows] = 0
        partials['v_pre', 'u_pre'][ind_rows] = 0
        partials['v_pre', 'v_pre'][ind_rows] = 0
        partials['v_pre', 'u'][ind_rows] = 0
        partials['v_pre', 'v'][ind_rows] = 0
        partials['v_pre', 'T'][ind_rows] = 0
        partials['u_pre', 'u_pre'][(self.cols == self.rows) * ind_rows] = 1  # set 1 on main diagonal
        partials['v_pre', 'v_pre'][(self.cols == self.rows) * ind_rows] = 1
