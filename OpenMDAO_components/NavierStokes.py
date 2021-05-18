import numpy as np
import SEM
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

        # check singularity
        if self.Re == 0 and self.Gr != 0:
            raise ValueError('Cannot have Re == 0 and Gr != 0')
        self.Gr_over_Re = self.Gr/self.Re if self.Re != 0 else 0.

        # declare variables
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
        u = outputs['u']
        v = outputs['v']
        pressure = outputs['pressure']
        T = inputs['T']

        # left-hand-side multiplication of convection velocities, i.e. Re*(u @ C_x + v @ C_y)
        Conv = self.Re * (sparse.tensordot(self.C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                        + sparse.tensordot(self.C_y, v, (1, 0), return_type=sparse.COO).tocsr())

        # system matrix
        self.S = self.K + Conv

        # residuals
        residuals['u'] = self.S @ u + self.G_x @ pressure
        residuals['v'] = self.S @ v + self.G_y @ pressure - self.Gr_over_Re * self.M @ T
        residuals['pressure'] = self.G_x @ u + self.G_y @ v
        res_neumann = self.K @ pressure

        # apply DIRICHLET and artificial NEUMANN pressure conditions
        mask = np.isclose(self.points[0], 0)  # eastern points
        residuals['u'][mask] = u[mask] - 0
        residuals['v'][mask] = v[mask] - self.options['v_E']
        residuals['pressure'][mask] = res_neumann[mask]
        mask = np.isclose(self.points[0], self.L_x)  # western points
        residuals['u'][mask] = u[mask] - 0
        residuals['v'][mask] = v[mask] - self.options['v_W']
        residuals['pressure'][mask] = res_neumann[mask]
        mask = np.isclose(self.points[1], 0)  # south points
        residuals['u'][mask] = u[mask] - self.options['u_S']
        residuals['v'][mask] = v[mask] - 0
        residuals['pressure'][mask] = res_neumann[mask]
        mask = np.isclose(self.points[1], self.L_y)  # north points
        residuals['u'][mask] = u[mask] - self.options['u_N']
        residuals['v'][mask] = v[mask] - 0
        residuals['pressure'][mask] = res_neumann[mask]

        # print('lg(max|res[pressure]|) = ', np.log10(np.max(np.abs(residuals['pressure']))), '; '
        #       'lg(max|res[u]|) = ', np.log10(max(np.max(np.abs(residuals['u'])),
        #                                          np.max(np.abs(residuals['v'])))))

    def linearize(self, inputs, outputs, partials, **kwargs):
        # load variables
        u = outputs['u']
        v = outputs['v']

        # JACOBI matrices of the predictor equations including chain rule,
        # i.e. right-hand-side multiplication of the velocities, e.g. Re * C_x @ u
        jac_u_u = self.S\
                  + self.Re * sparse.tensordot(self.C_x, u, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_v = self.S\
                  + self.Re * sparse.tensordot(self.C_y, v, (2, 0), return_type=sparse.COO).tocsr()
        jac_u_v = self.Re * sparse.tensordot(self.C_y, u, (2, 0), return_type=sparse.COO).tocsr()
        jac_v_u = self.Re * sparse.tensordot(self.C_x, v, (2, 0), return_type=sparse.COO).tocsr()

        # hand over values on the relevant indices; '.getA1()' to convert from np.matrix to np.array
        partials['u', 'u'] = jac_u_u[self.rows, self.cols].getA1()
        partials['u', 'v'] = jac_u_v[self.rows, self.cols].getA1()
        partials['v', 'u'] = jac_v_u[self.rows, self.cols].getA1()
        partials['v', 'v'] = jac_v_v[self.rows, self.cols].getA1()
        partials['u', 'pressure'] = self.G_x[self.rows, self.cols].getA1()
        partials['v', 'pressure'] = self.G_y[self.rows, self.cols].getA1()
        partials['v', 'T'] = - self.Gr_over_Re * self.M[self.rows, self.cols].getA1()
        partials['pressure', 'u'] = self.G_x[self.rows, self.cols].getA1()
        partials['pressure', 'v'] = self.G_y[self.rows, self.cols].getA1()
        partials['pressure', 'pressure'] = self.K[self.rows, self.cols].getA1()

        # apply DIRICHLET and artificial NEUMANN pressure conditions
        # subset of the relevant indices whose rows correspond to boundary points
        mask_rows = np.isclose(self.points[0][self.rows], 0) \
                  + np.isclose(self.points[0][self.rows], self.L_x) \
                  + np.isclose(self.points[1][self.rows], 0) \
                  + np.isclose(self.points[1][self.rows], self.L_y)
        partials['u', 'u'][mask_rows] = 0  # clear rows
        partials['u', 'v'][mask_rows] = 0
        partials['u', 'pressure'][mask_rows] = 0
        partials['v', 'u'][mask_rows] = 0
        partials['v', 'v'][mask_rows] = 0
        partials['v', 'pressure'][mask_rows] = 0
        partials['v', 'T'][mask_rows] = 0
        partials['pressure', 'u'][mask_rows] = 0
        partials['pressure', 'v'][mask_rows] = 0
        partials['pressure', 'pressure'][~mask_rows] = 0
        partials['u', 'u'][(self.cols == self.rows) * mask_rows] = 1  # set 1 on main diagonal
        partials['v', 'v'][(self.cols == self.rows) * mask_rows] = 1
