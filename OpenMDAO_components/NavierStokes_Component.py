import numpy as np
from Examples.NavierStokesSolver import NavierStokesSolver
import openmdao.api as om


class NavierStokes_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('L_x', types=(float, int), desc='length in x direction')
        self.options.declare('L_y', types=(float, int), desc='length in y direction')
        self.options.declare('Re', types=(float, int), desc='REYNOLDS number')
        self.options.declare('Gr', types=(float, int), default=0, desc='GRASHOF number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('u_N', types=(float, int), default=0, desc='tangential velocity at y=L_y')
        self.options.declare('u_S', types=(float, int), default=0, desc='tangential velocity at y=0')
        self.options.declare('v_E', types=(float, int), default=0, desc='tangential velocity at x=0')
        self.options.declare('v_W', types=(float, int), default=0, desc='tangential velocity at x=L_x')
        self.options.declare('mtol', types=(float, int), default=1e-7, desc='tolerance on root mean square residual')

    def setup(self):
        # initialize backend solver
        self.ns = NavierStokesSolver(self.options['L_x'], self.options['L_y'], self.options['Re'], self.options['Gr'],
                                     self.options['P'], self.options['N_ex'], self.options['N_ey'],
                                     self.options['v_W'], self.options['v_E'], self.options['u_S'], self.options['u_N'],
                                     self.options['mtol'], self.options['mtol'], ['NEWTON_suc'])

        self.iter_count_solve = 0  # num of get_update calls

        # declare variables
        self.add_input('T', val=np.zeros(self.ns.N), desc='T as global vector')
        self.add_output('u', val=np.zeros(self.ns.N), desc='u as global vector')
        self.add_output('v', val=np.zeros(self.ns.N), desc='v as global vector')
        self.add_output('p', val=np.zeros(self.ns.N), desc='p as global vector')

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['u'], residuals['v'], residuals['p'] = \
            self.ns._get_residuals(outputs['u'], outputs['v'], outputs['p'], inputs['T'])

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.ns._calc_jacobians(outputs['u'], outputs['v'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        if d_outputs._names.__len__() == 0:  # if called by precon, only w.r.t. inputs
            pass
            # zero = np.zeros(self.ns.N)
            # d_residuals['u'], d_residuals['v'], d_residuals['p'] = \
            #     self.ns._get_dresiduals(zero, zero, zero, d_inputs['T'])
        else:
            d_residuals['u'], d_residuals['v'], d_residuals['p'] = \
                self.ns._get_dresiduals(d_outputs['u'], d_outputs['v'], d_outputs['p'], d_inputs['T'])

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_outputs['u'], d_outputs['v'], d_outputs['p'] = \
            self.ns._get_update(d_residuals['u'], d_residuals['v'], d_residuals['p'],
                                d_outputs['u'], d_outputs['v'], d_outputs['p'])

        self.iter_count_solve += 1

    def solve_nonlinear(self, inputs, outputs):
        sol = self.ns._get_solution(inputs['T'], outputs['u'], outputs['v'], outputs['p'])
        outputs['u'], outputs['v'], outputs['p'] = sol
        self.iter_count_solve += self.ns._k  # get num of get_update calls as num of inner NEWTON iterations
