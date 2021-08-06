import numpy as np
from Examples.ConvectionDiffusionSolver import ConvectionDiffusionSolver
import openmdao.api as om


class ConvectionDiffusion_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('L_x', types=(float, int), desc='length in x direction')
        self.options.declare('L_y', types=(float, int), desc='length in y direction')
        self.options.declare('Pe', types=(float, int), default=1., desc='PECLET number')
        self.options.declare('P', types=int, desc='polynomial order')
        self.options.declare('N_ex', types=int, desc='num of elements in x direction')
        self.options.declare('N_ey', types=int, desc='num of elements in y direction')
        self.options.declare('T_W', types=(float, int), default=None, desc='T at x=0')
        self.options.declare('T_E', types=(float, int), default=None, desc='T at x=L_x')
        self.options.declare('T_S', types=(float, int), default=None, desc='T at y=0')
        self.options.declare('T_N', types=(float, int), default=None, desc='T at y=L_y')
        self.options.declare('mtol', types=(float, int), default=1e-7, desc='tolerance on root mean square residual')

    def setup(self):
        # initialize backend solver
        self.cd = ConvectionDiffusionSolver(self.options['L_x'], self.options['L_y'], self.options['Pe'],
                                            self.options['P'], self.options['N_ex'], self.options['N_ey'],
                                            self.options['T_W'], self.options['T_E'],
                                            self.options['T_S'], self.options['T_N'],
                                            self.options['mtol'])

        self.iter_count_solve = 0

        # declare variables
        self.add_output('T', val=np.zeros(self.cd.N), desc='T as global vector')
        self.add_input('u', val=np.zeros(self.cd.N), desc='u as global vector')
        self.add_input('v', val=np.zeros(self.cd.N), desc='v as global vector')

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['T'] = self.cd._get_residuals(outputs['T'], inputs['u'], inputs['v'])

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.cd._calc_jacobians(outputs['T'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        if d_outputs._names.__len__() == 0:  # if called by precon, only w.r.t. inputs
            pass
            # d_residuals['T'] = self.cd._get_dresiduals(np.zeros(self.cd.N), d_inputs['u'], d_inputs['v'])
        else:
            d_residuals['T'] = self.cd._get_dresiduals(d_outputs['T'], d_inputs['u'], d_inputs['v'])

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_outputs['T'] = self.cd._get_update(d_residuals['T'], dT0=d_outputs['T'])

        self.iter_count_solve += 1
