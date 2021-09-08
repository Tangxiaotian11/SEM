import numpy as np
import openmdao.api as om


class ConvectionDiffusion_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver', desc='solver object')

    def setup(self):
        self.cd = self.options['solver']

        # declare variables
        self.add_output('T', val=np.zeros(self.cd.N), desc='T as global vector')
        self.add_input('u', val=np.zeros(self.cd.N), desc='u as global vector')
        self.add_input('v', val=np.zeros(self.cd.N), desc='v as global vector')

        self.iter_count_solve = 0  # num of get_update calls

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['T'] = self.cd._get_residuals(outputs['T'], inputs['u'], inputs['v'])

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.cd._calc_jacobians(outputs['T'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        if d_outputs._names.__len__() == 0:  # if called by precon
            pass  # do not modify RHS
        else:
            d_residuals['T'] = self.cd._get_dresiduals(d_outputs['T'], d_inputs['u'], d_inputs['v'])

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_outputs['T'] = self.cd._get_update(d_residuals['T'], dT0=d_outputs['T'])

        self.iter_count_solve += 1

    def solve_nonlinear(self, inputs, outputs):
        outputs['T'] = self.cd._get_solution(inputs['u'], inputs['v'], outputs['T'])
        self.iter_count_solve += 1  # problem is linear so only one get_update call
