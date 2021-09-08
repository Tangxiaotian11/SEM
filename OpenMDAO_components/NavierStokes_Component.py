import numpy as np
import openmdao.api as om


class NavierStokes_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver', desc='solver object')

    def setup(self):
        self.ns = self.options['solver']

        # declare variables
        self.add_input('T', val=np.zeros(self.ns.N), desc='T as global vector')
        self.add_output('u', val=np.zeros(self.ns.N), desc='u as global vector')
        self.add_output('v', val=np.zeros(self.ns.N), desc='v as global vector')
        self.add_output('p', val=np.zeros(self.ns.N), desc='p as global vector')

        self.iter_count_solve = 0  # num of get_update calls

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['u'], residuals['v'], residuals['p'] = \
            self.ns._get_residuals(outputs['u'], outputs['v'], outputs['p'], inputs['T'])

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.ns._calc_jacobians(outputs['u'], outputs['v'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        if d_outputs._names.__len__() == 0:  # if called by precon
            pass  # do not modify RHS
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
