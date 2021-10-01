import numpy as np
import openmdao.api as om


class CD2NS_Component(om.ExplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver_from', desc='convection-diffusion solver object to change from')
        self.options.declare('solver_to', desc='NAVIER-STOKES solver object to change to')

    def setup(self):
        self.cd = self.options['solver_from']
        self.ns = self.options['solver_to']
        self._N_x = self.ns._P*self.ns._N_ex+1
        self._N_y = self.ns._P*self.ns._N_ey+1
        # declare variables
        self.add_input('T', val=np.zeros(self.cd.N), desc='T in CD basis')
        self.add_output('T_int', val=np.zeros(self.ns.N), desc='T in NS basis')

    def compute(self, inputs, outputs, **kwargs):
        """Note that get_vector provides points as global vectors, but get_interpol requires a mesh grid"""
        T_call = lambda x, y: self.cd._get_interpol(inputs['T'],
                                                    np.reshape((x, y), (2, self._N_x, self._N_y))  # mesh grid, as required
                                                    ).flatten()
        outputs['T_int'] = self.ns._get_vector(f_func=T_call)
        # outputs['T_int'] = inputs['T']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, **kwargs):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        dT_call = lambda x, y: self.cd._get_interpol(d_inputs['T'],
                                                     np.reshape((x, y), (2, self._N_x, self._N_y))  # mesh grid, as required
                                                     ).flatten()
        d_outputs['T_int'] += self.ns._get_vector(f_func=dT_call)  # ATTENTION, MUST BE ADDED
        # d_outputs['T_int'] += d_inputs['T']
