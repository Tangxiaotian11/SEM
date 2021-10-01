import numpy as np
import openmdao.api as om


class NS2CD_Component(om.ExplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver_from', desc='NAVIER-STOKES solver object to change from')
        self.options.declare('solver_to', desc='convection-diffusion solver object to change to')

    def setup(self):
        self.ns = self.options['solver_from']
        self.cd = self.options['solver_to']
        self._N_x = self.cd._P*self.ns._N_ex+1
        self._N_y = self.cd._P*self.ns._N_ey+1
        # declare variables
        self.add_input('u', val=np.zeros(self.ns.N), desc='u in NS basis')
        self.add_input('v', val=np.zeros(self.ns.N), desc='v in NS basis')
        self.add_output('u_int', val=np.zeros(self.cd.N), desc='u in CD basis')
        self.add_output('v_int', val=np.zeros(self.cd.N), desc='v in CD basis')

    def compute(self, inputs, outputs, **kwargs):
        """Note that get_vector provides points as global vectors, but get_interpol requires a mesh grid"""
        u_call = lambda x, y: self.ns._get_interpol(inputs['u'],
                                                    np.reshape((x, y), (2, self._N_x, self._N_y))  # mesh grid, as required
                                                    ).flatten()
        v_call = lambda x, y: self.ns._get_interpol(inputs['v'],
                                                    np.reshape((x, y), (2, self._N_x, self._N_y))
                                                    ).flatten()
        outputs['u_int'] = self.cd._get_vector(f_func=u_call)
        outputs['v_int'] = self.cd._get_vector(f_func=v_call)
        # outputs['u_int'] = inputs['u']
        # outputs['v_int'] = inputs['v']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, **kwargs):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        du_call = lambda x, y: self.ns._get_interpol(d_inputs['u'],
                                                     np.reshape((x, y), (2, self._N_x, self._N_y))  # mesh grid, as required
                                                     ).flatten()
        dv_call = lambda x, y: self.ns._get_interpol(d_inputs['v'],
                                                     np.reshape((x, y), (2, self._N_x, self._N_y))
                                                     ).flatten()
        d_outputs['u_int'] += self.cd._get_vector(f_func=du_call)  # ATTENTION, must be added
        d_outputs['v_int'] += self.cd._get_vector(f_func=dv_call)
        # d_outputs['u_int'] += d_inputs['u']
        # d_outputs['v_int'] += d_inputs['v']
