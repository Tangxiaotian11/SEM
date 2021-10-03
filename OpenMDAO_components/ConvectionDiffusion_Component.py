import numpy as np
import openmdao.api as om


class ConvectionDiffusion_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver_CD', desc='convection-diffusion solver object')
        self.options.declare('solver_NS', desc='NAVIER-STOKES solver object')

    def setup(self):
        self.cd = self.options['solver_CD']
        self.ns = self.options['solver_NS']

        # declare variables
        self.add_output('T_cd', val=np.zeros(self.cd.N), desc='T as CD global vector')
        self.add_input('u_ns', val=np.zeros(self.ns.N), desc='u as NS global vector')
        self.add_input('v_ns', val=np.zeros(self.ns.N), desc='v as NS global vector')

        self.iter_count_solve = 0  # num of get_update calls

    def change_inputs(self, u_ns, v_ns):  # TODO types
        """
        changes the basis of u and v from NS to CD; this is a linear map
        :param u_ns: u as NS global vector
        :param v_ns: v as NS global vector
        :return: (u as CD global vector, v as CD global vector)
        """
        # Note that, get_interpol requires a mesh grid but get_vector passes global vectors
        shape = (2, self.ns._P*self.ns._N_ex+1, self.ns._P*self.ns._N_ey+1)  # shape of mesh grid
        u_call = lambda x, y: self.ns._get_interpol(u_ns, np.reshape((x, y), shape)).flatten()  # mesh grid, as required
        v_call = lambda x, y: self.ns._get_interpol(v_ns, np.reshape((x, y), shape)).flatten()
        u_cd = self.cd._get_vector(f_func=u_call)
        v_cd = self.cd._get_vector(f_func=v_call)
        # u_cd, v_cd = u_ns, v_ns
        return u_cd, v_cd

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['T_cd'] = self.cd._get_residuals(outputs['T_cd'], *self.change_inputs(inputs['u_ns'], inputs['v_ns']))

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.cd._calc_jacobians(outputs['T_cd'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        dT = d_outputs['T_cd'] if 'T_cd' in d_outputs else np.zeros(self.cd.N)
        d_residuals['T_cd'] = self.cd._get_dresiduals(dT, *self.change_inputs(d_inputs['u_ns'], d_inputs['v_ns']))

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_outputs['T_cd'] = self.cd._get_update(d_residuals['T_cd'], dT0=d_outputs['T_cd'])

        self.iter_count_solve += 1

    def solve_nonlinear(self, inputs, outputs):
        outputs['T_cd'] = self.cd._get_solution(*self.change_inputs(inputs['u_ns'], inputs['v_ns']), T0=outputs['T_cd'])
        self.iter_count_solve += 1  # problem is linear so only one get_update call
