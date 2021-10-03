import numpy as np
import openmdao.api as om


class NavierStokes_Component(om.ImplicitComponent):
    def initialize(self):
        # declare parameters
        self.options.declare('solver_NS', desc='NAVIER-STOKES solver object')
        self.options.declare('solver_CD', desc='convection-diffusion solver object')

    def setup(self):
        self.ns = self.options['solver_NS']
        self.cd = self.options['solver_CD']

        # declare variables
        self.add_input('T_cd', val=np.zeros(self.ns.N), desc='T as CD global vector')
        self.add_output('u_ns', val=np.zeros(self.ns.N), desc='u as NS global vector')
        self.add_output('v_ns', val=np.zeros(self.ns.N), desc='v as NS global vector')
        self.add_output('p_ns', val=np.zeros(self.ns.N), desc='p as NS global vector')

        self.iter_count_solve = 0  # num of get_update calls
        
    def change_inputs(self, T_cd):  # TODO types
        """
        changes the basis of T from CD to NS; this is a linear map
        :param T_cd: T as CD global vector
        :return: T as NS global vector
        """
        # Note that, get_interpol requires a mesh grid but get_vector passes global vectors
        shape = (2, self.cd._P*self.cd._N_ex+1, self.cd._P*self.cd._N_ey+1)  # shape of mesh grid
        T_call = lambda x, y: self.ns._get_interpol(T_cd, np.reshape((x, y), shape)).flatten()  # mesh grid, as required
        T_ns = self.cd._get_vector(f_func=T_call)
        # T_ns = T_cd
        return T_ns

    def apply_nonlinear(self, inputs, outputs, residuals, **kwargs):
        residuals['u_ns'], residuals['v_ns'], residuals['p_ns'] = \
            self.ns._get_residuals(outputs['u_ns'], outputs['v_ns'], outputs['p_ns'], self.change_inputs(inputs['T_cd']))

    def linearize(self, inputs, outputs, partials, **kwargs):
        self.ns._calc_jacobians(outputs['u_ns'], outputs['v_ns'])

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        du = d_outputs['u_ns'] if 'u_ns' in d_outputs else np.zeros(self.ns.N)
        dv = d_outputs['v_ns'] if 'v_ns' in d_outputs else np.zeros(self.ns.N)
        dp = d_outputs['p_ns'] if 'p_ns' in d_outputs else np.zeros(self.ns.N)
        d_residuals['u_ns'], d_residuals['v_ns'], d_residuals['p_ns'] = \
            self.ns._get_dresiduals(du, dv, dp, self.change_inputs(d_inputs['T_cd']))

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        d_outputs['u_ns'], d_outputs['v_ns'], d_outputs['p_ns'] = \
            self.ns._get_update(d_residuals['u_ns'], d_residuals['v_ns'], d_residuals['p_ns'],
                                du0=d_outputs['u_ns'], dv0=d_outputs['v_ns'], dp0=d_outputs['p_ns'])

        self.iter_count_solve += 1

    def solve_nonlinear(self, inputs, outputs):
        outputs['u_ns'], outputs['v_ns'], outputs['p_ns']\
            = self.ns._get_solution(self.change_inputs(inputs['T_cd']), u0=outputs['u_ns'], v0=outputs['v_ns'], p0=outputs['p_ns'])
        self.iter_count_solve += self.ns._k  # get num of get_update calls as num of inner NEWTON iterations
