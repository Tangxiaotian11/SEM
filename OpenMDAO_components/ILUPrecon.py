import scipy.sparse.linalg as linalg
from openmdao.solvers.solver import LinearSolver
import time


class ILUPrecon(LinearSolver):
    """
    LinearSolver that uses SciPy spILU
    """

    SOLVER = 'LN: ILU'

    def _declare_options(self):
        super()._declare_options()

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")
        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # Use an assembled jacobian by default
        self.options['assemble_jac'] = True

        # ILU parameter
        self.options.declare('drop_tol', types=float, default=1.e-4, desc='ILU drop tolerance')
        self.options.declare('fill_factor', types=float, default=10., desc='ILU fill Factor')

    def _setup_solvers(self, system, depth):
        super()._setup_solvers(system, depth)
        self._disallow_distrib_solve()

    def _linearize(self):
        if self.options['iprint'] > 0:
            print(f"{self._solver_info.prefix}"
                  f"| precon:{self.SOLVER} Decomposition start")
        tStart = time.perf_counter()
        matrix = self._assembled_jac._int_mtx._matrix
        self._lu = linalg.spilu(matrix, self.options['drop_tol'], self.options['fill_factor'])
        if self.options['iprint'] > 0:
            print(f"{self._solver_info.prefix}"
                  f"| precon:{self.SOLVER} Decomposition successful ({time.perf_counter()-tStart:0.2f}s)")

    def solve(self, vec_names, mode, rel_systems=None):
        if mode != 'fwd':
            raise ValueError('only forward mode implemented')

        system = self._system()
        d_residuals = system._vectors['residual']['linear']
        d_outputs = system._vectors['output']['linear']

        with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
            d_outputs.asarray()[:] = self._lu.solve(d_residuals.asarray())
