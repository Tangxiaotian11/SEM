import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
from Solvers import SEM
from Solvers.ConvectionDiffusion_Solver import ConvectionDiffusionSolver
from Solvers.NavierStokes_Solver import NavierStokesSolver
from OpenMDAO.ConvectionDiffusion_Component import ConvectionDiffusion_Component
from OpenMDAO.NavierStokes_Component import NavierStokes_Component
import openmdao.api as om
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


class Logger(object):
    def __init__(self, file):
        self.terminal = sys.__stdout__
        self.log = open(file, "w")

    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)

    def flush(self): pass


def run(log=False, save=True,
        L_x=1., L_y=1.,
        Re=1.e3, Ra=1.e3, Pr=0.71,
        P=4, N_e=8,
        mode='JNK', backend='PETSc',
        mtol_nonlin=1e-10, AGi=8, AGr=0.8, AGc=0.2,
        mtol_gmres=1e-13, restart=20,
        mtol_internal=1e-13):

    title = f"Boussinesq{mode}_{Re:.1e}~{Ra:.1e}~{Pr}_{P}~{N_e}_"
    if mode == 'GS':
        title += f"{mtol_nonlin:.0e}_{mtol_internal:.0e}"
    elif mode == 'NJ':
        title += f"{mtol_nonlin:.0e}~{AGi}~{AGr}~{AGc}_{mtol_internal:.0e}"
    elif mode == 'JNK':
        title += f"{mtol_nonlin:.0e}_{mtol_gmres:.0e}~{restart}_{mtol_internal:.0e}"
    else:
        raise RuntimeError('Unknown method')

    if rank == 0:
        print(title)

    # initialize backend solvers
    cd = ConvectionDiffusionSolver(L_x=L_x, L_y=L_y, Pe=Re*Pr,
                                   P=P, N_ex=int(N_e/2), N_ey=int(N_e/2),
                                   T_W=0.5, T_E=-0.5,
                                   mtol=mtol_internal)
    ns = NavierStokesSolver(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr,
                            P=P, N_ex=N_e, N_ey=N_e,
                            mtol=mtol_internal, mtol_newton=mtol_internal, iprint=['NEWTON_suc'])

    DOF = 3*ns.N + 1*cd.N  # T,u,v,p
    atol_gmres = mtol_gmres * np.sqrt(DOF)
    atol_nonlin = mtol_nonlin * np.sqrt(DOF)

    # initialize OpenMDAO solver
    prob = om.Problem()
    model = prob.model
    pg = model.add_subsystem('PG', om.ParallelGroup())
    pg.add_subsystem('ConvectionDiffusion', ConvectionDiffusion_Component(solver_CD=cd, solver_NS=ns))  # rank 0
    pg.add_subsystem('NavierStokes', NavierStokes_Component(solver_CD=cd, solver_NS=ns))  # rank 1
    pg.connect('ConvectionDiffusion.T_cd', 'NavierStokes.T_cd')
    pg.connect('NavierStokes.u_ns', 'ConvectionDiffusion.u_ns')
    pg.connect('NavierStokes.v_ns', 'ConvectionDiffusion.v_ns')

    if mode == 'GS':  # requires change in nonlinear_block_gs.py
        pg.nonlinear_solver = om.NonlinearBlockGS(iprint=2, err_on_non_converge=True,
                                                  use_apply_nonlinear=True, maxiter=1000,
                                                  atol=atol_nonlin, rtol=0)
    else:  # requires change in linear_block_jac.py
        pg.nonlinear_solver = om.NewtonSolver(iprint=2, err_on_non_converge=True,
                                              solve_subsystems=True, max_sub_solves=0,
                                              atol=atol_nonlin, rtol=0)
        if mode == 'NJ':
            pg.nonlinear_solver.options['maxiter'] = 1000
            pg.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=AGi, rho=AGr, c=AGc)
            pg.linear_solver = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)
        else:
            pg.nonlinear_solver.options['maxiter'] = 100
            if backend == 'SciPy':
                pg.linear_solver = om.ScipyKrylov(iprint=2, err_on_non_converge=True,
                                                  atol=atol_gmres, rtol=0, restart=restart, maxiter=5000)
            elif backend == 'PETSc':
                pg.linear_solver = om.PETScKrylov(iprint=2, err_on_non_converge=True,
                                                  atol=atol_gmres, rtol=0, restart=restart, maxiter=5000,
                                                  ksp_type='gmres', precon_side='left')
            else:
                raise ValueError('Unknown backend')
            pg.linear_solver.precon = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)
    prob.setup()

    # solve
    if log:
        if rank == 0:
            try:
                sys.stdout = Logger(f'Boussinesq_study/{title}.log')
            except FileNotFoundError:
                os.mkdir("Boussinesq_study")
                sys.stdout = Logger(f'Boussinesq_study/{title}.log')
        prob.run_model()
        sys.stdout = sys.__stdout__
    else:
        prob.run_model()

    # Result
    if rank == 0:
        T_e = SEM.scatter(prob['PG.ConvectionDiffusion.T_cd'], cd._P, cd._N_ex, cd._N_ey)
        results = T_e
        iters = pg.ConvectionDiffusion.iter_count_solve
    if rank == 1:
        u_e = SEM.scatter(prob['PG.NavierStokes.u_ns'], ns._P, ns._N_ex, ns._N_ey)
        v_e = SEM.scatter(prob['PG.NavierStokes.v_ns'], ns._P, ns._N_ex, ns._N_ey)
        results = [u_e, v_e]
        iters = pg.NavierStokes.iter_count_solve
    # gather to rank 0
    results = MPI.COMM_WORLD.gather(results, root=0)
    iters = MPI.COMM_WORLD.gather(iters, root=0)

    # post-processing
    if rank == 0:
        iter_nonlin = pg.nonlinear_solver._iter_count
        iter = [iters[0], iters[1], iter_nonlin]
        print(iter)
        # save
        if save:
            try:
                np.savez("Boussinesq_study/"+title, results[0], *results[1], iter)
            except FileNotFoundError:
                os.mkdir("Boussinesq_study")
                np.savez("Boussinesq_study/"+title, results[0], *results[1], iter)


if __name__ == "__main__":
    save = True
    log = False
    mode = 'JNK'
    P_set = [4]
    Ne_set = [8]
    Re_set = [1.e3]
    Ra_set = [1.e3]
    backend = 'PETSc'

    for i, arg in enumerate(sys.argv):
        if arg == '-P':
            P_set = np.array(sys.argv[i+1].split(','), dtype=int)
        if arg == '-Ne':
            Ne_set = np.array(sys.argv[i+1].split(','), dtype=int)
        if arg == '-Re':
            Re_set = np.array(sys.argv[i+1].split(','), dtype=float)
        if arg == '-Ra':
            Ra_set = np.array(sys.argv[i+1].split(','), dtype=float)
        if arg == '-mode':
            mode = sys.argv[i+1]
        if arg == '-log':
            log = eval(sys.argv[i+1])
        if arg == '-save':
            save = eval(sys.argv[i+1])
        if arg == '-backend':
            backend = sys.argv[i+1]

    for Re in Re_set:
        for Ra in Ra_set:
            for P in P_set:
                for Ne in Ne_set:
                    run(mode=mode, log=log, save=save, Re=Re, Ra=Ra, P=int(P), N_e=int(Ne), backend=backend)
