import sys, os
sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../..')
import numpy as np
import SEM
from Examples.ConvectionDiffusionSolver import ConvectionDiffusionSolver
from Examples.NavierStokesSolver import NavierStokesSolver
from OpenMDAO_components.ConvectionDiffusion_Component import ConvectionDiffusion_Component
from OpenMDAO_components.NavierStokes_Component import NavierStokes_Component
import openmdao.api as om
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()


def run(log=False, save=True, mode='JNK', backend='PETSc',
        L_x=1., L_y=1., Re=1.e2, Ra=1.e3, Pr=0.71,
        P=4, Ne=8,
        mtol_nonlin=1e-8, AGi=8, AGr=0.8, AGc=0.2,
        mtol_gmres=1e-10, restart=20,
        mtol_internal=1e-13):

        N_ex = Ne
        N_ey = Ne
        N = (N_ex*P+1)*(N_ey*P+1)
        DOF = 4*N
        atol_gmres = mtol_gmres*np.sqrt(DOF)
        atol_nonlin = mtol_nonlin*np.sqrt(DOF)

        title = f"Boussinesq{mode}_{Re:.1e}~{Ra:.1e}~{Pr}_{P}~{N_ex}~{N_ey}_{mtol_nonlin:.0e}~{AGi}~{AGr}~{AGc}_{mtol_gmres:.0e}~{restart}_{mtol_internal:.0e}"
        if rank == 0:
            print(title)

        # initialize backend solvers
        if rank == 0:
            cd = ConvectionDiffusionSolver(L_x=L_x, L_y=L_y, Pe=Re*Pr,
                                           P=P, N_ex=N_ex, N_ey=N_ey,
                                           T_W=0.5, T_E=-0.5,
                                           mtol=mtol_internal)
            ns = None
        if rank == 1:
            ns = NavierStokesSolver(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr,
                                    P=P, N_ex=N_ex, N_ey=N_ey,
                                    mtol=mtol_internal, mtol_newton=mtol_internal, iprint=['NEWTON_suc'])
            cd = None

        # initialize OpenMDAO solver
        prob = om.Problem()
        model = prob.model
        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('ConvectionDiffusion', ConvectionDiffusion_Component(solver=cd))
        parallel.add_subsystem('NavierStokes', NavierStokes_Component(solver=ns))
        parallel.connect('ConvectionDiffusion.T', 'NavierStokes.T')
        parallel.connect('NavierStokes.u', 'ConvectionDiffusion.u')
        parallel.connect('NavierStokes.v', 'ConvectionDiffusion.v')

        if mode == 'GS':
            model.nonlinear_solver = om.NonlinearBlockGS(iprint=2, use_apply_nonlinear=True, maxiter=1000, atol=atol_nonlin, rtol=0)  # runs as Jac
        else:
            parallel.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, max_sub_solves=0, maxiter=1000, atol=atol_nonlin, rtol=0)
            parallel.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=AGi, rho=AGr, c=AGc)
            if mode == 'NJ':
                parallel.linear_solver = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)
            elif mode == 'JNK':
                if backend == 'SciPy':
                    parallel.linear_solver = om.ScipyKrylov(iprint=2, atol=atol_gmres, rtol=0, restart=restart, maxiter=5000)
                elif backend == 'PETSc':
                    parallel.linear_solver = om.PETScKrylov(iprint=2, atol=atol_gmres, rtol=0, restart=restart, maxiter=5000,
                                                            ksp_type='gmres', precon_side='left')
                else:
                    raise ValueError('Unknown backend')
                parallel.linear_solver.precon = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)
            else:
                raise ValueError('Unknown method')
        prob.setup()

        # preprocessing
        if rank == 0:
            # grid
            points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
            # initial guess
            T = np.zeros(N)
            T[np.isclose(points[0], 0)] = 0.5
            T[np.isclose(points[0], L_x)] = -0.5
            # hand over guess
            prob['parallel.ConvectionDiffusion.T'] = T
        if rank == 1:
            pass

        # solve
        if log:
            try:
                sys.stdout = open(f'Boussinesq_study/{title}.log', 'w')
            except FileNotFoundError:
                os.mkdir("Boussinesq_study")
                sys.stdout = open(f'Boussinesq_study/{title}.log', 'w')
            prob.run_model()
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        else:
            prob.run_model()

        # Result
        if rank == 0:
            T_e = SEM.scatter(prob['parallel.ConvectionDiffusion.T'], P, N_ex, N_ey)
            results = T_e
            iters = model.parallel.ConvectionDiffusion.iter_count_solve
        if rank == 1:
            u_e = SEM.scatter(prob['parallel.NavierStokes.u'], P, N_ex, N_ey)
            v_e = SEM.scatter(prob['parallel.NavierStokes.v'], P, N_ex, N_ey)
            results = [u_e, v_e]
            iters = model.parallel.NavierStokes.iter_count_solve
        # gather to rank 0
        results = MPI.COMM_WORLD.gather(results, root=0)
        iters = MPI.COMM_WORLD.gather(iters, root=0)

        # post-processing
        if rank == 0:
            points_e = SEM.element_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
            iter_nonlin = model.parallel.nonlinear_solver._iter_count
            iter = [iters[0], iters[1], iter_nonlin]
            print(iter)
            # save
            if save:
                try:
                    np.savez("Boussinesq_study/"+title, points_e, *results[1], results[0], iter)
                except FileNotFoundError:
                    os.mkdir("Boussinesq_study")
                    np.savez("Boussinesq_study/"+title, points_e, *results[1], results[0], iter)


if __name__ == "__main__":
    save = True
    log = False
    mode = 'JNK'
    P_set = [4]
    Ne_set = [8]
    Re_set = [1.e2]
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
            log = bool(sys.argv[i+1])
        if arg == '-backend':
            backend = sys.argv[i+1]

    for Re in Re_set:
        for Ra in Ra_set:
            for P in P_set:
                for Ne in Ne_set:
                    run(mode=mode, log=log, save=save, Re=Re, Ra=Ra, P=int(P), Ne=int(Ne), backend=backend)
