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
import sys


def run(log=False, save=True, mode='',
        L_x=1., L_y=1., Re=1.e2, Ra=1.e3, Pr=0.71,
        P=4, Ne=8,
        mtol_nonlin=1e-8, AGi=8, AGr=0.8, AGc=0.2,
        mtol_gmres=1e-10, restart=20,
        mtol_internal=1e-13):

        N_ex = Ne
        N_ey = Ne
        N = (N_ex*P+1)*(N_ey*P+1)
        atol_gmres = mtol_gmres*np.sqrt(N*4)
        atol_nonlin = mtol_nonlin*np.sqrt(N*4)

        title = f"Boussinesq{mode}_{Re:.1e}~{Ra:.1e}~{Pr}_{P}~{N_ex}~{N_ey}_{mtol_nonlin:.0e}~{AGi}~{AGr}~{AGc}_{mtol_gmres:.0e}~{restart}_{mtol_internal:.0e}"
        print(title)

        # grid
        points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
        points_e = SEM.element_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)

        # initialize global coefficients vectors
        u = np.zeros(N)
        v = np.zeros(N)
        T = np.zeros(N)
        T[np.isclose(points[0], 0)] = 0.5  # initial guess
        T[np.isclose(points[0], L_x)] = -0.5

        # initialize backend solvers
        cd = ConvectionDiffusionSolver(L_x=L_x, L_y=L_y, Pe=Re*Pr, P=P, N_ex=N_ex, N_ey=N_ey, T_W=0.5, T_E=-0.5,
                                       mtol=mtol_internal)
        ns = NavierStokesSolver(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr, P=P, N_ex=N_ex, N_ey=N_ey,
                                mtol=mtol_internal, mtol_newton=mtol_internal, iprint=['NEWTON_suc'])

        # initialize OpenMDAO solver
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('ConvectionDiffusion', ConvectionDiffusion_Component(solver=cd))
        model.add_subsystem('NavierStokes', NavierStokes_Component(solver=ns))
        model.connect('ConvectionDiffusion.T', 'NavierStokes.T')
        model.connect('NavierStokes.u', 'ConvectionDiffusion.u')
        model.connect('NavierStokes.v', 'ConvectionDiffusion.v')
        if mode == 'JN' or mode == 'IN':
            model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, max_sub_solves=0, maxiter=1000, atol=atol_nonlin, rtol=0)
            model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=AGi, rho=AGr, c=AGc)
            if mode == 'IN':
                model.linear_solver = om.LinearRunOnce()
            else:
                model.linear_solver = om.ScipyKrylov(iprint=2, atol=atol_gmres, rtol=0, restart=restart, maxiter=5000)
                model.linear_solver.precon = om.LinearBlockJac(iprint=-1, maxiter=1)
        elif mode == 'GS':
            model.nonlinear_solver = om.NonlinearBlockGS(iprint=2, use_apply_nonlinear=True, maxiter=1000, atol=atol_nonlin, rtol=0)
        prob.setup()

        # solve
        prob['NavierStokes.u'] = u  # hand over guess
        prob['NavierStokes.v'] = v
        prob['ConvectionDiffusion.T'] = T

        if log:
            sys.stdout = open(f'Boussinesq_study/{title}.log', 'w')
            prob.run_model()
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        else:
            prob.run_model()

        # Result
        iter_CD = model.ConvectionDiffusion.iter_count_solve
        iter_NS = model.NavierStokes.iter_count_solve
        iter_nonlin = model.nonlinear_solver._iter_count
        iter = [iter_CD, iter_NS, iter_nonlin]
        u = prob['NavierStokes.u']
        v = prob['NavierStokes.v']
        T = prob['ConvectionDiffusion.T']
        u_e = SEM.scatter(u, P, N_ex, N_ey)
        v_e = SEM.scatter(v, P, N_ex, N_ey)
        T_e = SEM.scatter(T, P, N_ex, N_ey)
        # save
        if save:
            try:
                np.savez("Boussinesq_study/"+title, points_e, u_e, v_e, T_e, iter)
            except FileNotFoundError:
                os.mkdir("Boussinesq_study")
                np.savez("Boussinesq_study/"+title, points_e, u_e, v_e, T_e, iter)


if __name__ == "__main__":
    save = True
    log = False
    mode = 'JN'
    P_set = [4]
    Ne_set = [8]
    Re_set = [1.e2]
    Ra_set = [1.e3]

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

    for Re in Re_set:
        for Ra in Ra_set:
            for P in P_set:
                for Ne in Ne_set:
                    run(mode=mode, log=log, save=save, Re=Re, Ra=Ra, P=int(P), Ne=int(Ne))
