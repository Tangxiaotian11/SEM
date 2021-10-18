import numpy as np
from Solvers.ConvectionDiffusion_Solver import ConvectionDiffusionSolver
from Solvers.NavierStokes_Solver import NavierStokesSolver
from OpenMDAO.ConvectionDiffusion_Component import ConvectionDiffusion_Component
from OpenMDAO.NavierStokes_Component import NavierStokes_Component
import openmdao.api as om


def run(points_plot, L_x, L_y,
        Re=1.e3, Ra=1.e3, Pr=0.71,
        P_cd=4, N_ex_cd=8, N_ey_cd=8,
        P_ns=4, N_ex_ns=8, N_ey_ns=8,
        mode='JNK',
        mtol_nonlin=1e-9, AGi=8, AGr=0.8, AGc=0.2,
        mtol_gmres=1e-10, restart=20,
        mtol_internal=1e-13):

    # initialize backend solvers
    cd = ConvectionDiffusionSolver(L_x=L_x, L_y=L_y, Pe=Re*Pr,
                                   P=P_cd, N_ex=N_ex_cd, N_ey=N_ey_cd,
                                   T_W=0.5, T_E=-0.5,
                                   mtol=mtol_internal)
    ns = NavierStokesSolver(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr,
                            P=P_ns, N_ex=N_ex_ns, N_ey=N_ey_ns,
                            mtol=mtol_internal, mtol_newton=mtol_internal, iprint=['NEWTON_suc'])

    DOF = 3*ns.N + 1*cd.N  # T,u,v,p
    atol_gmres = mtol_gmres * np.sqrt(DOF)
    atol_nonlin = mtol_nonlin * np.sqrt(DOF)

    # initialize OpenMDAO solver
    prob = om.Problem()
    model = prob.model
    pg = model.add_subsystem('PG', om.Group())
    pg.add_subsystem('ConvectionDiffusion', ConvectionDiffusion_Component(solver_CD=cd, solver_NS=ns))
    pg.add_subsystem('NavierStokes', NavierStokes_Component(solver_CD=cd, solver_NS=ns))
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
        elif mode == 'JNK':
            pg.nonlinear_solver.options['maxiter'] = 100
            pg.linear_solver = om.ScipyKrylov(iprint=2, err_on_non_converge=True,
                                              atol=atol_gmres, rtol=0, restart=restart, maxiter=5000)
            pg.linear_solver.precon = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)
        else:
            raise ValueError('Unknown method')
    prob.setup()

    # solve
    prob.run_model()

    # results
    T = pg.get_val('ConvectionDiffusion.T_cd')
    u = pg.get_val('NavierStokes.u_ns')
    v = pg.get_val('NavierStokes.v_ns')

    # post processing
    T_plot = cd._get_interpol(T, points_plot)
    u_plot = ns._get_interpol(u, points_plot)
    v_plot = ns._get_interpol(v, points_plot)
    return T_plot, u_plot, v_plot