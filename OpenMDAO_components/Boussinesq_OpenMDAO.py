import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import SEM
from Examples.ConvectionDiffusionSolver import ConvectionDiffusionSolver
from Examples.NavierStokesSolver import NavierStokesSolver
from OpenMDAO_components.ConvectionDiffusion_Component import ConvectionDiffusion_Component
from OpenMDAO_components.NavierStokes_Component import NavierStokes_Component
import openmdao.api as om
import matplotlib.pyplot as plt
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

"""
Solves the dimensionless steady-state BOUSSINESQ equations for u(x,y), v(x,y) and T(x,y) on (x,y)∈[0,L_x]×[0,L_y]
Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v] + Gr/Re [0, T]
∇∘[u, v] = 0
Pe [u, v]∘∇T = ∇²T
with isothermal walls and adiabatic floor/ceiling
T(0,y) = -1/2, T(L_x,y) = 1/2 ∀y∈[0,L_y]
∂ₙT(x,0) = ∂ₙT(x,L_y) = 0 ∀x∈[0,L_x]
and no-slip condition
u(x,y) = v(x,y) = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])
Backend solver is OpenMDAO with appropriate connected components.
Possible reference solutions from MARKATOS-PERICLEOUS (doi.org/10.1016/0017-9310(84)90145-5).
"""

# input
L_x = 1      # length in x direction
L_y = 1      # length in y direction
Re = 1e2     # REYNOLDS number
Ra = 1e3     # RAYLEIGH number
Pr = 0.71    # PRANDTL number
P = 4        # polynomial order
N_ex = 8     # num of elements in x direction
N_ey = 8     # num of elements in y direction
mtol_internal = 1e-13  # tolerance on root mean square residual for internal solvers
mtol_gmres = 1e-10  # tolerance on root mean square residual for GMRES
mtol_nonlin = 1e-8  # tolerance on root mean square residual for NEWTON

N = (N_ex*P+1)*(N_ey*P+1)  # grid points per variable
DOF = 4*N  # T,u,v,p
atol_gmres = mtol_gmres * np.sqrt(DOF)
atol_nonlin = mtol_nonlin * np.sqrt(DOF)

# initialize backend solvers
cd = ConvectionDiffusionSolver(L_x=L_x, L_y=L_y, Pe=Re*Pr,
                               P=P, N_ex=N_ex, N_ey=N_ey,
                               T_W=0.5, T_E=-0.5,
                               mtol=mtol_internal)
ns = NavierStokesSolver(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr,
                        P=P, N_ex=N_ex, N_ey=N_ey,
                        mtol=mtol_internal, mtol_newton=mtol_internal, iprint=['NEWTON_suc'])

# initialize OpenMDAO solver
prob = om.Problem()
model = prob.model
pg = model.add_subsystem('PG', om.ParallelGroup())
pg.add_subsystem('ConvectionDiffusion', ConvectionDiffusion_Component(solver_CD=cd, solver_NS=ns))  # rank 0
pg.add_subsystem('NavierStokes', NavierStokes_Component(solver_CD=cd, solver_NS=ns))  # rank 1
pg.connect('ConvectionDiffusion.T_cd', 'NavierStokes.T_cd')
pg.connect('NavierStokes.u_ns', 'ConvectionDiffusion.u_ns')
pg.connect('NavierStokes.v_ns', 'ConvectionDiffusion.v_ns')

# - NEWTON
pg.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, max_sub_solves=0, maxiter=1000, atol=atol_nonlin, rtol=0)
#pg.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=8, rho=0.8, c=0.2)
# --- JACOBI preconditioned NEWTON-KRYLOV
pg.linear_solver = om.PETScKrylov(iprint=2, atol=atol_gmres, rtol=0, restart=20, ksp_type='gmres', precon_side='left')
pg.linear_solver.precon = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)  # requires change in linear_block_jac.py
# --- NEWTON-JACOBI
#pg.linear_solver = om.LinearBlockJac(iprint=-1, rtol=0, atol=0, maxiter=1)  # requires change in linear_block_jac.py
# - Nonlinear GAUSS-SEIDEL
#pg.nonlinear_solver = om.NonlinearBlockGS(iprint=2, use_apply_nonlinear=True, maxiter=1000, atol=atol_nonlin, rtol=0)  # requires change in nonlinear_block_gs.py
prob.setup()

# solve
prob.run_model()

# local post-processing
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')
if rank == 0:
    T_plot = cd._get_interpol(pg.get_val('PG.ConvectionDiffusion.T_cd'), (x_plot, y_plot))
    results = T_plot
    iters = pg.ConvectionDiffusion.iter_count_solve
if rank == 1:
    u_plot = ns._get_interpol(pg.get_val('PG.NavierStokes.u_ns'), (x_plot, y_plot))
    v_plot = ns._get_interpol(pg.get_val('PG.NavierStokes.v_ns'), (x_plot, y_plot))
    results = [u_plot, v_plot]
    iters = pg.NavierStokes.iter_count_solve
# gather to rank 0
results = MPI.COMM_WORLD.gather(results, root=0)
iters = MPI.COMM_WORLD.gather(iters, root=0)

# post-processing
if rank == 0:
    T_plot, u_plot, v_plot = results[0], *results[1]
    fig = plt.figure(figsize=(L_x*6, L_y*6))
    ax = fig.gca()
    ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=3)
    CS = ax.contour(x_plot, y_plot, T_plot, levels=11, colors='k', linestyles='solid')
    ax.clabel(CS, inline=True)
    ax.set_title(f"Re={Re:.1e}, Ra={Ra:.1e}, Pr={Pr}, P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={mtol_nonlin:.0e}",
                 fontsize='small')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.savefig('temp.png', dpi=fig.dpi)

    print(f"num of NonLin iterations: {pg.nonlinear_solver._iter_count}")
    print(f"num of get_update calls in CD: {iters[0]}")
    print(f"num of get_update calls in NS: {iters[1]}")
    print(f"u_max*RePr = {np.max(u_plot)*Re*Pr:.2f}")
    print(f"v_max*RePr = {np.max(v_plot)*Re*Pr:.2f}")