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
Ra = 1e4     # RAYLEIGH number
Pr = 0.71    # PRANDTL number
P = 4        # polynomial order
N_ex = 8     # num of elements in x direction
N_ey = 8     # num of elements in y direction
mtol_internal = 1e-13  # tolerance on root mean square residual for internal solvers
mtol_gmres = 1e-10  # tolerance on root mean square residual for GMRES
mtol_newton = 1e-8  # tolerance on root mean square residual for NEWTON

N = (N_ex*P+1)*(N_ey*P+1)
atol_gmres = mtol_gmres*np.sqrt(N*4)  # N * num var = size of linear system
atol_newton = mtol_newton*np.sqrt(N*4)

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

# - NEWTON
parallel.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, max_sub_solves=0, maxiter=1000, atol=atol_newton, rtol=0)
parallel.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=8, rho=0.8, c=0.2)
# --- JACOBI preconditioned NEWTON-KRYLOV
parallel.linear_solver = om.PETScKrylov(iprint=2, atol=atol_gmres, rtol=0, restart=20, ksp_type='gmres', precon_side='left')
parallel.linear_solver.precon = om.LinearRunOnce()
# --- Inexact NEWTON
#parallel.linear_solver = om.LinearRunOnce()
# - Nonlinear JACOBI
#parallel.nonlinear_solver = om.NonlinearBlockGS(iprint=2, use_apply_nonlinear=True, maxiter=1000, atol=atol_newton, rtol=0)
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
    # initial guess
    u = np.zeros(N)
    v = np.zeros(N)
    # hand over guess
    prob['parallel.NavierStokes.u'] = u
    prob['parallel.NavierStokes.v'] = v

# solve
prob.run_model()

# local post-processing
if rank == 0:
    results = None
    iters = model.parallel.ConvectionDiffusion.iter_count_solve
if rank == 1:
    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')
    u_plot = ns._get_interpol(prob['parallel.NavierStokes.u'], (x_plot, y_plot))
    v_plot = ns._get_interpol(prob['parallel.NavierStokes.v'], (x_plot, y_plot))
    results = [np.max(u_plot), np.max(v_plot)]
    iters = model.parallel.NavierStokes.iter_count_solve
# gather to rank 0
results = MPI.COMM_WORLD.gather(results, root=0)
iters = MPI.COMM_WORLD.gather(iters, root=0)

# post-processing
if rank == 0:
    print(f"num of NonLin iterations: {model.parallel.nonlinear_solver._iter_count}")
    print(f"num of get_update calls in CD: {iters[0]}")
    print(f"num of get_update calls in NS: {iters[1]}")
    print(f"u_max*RePr = {results[1][0]*Re*Pr:.2f}")
    print(f"v_max*RePr = {results[1][1]*Re*Pr:.2f}")