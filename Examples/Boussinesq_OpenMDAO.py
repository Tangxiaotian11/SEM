import numpy as np
import SEM
from OpenMDAO_components.NavierStokes import NavierStokes
from OpenMDAO_components.ConvectionDiffusion import ConvectionDiffusion
import openmdao.api as om
import matplotlib.pyplot as plt

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
Re = 1       # REYNOLDS number
Ra = 1e3     # RAYLEIGH number
Pr = 0.71    # PRANDTL number
P = 4        # polynomial order
N_ex = 8     # num of elements in x direction
N_ey = 8     # num of elements in y direction
mtol = 1e-4  # tolerance on mean square residual

N = (N_ex*P+1)*(N_ey*P+1)
tol = mtol*np.sqrt(N)

# grid
points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
points_e = SEM.element_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)

# initialize global coefficients vectors
u = np.zeros(N)
v = np.zeros(N)
T = np.zeros(N)
T[np.isclose(points[0], 0)] = 0.5  # initial guess
T[np.isclose(points[0], L_x)] = -0.5

# initialize OpenMDAO solver
prob = om.Problem()
model = prob.model
model.add_subsystem('ConvectionDiffusion', ConvectionDiffusion(L_x=L_x, L_y=L_y, Pe=Re*Pr, T_W=0.5, T_E=-0.5,
                                                               P=P, N_ex=N_ex, N_ey=N_ey, points=points,
                                                               precon_type='ilu'))
model.add_subsystem('NavierStokes', NavierStokes(L_x=L_x, L_y=L_y, Re=Re, Gr=Ra/Pr,
                                                 P=P, N_ex=N_ex, N_ey=N_ey, points=points,
                                                 solver_type='qmr', iprecon_type='ilu', fill_factor=3, drop_tol=1e-5))
model.connect('NavierStokes.u', 'ConvectionDiffusion.u')
model.connect('NavierStokes.v', 'ConvectionDiffusion.v')
model.connect('ConvectionDiffusion.T', 'NavierStokes.T')
model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, maxiter=200, atol=tol, rtol=0)
model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=5, rho=0.8, c=0.2)
model.linear_solver = om.LinearRunOnce()
prob.setup()
# om.n2(prob) # prints N2-diagram

# solve
prob['NavierStokes.u'] = u  # hand over guess
prob['NavierStokes.v'] = v
prob['ConvectionDiffusion.T'] = T
prob.run_model()
u = prob['NavierStokes.u']
v = prob['NavierStokes.v']
T = prob['ConvectionDiffusion.T']

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
v_e = SEM.scatter(v, P, N_ex, N_ey)
T_e = SEM.scatter(T, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
v_plot = SEM.eval_interpolation(v_e, points_e, (x_plot, y_plot))
T_plot = SEM.eval_interpolation(T_e, points_e, (x_plot, y_plot))
fig = plt.figure(figsize=(L_x*6, L_y*6))
ax = fig.gca()
ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=3)
CS = ax.contour(x_plot, y_plot, T_plot, levels=11, colors='k', linestyles='solid')
ax.clabel(CS, inline=True)
ax.set_title(f"Re={Re:.0e}, Ra={Ra:.0e}, Pr={Pr}, P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={mtol:.0e}",
             fontsize='small')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# info
u_max = np.max(u_plot[np.isclose(x_plot, 0.5)])
y_max = y_plot[np.isclose(u_plot, u_max)][0]
v_max = np.max(v_plot[np.isclose(y_plot, 0.5)])
x_max = x_plot[np.isclose(v_plot, v_max)][0]
print(f"u_max(x=0.5) = {u_max:.2f} @ y = {y_max:.2f}")
print(f"v_max(y=0.5) = {v_max:.2f} @ x = {x_max:.2f}")