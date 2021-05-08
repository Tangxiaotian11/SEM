import numpy as np
import SEM
from OpenMDAO_components.NavierStokes import NavierStokes
import openmdao.api as om
import matplotlib.pyplot as plt

"""
Solves the dimensionless steady-state NAVIER-STOKES equations for u(x,y) and v(x,y) on (x,y)∈[0,L_x]×[0,L_y]
Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v]
∇∘[u, v] = 0
as lid-driven cavity flow
u(x,L_y) = 1 ∀x∈[0,L_x]
Backend solver is OpenMDAO with appropriate component.
Possible reference solutions from GHIA (doi.org/10.1016/0021-9991(82)90058-4).
"""

# input
L_x = 1.    # length in x direction
L_y = 1.    # length in y direction
Re = 4.0e2  # REYNOLDS number
P = 6       # polynomial order
N_ex = 5    # num of elements in x direction
N_ey = 5    # num of elements in y direction
dt = 1.e-4  # step size in time
tol = 1e-2  # tolerance on residuals

# grid
points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
points_e = SEM.element_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)

# initialize global coefficients vectors
u = np.zeros(points.shape[1])
v = np.zeros(points.shape[1])
u[np.isclose(points[1], L_y)] = 1.  # initial guess

# initialize OpenMDAO solver
prob = om.Problem()
model = prob.model
model.add_subsystem('NavierStokes', NavierStokes(L_x=L_x, L_y=L_y, Re=Re, u_N=1.,
                                                 P=P, N_ex=N_ex, N_ey=N_ey, points=points, dt=dt))
model.nonlinear_solver = om.NewtonSolver(iprint=2, solve_subsystems=True, maxiter=800, atol=tol, rtol=1e-18)
model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(iprint=2, maxiter=5, rho=0.8, c=0.2)
model.linear_solver = om.ScipyKrylov(iprint=1, atol=1e-4, rtol=1e-18, maxiter=4000, restart=1000)
prob.setup()


# solve
prob['NavierStokes.u'], prob['NavierStokes.v'] = u, v  # hand over guess
prob.run_model()
u, v = prob['NavierStokes.u'], prob['NavierStokes.v']

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
v_e = SEM.scatter(v, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
v_plot = SEM.eval_interpolation(v_e, points_e, (x_plot, y_plot))
fig = plt.figure(figsize=(L_x*4, L_y*4))
ax = fig.gca()
ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=3)
ax.set_title(f"Re={Re:.0e}, P={P}, N_ex={N_ex}, N_ey={N_ey}, dt={dt:.0e}, tol={tol:.0e}", fontsize='small')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()