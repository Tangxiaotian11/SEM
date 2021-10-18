import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import matplotlib.pyplot as plt
from Solvers.NavierStokes_Solver import NavierStokesSolver

"""
Solves the dimensionless steady-state NAVIER-STOKES equations for u(x,y), v(x,y) and p(x,y)
Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v] ∀(x,y)∈[0,L_x]×[0,L_y]
∇∘[u, v] = 0 ∀(x,y)∈[0,L_x]×[0,L_y]
as lid-driven flow
u(x,L_y) = 1 ∀x∈[0,L_x]
with artificial homogeneous NEUMANN boundary condition for p
∂ₙp = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])
"""

if __name__ == "__main__":
    # input
    L_x = 1
    L_y = 1
    Re = 400
    P = 4
    N_ex = 16
    N_ey = 16

    # solver
    ns = NavierStokesSolver(L_x, L_y, Re, 0, P, N_ex, N_ey, u_N=1,
                            iprint=['NEWTON_suc', 'NEWTON_iter', 'LGMRES_suc', 'LGMRES_iter', 'LU_suc'])

    # plotting points
    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')

    # run
    u_plot, v_plot, p_plot = ns.run(T_func=lambda x, y: 0*x*y,
                                    points_plot=(x_plot, y_plot))

    fig = plt.figure(figsize=(L_x*4, L_y*4))
    ax = fig.gca()
    ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=2)
    ax.set_title(f"Re={Re:.0e}, P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={ns._mtol_newton:.0e}", fontsize='small')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()