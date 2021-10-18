import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import matplotlib.pyplot as plt
from Solvers.ConvectionDiffusion_Solver import ConvectionDiffusionSolver

"""
Solves the dimensionless steady-state convection-diffusion equations for T(x,y)
Pe([u, v]∘∇)T = ∇²T ∀(x,y)∈[0,L_x]×[0,L_y]
with circular flow
[u, v] = [y-L_y/2, L_x/2+x] ∀(x,y)∈[0,L_x]×[0,L_y]
and homogenous NEUMANN boundary conditions
∂ₙT(x,L_y) = ∂ₙT(x,0) = 0 ∀x∈[0,L_x]
and symmetric DIRICHLET conditions
T(y,0) = 0.5, T(y,L_x) = -0.5 ∀y∈[0,L_y]
"""

if __name__ == "__main__":
    # input
    L_x = 1
    L_y = 1
    P = 4
    N_ex = 16
    N_ey = 16
    Pe = 40
    u = lambda x, y: y - L_y/2
    v = lambda x, y: L_x/2 - x

    # solver
    cd = ConvectionDiffusionSolver(L_x, L_y, Pe, P, N_ex, N_ey, T_E=-0.5, T_W=0.5, iprint=['LGMRES_suc', 'LGMRES_iter'])

    # plotting points
    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')

    # run
    T_plot = cd.run(u, v, (x_plot, y_plot))

    # plot
    fig = plt.figure(figsize=(L_x*4, L_y*4))
    ax = fig.gca()
    CS = ax.contour(x_plot, y_plot, T_plot, levels=11, colors='k', linestyles='solid')
    ax.streamplot(x_plot.T, y_plot.T, u(x_plot, y_plot).T, v(x_plot, y_plot).T, density=1)
    ax.clabel(CS, inline=True)
    ax.set_title(f"P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={cd._mtol:.0e}", fontsize='small')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, L_x])
    ax.set_ylim([0, L_y])
    plt.show()