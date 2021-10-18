import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import matplotlib.pyplot as plt
from OpenMDAO.Boussinesq_SequentialCoupler import run
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

if __name__ == "__main__":
    # input
    L_x = 1.     # length in x direction
    L_y = 1.     # length in y direction
    Re = 1e3     # REYNOLDS number
    Ra = 1e3     # RAYLEIGH number
    Pr = 0.71    # PRANDTL number
    P = 4        # polynomial order
    N_ex = 8     # num of elements in x direction
    N_ey = 8     # num of elements in y direction

    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')

    T_plot, u_plot, v_plot = run((x_plot, y_plot), L_x, L_y,
                                 Re, Ra, Pr,
                                 P, N_ex, N_ey,
                                 P, N_ex, N_ey,
                                 mode='JNK')

    if rank == 0:
        print(f"u_max*RePr = {np.max(u_plot)*Re*Pr:.2f}")
        print(f"v_max*RePr = {np.max(v_plot)*Re*Pr:.2f}")

        fig = plt.figure(figsize=(L_x*6, L_y*6))
        ax = fig.gca()
        ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=3)
        CS = ax.contour(x_plot, y_plot, T_plot, levels=11, colors='k', linestyles='solid')
        ax.clabel(CS, inline=True)
        ax.set_title(f"Ra={Ra:.1e}, P={P}, N_ex={N_ex}, N_ey={N_ey}",
                     fontsize='small')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.savefig('temp.png', dpi=fig.dpi)