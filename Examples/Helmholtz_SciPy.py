import numpy as np
import scipy.sparse.linalg as linalg
import SEM
import matplotlib.pyplot as plt

"""
Solves the dimensionless HELMHOLTZ equation for u(x,y)
lam⋅u - ∇²u = f(x,y) ∀(x,y)∈[0,L_x]×[0,L_y]
with homogeneous NEUMANN boundary conditions.
∂ₙu = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])
Backend solver is SciPy.
"""

# setup
L_x = 2     # length in x direction
L_y = 1     # length in y direction
lam = 1     # HELMHOLTZ parameter != 0
P = 4       # polynomial order
N_ex = 2    # num of elements in x direction
N_ey = 3    # num of elements in y direction
exact = lambda x, y: np.cos(x/L_x*np.pi)*np.cos(y/L_y*2*np.pi)  # exact solution
f = lambda x, y: (lam + (np.pi/L_x)**2 + (2*np.pi/L_y)**2)*exact(x, y)  # f(x,y)

# grid
dx = L_x / N_ex
dy = L_y / N_ey
points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)

# matrices
M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
H = lam * M + K
# RHS vector
g = M @ f(points[0], points[1])

# solve
u, info = linalg.cg(H, g)
if info != 0:
    raise RuntimeError('CG failed to converge')

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
points_e = SEM.element_nodes(P, N_ex, N_ey, dx, dy)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
exact_plot = exact(x_plot, y_plot)
fig = plt.figure()
ax = fig.gca(projection='3d')
for m in range(u_e.shape[0]):
    for n in range(u_e.shape[1]):
        ax.scatter(points_e[0, m, n, :, :],
                   points_e[1, m, n, :, :],
                   u_e[m, n], c='r')
ax.plot_wireframe(x_plot, y_plot, u_plot, rstride=2, cstride=2, color='r', label='approximate solution')
if exact is not None:
    ax.plot_wireframe(x_plot, y_plot, exact_plot, rstride=5, cstride=5, color='b', label='exact solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# error
print(f"lg(max(|exact - u|)) = {np.log10(np.max(np.abs(u_plot - exact_plot)))}")
