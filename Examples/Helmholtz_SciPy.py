import numpy as np
import scipy.sparse.linalg as sp_sparse_linalg
import SEM
import matplotlib.pyplot as plt

"""
Solves the dimensionless HELMHOLTZ equation for u(x,y)
lam⋅u - ∇²u = f(x,y) ∀(x,y)∈[0,L_x]×[0,L_y]
with DIRICHLET conditions
u(0,y)   = u_W(y) ∀y∈[0,L_y]
u(L_x,y) = u_E(y) ∀y∈[0,L_y]
u(x,0)   = u_S(x) ∀x∈[0,L_x]
u(x,L_y) = u_N(x) ∀x∈[0,L_x]
Backend solver is SciPy.
"""

# setup
L_x = 2     # length in x direction
L_y = 1     # length in y direction
lam = 1     # HELMHOLTZ parameter
P = 4       # polynomial order
N_ex = 2    # num of elements in x direction
N_ey = 3    # num of elements in y direction
exact = lambda x,y: np.sin(x/L_x*np.pi)*np.sin(y/L_y*2*np.pi) + x/L_x + y/L_y  # exact solution
f = lambda x,y: exact(x, y)*lam + ((np.pi/L_x)**2+(2*np.pi/L_y)**2)*np.sin(x/L_x*np.pi)*np.sin(y/L_y*2*np.pi)  # f(x,y)
u_W = lambda y: y/L_y      # DIRICHLET boundary condition at x=0
u_E = lambda y: y/L_y + 1  # DIRICHLET boundary condition at x=L_x
u_S = lambda x: x/L_x      # DIRICHLET boundary condition at y=0
u_N = lambda x: x/L_x + 1  # DIRICHLET boundary condition at y=L_y

# grid
dx = L_x / N_ex
dy = L_y / N_ey
points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)
x_1d = SEM.global_nodes_1d(P, N_ex, dx)
y_1d = SEM.global_nodes_1d(P, N_ey, dy)
points_e = SEM.element_nodes(P, N_ex, N_ey, dx, dy)

# matrices
M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
H = lam * M + K
# RHS vector
g = M @ f(points[0], points[1])

# DIRICHLET condition vectors
dirichlet = np.full(points.shape[1], np.nan)
dirichlet[np.isclose(points[0], 0)] = u_W(points[1, np.isclose(points[0], 0)])
dirichlet[np.isclose(points[0], L_x)] = u_E(points[1, np.isclose(points[0], L_x)])
dirichlet[np.isclose(points[1], 0)] = u_S(points[0, np.isclose(points[1], 0)])
dirichlet[np.isclose(points[1], L_y)] = u_N(points[0, np.isclose(points[1], L_y)])

# solve with DIRICHLET conditions
u = np.zeros(points.shape[1])
ind_dirichlet = ~np.isnan(dirichlet)  # indices where DIRICHLET condition
u[ind_dirichlet] = dirichlet[ind_dirichlet]  # set known solution
g -= H[:, ind_dirichlet] @ dirichlet[ind_dirichlet]  # bring columns on RHS
g = g[~ind_dirichlet]  # skip rows
H = H[~ind_dirichlet, :][:, ~ind_dirichlet]  # skip rows/cols
u[~ind_dirichlet], info = sp_sparse_linalg.cg(H, g)  # solve on free indices
if info != 0:
    raise RuntimeError('CG failed to converge')

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
if exact is not None:
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
if exact is not None:
    print(f"lg(max(|exact - u|)) = {np.log10(np.max(np.abs(u_plot - exact_plot)))}")