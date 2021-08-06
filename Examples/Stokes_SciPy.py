import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import scipy.sparse.linalg as linalg
import scipy.sparse as sp_sparse
import SEM
import matplotlib.pyplot as plt

"""
Solves the dimensionless steady-state STOKES equations for u(x,y), v(x,y) and p(x,y)
0 = -∇p + ∇²[u, v] ∀(x,y)∈[0,L_x]×[0,L_y]
∇∘[u, v] = 0 ∀(x,y)∈[0,L_x]×[0,L_y]
as lid-driven flow
u(x,L_y) = 1 ∀x∈[0,L_x]
with artificial homogeneous NEUMANN boundary condition for p
∂ₙp = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])
Backend solver is SciPy.
Possible reference solutions from GHIA (doi.org/10.1016/0021-9991(82)90058-4).
"""

# setup
L_x = 1     # length in x direction
L_y = 1     # length in y direction
P = 4       # polynomial order
N_ex = 16   # num of elements in x direction
N_ey = 16   # num of elements in y direction
u_lid = 1   # lid velocity

# grid
dx = L_x / N_ex
dy = L_y / N_ey
points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)
points_e = SEM.element_nodes(P, N_ex, N_ey, dx, dy)

# matrices
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
G_x, G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, dx, dy)

# DIRICHLET condition vectors
dirichlet_u = np.full(points.shape[1], np.nan)
dirichlet_v = np.full(points.shape[1], np.nan)
dirichlet_p = np.full(points.shape[1], np.nan)
dirichlet_u[np.isclose(points[0], 0)] = 0
dirichlet_u[np.isclose(points[0], L_x)] = 0
dirichlet_u[np.isclose(points[1], 0)] = 0
dirichlet_u[np.isclose(points[1], L_y)] = u_lid
dirichlet_v[np.isclose(points[0], 0)] = 0
dirichlet_v[np.isclose(points[0], L_x)] = 0
dirichlet_v[np.isclose(points[1], 0)] = 0
dirichlet_v[np.isclose(points[1], L_y)] = 0
dirichlet_p[np.isclose(points[0], L_x/2) * np.isclose(points[1], L_y/2)] = 0

# System matrix
mask = ~np.isnan(dirichlet_u)
mask_p = ~np.isnan(dirichlet_p)
K_dir = K.copy()
B = K.copy()
G_x_dir = G_x.copy()
G_y_dir = G_y.copy()
K_dir[mask, :] = G_x_dir[mask, :] = G_x_dir[mask, :] = 0
K_dir[mask, mask] = 1
B[~mask, :] = 0
B[mask_p, :] = 0
B[mask_p, mask_p] = 1

S = sp_sparse.bmat([[K_dir, None, G_x_dir],
                    [None, K_dir, G_y_dir],
                    [G_x_dir, G_y_dir, B]], format='csr')
S.eliminate_zeros()

# initial condition
u = np.zeros(points.shape[1])
v = np.zeros(points.shape[1])
u[~np.isnan(dirichlet_u)] = dirichlet_u[~np.isnan(dirichlet_u)]
v[~np.isnan(dirichlet_v)] = dirichlet_v[~np.isnan(dirichlet_v)]

sol, info = linalg.gmres(S, np.hstack((np.nan_to_num(dirichlet_u, nan=0.),
                                       np.nan_to_num(dirichlet_v, nan=0.),
                                       np.zeros(points.shape[1]))),
                         callback=lambda res: print(res), callback_type='pr_norm')
if info != 0:
    raise RuntimeError('GMRES failed to converge')
u, v, p = np.split(sol, 3)

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
v_e = SEM.scatter(v, P, N_ex, N_ey)
p_e = SEM.scatter(p, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
v_plot = SEM.eval_interpolation(v_e, points_e, (x_plot, y_plot))
p_plot = SEM.eval_interpolation(p_e, points_e, (x_plot, y_plot))
fig = plt.figure(figsize=(L_x*4, L_y*4))
ax = fig.gca()
ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=4)
# ax.contour(x_plot, y_plot, p_plot, levels=51, colors='k', linestyles='solid')
ax.set_title(f"P={P}, N_ex={N_ex}, N_ey={N_ey}", fontsize='small')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
