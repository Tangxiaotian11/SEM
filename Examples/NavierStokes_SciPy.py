import numpy as np
import scipy.sparse.linalg as sp_sparse_linalg
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import SEM
import matplotlib.pyplot as plt

"""
Solves the dimensionless steady-state NAVIER-STOKES equations for u(x,y) and v(x,y)
Re([u, v]∘∇)[u, v]  = -∇p + ∇²[u, v] ∀(x,y)∈[0,L_x]×[0,L_y]
∇∘[u, v] = 0 ∀(x,y)∈[0,L_x]×[0,L_y]
as lid-driven flow
u(x,L_y) = 1 ∀x∈[0,L_x]
The steady-state is found by iterating the time dependent equations as long as necessary.
Temporal discretization is performed using the pressure-correction method from KIM-MOIN
(doi.org/10.1016/0021-9991(85)90148-2).
Backend solver is SciPy.
Possible reference solutions from GHIA (doi.org/10.1016/0021-9991(82)90058-4).
"""

# setup
L_x = 1     # length in x direction
L_y = 1     # length in y direction
Re = 4e2    # REYNOLDS number
P = 6       # polynomial order
N_ex = 5    # num of elements in x direction
N_ey = 5    # num of elements in y direction
dt = 2e-4   # step size in time
tol = 1e-2  # tolerance for stationarity such that max(‖Δu‖∞,‖Δv‖∞) < tol
u_lid = 1   # lid velocity

# grid
dx = L_x / N_ex
dy = L_y / N_ey
points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)
points_e = SEM.element_nodes(P, N_ex, N_ey, dx, dy)

# matrices
M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
M_inv = sp_sparse.diags(1/M.diagonal()).tocsr()
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
G_x, G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, dx, dy)
C_x, C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)

# DIRICHLET condition vectors
dirichlet_u = np.full(points.shape[1], np.nan)
dirichlet_v = np.full(points.shape[1], np.nan)
dirichlet_phi = np.full(points.shape[1], np.nan)
dirichlet_u[np.isclose(points[0], 0)] = 0
dirichlet_u[np.isclose(points[0], L_x)] = 0
dirichlet_u[np.isclose(points[1], 0)] = 0
dirichlet_u[np.isclose(points[1], L_y)] = u_lid
dirichlet_v[np.isclose(points[0], 0)] = 0
dirichlet_v[np.isclose(points[0], L_x)] = 0
dirichlet_v[np.isclose(points[1], 0)] = 0
dirichlet_v[np.isclose(points[1], L_y)] = 0
# dirichlet_phi[0] = 0  # reference pseudo pressure at south-east corner; required if a direct solver would be used

# initial condition
u = np.zeros(points.shape[1])
v = np.zeros(points.shape[1])
u[~np.isnan(dirichlet_u)] = dirichlet_u[~np.isnan(dirichlet_u)]
v[~np.isnan(dirichlet_v)] = dirichlet_v[~np.isnan(dirichlet_v)]


# solve
def solve_with_dirichlet(A, b, dirichlet):
    x = np.zeros(b.shape)
    ind_dirichlet = ~np.isnan(dirichlet)  # indices where DIRICHLET condition
    x[ind_dirichlet] = dirichlet[ind_dirichlet]  # set known solution
    b -= A[:, ind_dirichlet] @ dirichlet[ind_dirichlet]  # bring columns on RHS
    b = b[~ind_dirichlet]  # skip rows
    A = A[~ind_dirichlet, :][:, ~ind_dirichlet]  # skip rows/cols
    x[~ind_dirichlet], info = sp_sparse_linalg.gmres(A, b, tol=1e-4)  # solve on free indices
    if info != 0:
        raise RuntimeError('GMRES failed to converge')
    return x


res = 1
n = 0
while res > tol:
    n += 1
    # pressure-correction method; Convection term OSEEN like
    Conv = Re * (sparse.tensordot(C_x, u, (1, 0), return_type=sparse.COO).tocsr()
               + sparse.tensordot(C_y, v, (1, 0), return_type=sparse.COO).tocsr())  # i.e. = Re*(u @ C_x + v @ C_y)
    u_pre = solve_with_dirichlet(1/dt*M + 0.5*Conv + 0.5*K, (1/dt*M - 0.5*Conv - 0.5*K) @ u, dirichlet_u)
    v_pre = solve_with_dirichlet(1/dt*M + 0.5*Conv + 0.5*K, (1/dt*M - 0.5*Conv - 0.5*K) @ v, dirichlet_v)
    phi = solve_with_dirichlet(K, -1/dt * (G_x @ u_pre + G_y @ v_pre), dirichlet_phi)
    u_new = u_pre - dt * M_inv @ G_x @ phi
    v_new = v_pre - dt * M_inv @ G_y @ phi
    res = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v)))/dt
    print(f"t = {n*dt:.3e}; lg(res) = {np.log10(res)}")
    u = u_new.copy()
    v = v_new.copy()

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
v_e = SEM.scatter(v, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
v_plot = SEM.eval_interpolation(v_e, points_e, (x_plot, y_plot))
fig = plt.figure(figsize=(L_x*4, L_y*4))
ax = fig.gca()
ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=3)
ax.set_title(f"Re={Re:.0e}, P={P}, N_ex={N_ex}, N_ey={N_ey}, dt={dt:.0e}, tol={tol:.0e}", fontsize='small')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# print y, u(x=L_x/2,y)
np.savetxt('Cavity.out', np.transpose([y_plot[0, :], u_plot[np.isclose(x_plot, L_x/2)]]), delimiter=',')
