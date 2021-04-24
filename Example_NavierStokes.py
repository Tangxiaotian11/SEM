import numpy as np
import SEM
import matplotlib.pyplot as plt

"""
Solves the dimensionless time dependent NAVIER-STOKES equations for u(t,x,y) and v(t,x,y)
∂ₜ[u, v] + Re([u, v]∘∇)[u, v]  = -∇p + ∇²[u, v] ∀t>0, (x,y)∈[0,L_x]×[0,L_y]
∇∘[u, v] = 0 ∀t>0, (x,y)∈[0,L_x]×[0,L_y]
with tangential DIRICHLET conditions
v(t,0,y)   = v_W ∀t>0, y∈[0,L_y]
v(t,L_x,y) = v_E ∀t>0, y∈[0,L_y]
u(t,x,0)   = u_S ∀t>0, x∈[0,L_x]
u(t,x,L_y) = u_N ∀t>0, x∈[0,L_x]
Temporal discretization is performed using the pressure-correction method from KIM-MOIN
(doi.org/10.1016/0021-9991(85)90148-2).
Possible reference solutions from GHIA (doi.org/10.1016/0021-9991(82)90058-4).
"""

# setup
L_x = 1     # length in x direction
L_y = 1     # length in y direction
Re = 1e2    # REYNOLDS number
P = 5       # polynomial order
N_ex = 4    # num of elements in x direction
N_ey = 4    # num of elements in y direction
D_t = 1e-3  # step size in time
tol = 1e-3  # tolerance for stationarity such that max(‖Δu‖∞,‖Δv‖∞) < tol
v_W = 0     # tangential velocity at x=0
v_E = 0     # tangential velocity at x=L_x
u_S = 0     # tangential velocity at y=0
u_N = 1     # tangential velocity at y=L_y

# grid
D_x = L_x / N_ex
D_y = L_y / N_ey
points = SEM.global_nodes(P, N_ex, N_ey, D_x, D_y)
points_e = SEM.element_nodes(P, N_ex, N_ey, D_x, D_y)

# matrices
G_x, G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, D_x, D_y)
C_x, C_y = SEM.global_convection_matrices(P, N_ex, N_ey, D_x, D_y)
M = SEM.global_mass_matrix(P, N_ex, N_ey, D_x, D_y)
M_inv = np.diag(1/np.diag(M))
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, D_x, D_y)

# dirichlet condition vectors
dirichlet_u = np.full(points.shape[1], np.nan)
dirichlet_v = np.full(points.shape[1], np.nan)
dirichlet_phi = np.full(points.shape[1], np.nan)
dirichlet_u[np.isclose(points[0], 0)] = 0
dirichlet_u[np.isclose(points[0], L_x)] = 0
dirichlet_u[np.isclose(points[1], 0)] = u_S
dirichlet_u[np.isclose(points[1], L_y)] = u_N
dirichlet_v[np.isclose(points[0], 0)] = v_W
dirichlet_v[np.isclose(points[0], L_x)] = v_E
dirichlet_v[np.isclose(points[1], 0)] = 0
dirichlet_v[np.isclose(points[1], L_y)] = 0
dirichlet_phi[0] = 0  # reference pseudo pressure at south-east corner

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
    x[~ind_dirichlet] = np.linalg.solve(A, b)  # solve on free indices
    return x


res = 1
n = 0
while res > tol:
    n += 1
    # pressure-correction method; Convection term OSEEN like
    Conv = Re * (u @ C_x + v @ C_y)
    u_pre = solve_with_dirichlet(1/D_t*M + 0.5*Conv + 0.5*K, (1/D_t*M - 0.5*Conv - 0.5*K) @ u, dirichlet_u)
    v_pre = solve_with_dirichlet(1/D_t*M + 0.5*Conv + 0.5*K, (1/D_t*M - 0.5*Conv - 0.5*K) @ v, dirichlet_v)
    phi = solve_with_dirichlet(K, -1/D_t * (G_x @ u_pre + G_y @ v_pre), dirichlet_phi)
    u_new = u_pre - D_t * M_inv @ G_x @ phi
    v_new = v_pre - D_t * M_inv @ G_y @ phi
    res = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v)))/D_t
    print(f"t = {n*D_t:.3e}; lg(res) = {np.log10(res)}")
    u = u_new.copy()
    v = v_new.copy()

# scatter for plot
u_e = SEM.scatter(u, P, N_ex, N_ey)
v_e = SEM.scatter(v, P, N_ex, N_ey)

# plot
x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')
u_plot = SEM.eval_interpolation(u_e, points_e, (x_plot, y_plot))
v_plot = SEM.eval_interpolation(v_e, points_e, (x_plot, y_plot))
fig = plt.figure(figsize=(L_x*4, L_y*4))
ax = fig.gca()
ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=2)
ax.set_title(f"Re={Re:.0e}, P={P}, N_ex={N_ex}, N_ey={N_ey}, D_t={D_t:.0e}, tol={tol:.0e}", fontsize='small')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# print y, u(x=L_x/2,y)
np.savetxt('Cavity.out', np.transpose([y_plot[0, :], u_plot[np.isclose(x_plot, L_x/2)]]), delimiter=',')
