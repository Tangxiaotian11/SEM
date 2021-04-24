"""
Continuous GALERKIN spectral element method
"""
import numpy as np
import scipy.interpolate as sp_interp
import GLL


def xi2x(e, xi, D_x):
    """
    Returns physical coordinate x from standard coordinate xi in element e\n
    :param e: element number
    :param xi: standard coordinate
    :param D_x: element width
    """
    return D_x/2 * (xi+1) + D_x*e


def element_nodes_1d(P: int, N_ex: int, D_x: float):
    """
    Returns element nodes for one dimension\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param D_x: element width
    :return: xᵐₖ[m,k]
    """
    nodes = GLL.standard_nodes(P)[0]
    return np.vstack([xi2x(m, nodes, D_x) for m in range(N_ex)])


def global_nodes_1d(P: int, N_ex: int, D_x: float):
    """
    Returns global nodes vector for one dimension\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param D_x: element width
    :return: xₚ[p]
    """
    x_e = element_nodes_1d(P, N_ex, D_x)
    return np.unique(np.ravel(x_e))  # TODO check round off error


def element_nodes(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns element nodes\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    :return: (xᵐⁿₖₗ[m,n,k,l], yᵐⁿₖₗ[m,n,k,l])
    """
    x_e1d = element_nodes_1d(P, N_ex, D_x)
    y_e1d = element_nodes_1d(P, N_ey, D_y)
    points_e = np.zeros((2, N_ex, N_ey, P+1, P+1))
    for m in range(N_ex):
        for n in range(N_ey):
            (points_e[0, m, n], points_e[1, m, n]) = np.meshgrid(x_e1d[m], y_e1d[n], indexing='ij')
    return points_e


def global_nodes(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns global nodes vector\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    :return: (xₚ[p], yₚ[p])
    """
    x_1d = global_nodes_1d(P, N_ex, D_x)
    y_1d = global_nodes_1d(P, N_ey, D_y)
    return np.reshape(np.array(np.meshgrid(x_1d, y_1d, indexing='ij')), (2, x_1d.size*y_1d.size))


def assemble(A_e: np.ndarray):  # TODO sparse and parallel implementation urgently needed
    """
    Returns global matrix/vector from element array\n
    :param A_e: element array Aᵐⁿᵢⱼᵣₛₖₗ[m,n,i,j,r,s,k,l] or Aᵐⁿᵢⱼₖₗ[m,n,i,j,k,l] or Aᵐⁿᵢⱼ[m,n,i,j]
    """
    N_ex = A_e.shape[0]
    N_ey = A_e.shape[1]
    P = A_e.shape[2] - 1

    def p(m, n, i, j):
        return n*P+j + (N_ey*P+1) * (m*P+i)

    if A_e.ndim == 4:
        A = np.zeros((P*N_ex+1)*(P*N_ey+1))
        for m in range(N_ex):
            for n in range(N_ey):
                for i in range(P+1):
                    for j in range(P+1):
                        A[p(m, n, i, j)] += A_e[m, n, i, j]
    if A_e.ndim == 6:
        A = np.zeros(((P*N_ex+1)*(P*N_ey+1),)*2)
        for m in range(N_ex):
            for n in range(N_ey):
                for i in range(P+1):
                    for j in range(P+1):
                        for k in range(P+1):
                            for l in range(P+1):
                                A[p(m, n, i, j), p(m, n, k, l)] += A_e[m, n, i, j, k, l]
    if A_e.ndim == 8:
        A = np.zeros(((P*N_ex+1)*(P*N_ey+1),)*3)
        for m in range(N_ex):
            for n in range(N_ey):
                for i in range(P+1):
                    for j in range(P+1):
                        for r in range(P+1):
                            for s in range(P+1):
                                for k in range(P+1):
                                    for l in range(P+1):
                                        A[p(m, n, i, j), p(m, n, r, s), p(m, n, k, l)] += A_e[m, n, i, j, r, s, k, l]
    return A


def scatter(u: np.ndarray, P: int, N_ex: int, N_ey: int):
    """
    Returns element coefficients from global coefficients vector
    :param u: global coefficients vector uₚ[p]
    :param P: polynomial order
    :param N_ex: num elements in x direction
    :param N_ey: ... in y direction
    :return: uᵐⁿᵢⱼ[m,n,i,j]
    """
    if u.shape[0] != (P*N_ex+1)*(P*N_ey+1):
        raise ValueError('Not a valid combination of global coefficients vector, P, N_ex, and N_ey')

    u_e = np.zeros((N_ex, N_ey, P+1, P+1))

    def p(m, n, i, j):
        return n*P+j + (N_ey*P+1) * (m*P+i)

    for m in range(N_ex):
        for n in range(N_ey):
            for i in range(P+1):
                for j in range(P+1):
                    u_e[m, n, i, j] = u[p(m, n, i, j)]
    return u_e


def element_mass_matrix_1d(P: int, N_ex: int, D_x: float):
    """
    Returns element mass matrix for one dimension\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param D_x: element width
    :return: Mᵐᵢₖ[m,i,k]
    """
    M_s = GLL.standard_mass_matrix(P)
    return np.einsum('m,ik->mik', D_x/2*np.ones(N_ex), M_s)


def element_stiffness_matrix_1d(P: int, N_ex: int, D_x: float):
    """
    Returns element stiffness matrix for one dimension\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param D_x: element width
    :return: Kᵐᵢₖ[m,i,k]
    """
    M_s = GLL.standard_mass_matrix(P)
    D_s = GLL.standard_differentiation_matrix(P)
    return np.einsum('m,ik->mik', 2/D_x*np.ones(N_ex), D_s.transpose() @ M_s @ D_s)


def global_mass_matrix(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns global mass matrix\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    """
    M_ex = element_mass_matrix_1d(P, N_ex, D_x)
    M_ey = element_mass_matrix_1d(P, N_ey, D_y)
    return assemble(np.einsum('mik,njl->mnijkl', M_ex, M_ey, optimize=True))


def global_stiffness_matrix(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns global stiffness matrix\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    """
    M_ex = element_mass_matrix_1d(P, N_ex, D_x)
    M_ey = element_mass_matrix_1d(P, N_ey, D_y)
    K_ex = element_stiffness_matrix_1d(P, N_ex, D_x)
    K_ey = element_stiffness_matrix_1d(P, N_ey, D_y)
    return assemble(np.einsum('mik,njl->mnijkl', K_ex, M_ey, optimize=True)
                  + np.einsum('mik,njl->mnijkl', M_ex, K_ey, optimize=True))


def global_gradient_matrices(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns global gradient matrices\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    :return: G_x, G_y
    """
    M_ex = element_mass_matrix_1d(P, N_ex, D_x)
    M_ey = element_mass_matrix_1d(P, N_ey, D_y)
    G_s = GLL.standard_gradient_matrix(P)
    return assemble(np.einsum('m,ik,njl->mnijkl', np.ones(N_ex), G_s, M_ey, optimize=True)), \
           assemble(np.einsum('mik,n,jl->mnijkl', M_ex, np.ones(N_ex), G_s, optimize=True))


def global_convection_matrices(P: int, N_ex: int, N_ey: int, D_x: float, D_y: float):
    """
    Returns global gradient matrices\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param D_x: element width in x direction
    :param D_y: ... in y direction
    :return: C_x, C_y
    """
    M_ex = element_mass_matrix_1d(P, N_ex, D_x)
    M_ey = element_mass_matrix_1d(P, N_ey, D_y)
    C_s = GLL.standard_convection_matrix(P)
    return assemble(np.einsum('m,irk,sl,njl->mnijrskl', np.ones(N_ex), C_s, np.identity(P+1), M_ey, optimize=True)), \
           assemble(np.einsum('rk,nik,m,jsl->mnijrskl', np.identity(P+1), M_ex, np.ones(N_ex), C_s, optimize=True))


def eval_interpolation(u_e: np.ndarray, points_e: np.ndarray, points_plot):
    """
    Returns the evaluation of u in the points points_plot\n
    :param u_e: element coefficients array uᵐⁿₖₗ[m,n,i,j]
    :param points_e: element nodes (xᵐⁿₖₗ[m,n,k,l],yᵐⁿₖₗ[m,n,k,l])
    :param points_plot: evaluation points (xᵢⱼ[i,j],yᵢⱼ[i,j])
    """
    N_ex = u_e.shape[0]
    N_ey = u_e.shape[1]
    P = u_e.shape[2]-1
    x_e = points_e[0, :, 0, :, 0]
    y_e = points_e[1, 0, :, 0, :]
    eye = np.identity(P+1)
    val = np.zeros((points_plot[0].shape[0], points_plot[0].shape[1]))
    for m in range(N_ex):
        for n in range(N_ey):
            ind = (points_plot[0] >= np.min(x_e[m]))\
                * (points_plot[0] <= np.max(x_e[m]))\
                * (points_plot[1] >= np.min(y_e[n]))\
                * (points_plot[1] <= np.max(y_e[n]))
            val[ind] = 0  # overwrite existing element contributions
            for k in range(P+1):
                for l in range(P+1):
                    basis_x = sp_interp.lagrange(x_e[m], eye[k])
                    basis_y = sp_interp.lagrange(y_e[n], eye[l])
                    val[ind] += basis_x(points_plot[0][ind]) * basis_y(points_plot[1][ind]) * u_e[m, n, k, l]
    return val
