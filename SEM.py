"""
Continuous GALERKIN spectral element method
"""
import typing
import numpy as np
import GLL
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices


def xi2x(e: int, xi: float, dx: float) -> float:
    """
    Returns physical coordinate x from standard coordinate xi in element e. This function is vectorized.\n
    :param e: element number
    :param xi: standard coordinate
    :param dx: element width
    """
    if np.any(xi > 1) or np.any(xi < -1):
        raise ValueError('xi out of range')
    return dx/2 * (xi+1) + dx*e


def x2xi(x: float, dx: float) -> typing.Tuple[int, float]:
    """
    Returns standard coordinate xi and element number e from physical coordinate x. This function is vectorized.\n
    :param x: physical coordinate
    :param dx: element width
    :return: [e, xi]
    """
    xi, e = np.modf(x/dx)
    xi = 2*xi-1
    # shifting (e, -1) -> (e-1, 1) for e>0
    mask = np.isclose(xi, -1)*(e > 0)
    e[mask] -= 1
    xi[mask] = 1
    return e.astype(int), xi


def element_nodes_1d(P: int, N_ex: int, dx: float):
    """
    Returns element nodes for one dimension.\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param dx: element width
    :return: xᵐₖ[m,k]
    """
    nodes = GLL.standard_nodes(P)[0]
    return np.vstack([xi2x(m, nodes, dx) for m in range(N_ex)])


def global_nodes_1d(P: int, N_ex: int, dx: float):
    """
    Returns global nodes vector for one dimension.\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param dx: element width
    :return: xₚ[p]
    """
    x_e = element_nodes_1d(P, N_ex, dx)
    return np.insert(np.ravel(x_e[:, 1:]), 0, 0)  # skip 0-th node of each element; insert 0 at start


def element_nodes(P: int, N_ex: int, N_ey: int, dx: float, dy: float):
    """
    Returns element nodes array.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    :return: [xᵐⁿₖₗ[m,n,k,l], yᵐⁿₖₗ[m,n,k,l]]
    """
    x_e1d = element_nodes_1d(P, N_ex, dx)
    y_e1d = element_nodes_1d(P, N_ey, dy)
    points_e = np.zeros((2, N_ex, N_ey, P+1, P+1))
    for m in range(N_ex):
        for n in range(N_ey):
            (points_e[0, m, n], points_e[1, m, n]) = np.meshgrid(x_e1d[m], y_e1d[n], indexing='ij')
    return points_e


def global_nodes(P: int, N_ex: int, N_ey: int, dx: float, dy: float):
    """
    Returns global nodes vectors.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    :return: [xₚ[p], yₚ[p]]
    """
    x_1d = global_nodes_1d(P, N_ex, dx)
    y_1d = global_nodes_1d(P, N_ey, dy)
    return np.reshape(np.array(np.meshgrid(x_1d, y_1d, indexing='ij')), (2, x_1d.size*y_1d.size))


def global_index(P: int, N_ex: int, N_ey: int, m: int, n: int, i: int, j: int) -> int:
    """
    Returns global index from local index. This function in vectorized.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param m: element index in x direction
    :param n: ... in y direction
    :param i: node index in x direction
    :param j: ... in y direction
    """
    if np.any(m >= N_ex) or np.any(n >= N_ey) or np.any(i > P) or np.any(j > P):
        raise ValueError('Indices out of range')
    return n*P+j + (N_ey*P+1) * (m*P+i)


def assemble(A_e: np.ndarray):
    """
    Returns global matrix/vector from element array.
    1d/2d/3d arrays are returned in NumPy-array/SciPy-CSR/Sparse-COO format respectively\n
    :param A_e: element array Aᵐⁿᵢⱼᵣₛₖₗ[m,n,i,j,r,s,k,l] or Aᵐⁿᵢⱼₖₗ[m,n,i,j,k,l] or Aᵐⁿᵢⱼ[m,n,i,j]
    """
    N_ex = A_e.shape[0]
    N_ey = A_e.shape[1]
    P = A_e.shape[2] - 1

    # The coordinates array will contain duplicates.
    # In the assembly of the sparse matrices, the duplicates will be summed over.

    if A_e.ndim == 4:
        (m, n, i, j) = np.nonzero(A_e)
        coords = np.vstack((global_index(P, N_ex, N_ey, m, n, i, j), np.zeros(m.size)))
        data = A_e[m, n, i, j]
        A = sp_sparse.coo_matrix((data, coords), shape=((P*N_ex+1)*(P*N_ey+1),1))
        A = A.toarray()[:, 0]
    if A_e.ndim == 6:
        (m, n, i, j, k, l) = np.nonzero(A_e)
        coords = np.vstack((global_index(P, N_ex, N_ey, m, n, i, j),
                            global_index(P, N_ex, N_ey, m, n, k, l)))
        data = A_e[m, n, i, j, k, l]
        A = sp_sparse.coo_matrix((data, coords), shape=((P*N_ex+1)*(P*N_ey+1),)*2)
        A = A.tocsr()
    if A_e.ndim == 8:
        (m, n, i, j, r, s, k, l) = np.nonzero(A_e)
        coords = np.vstack((global_index(P, N_ex, N_ey, m, n, i, j),
                            global_index(P, N_ex, N_ey, m, n, r, s),
                            global_index(P, N_ex, N_ey, m, n, k, l)))
        data = A_e[m, n, i, j, r, s, k, l]
        A = sparse.COO(coords, data, shape=((P*N_ex+1)*(P*N_ey+1),)*3)
    return A


def scatter(u: np.ndarray, P: int, N_ex: int, N_ey: int):
    """
    Returns element coefficients array from global coefficients vector.\n
    :param u: global coefficients vector uₚ[p]
    :param P: polynomial order
    :param N_ex: num elements in x direction
    :param N_ey: ... in y direction
    :return: uᵐⁿᵢⱼ[m,n,i,j]
    """
    if u.shape[0] != (P*N_ex+1)*(P*N_ey+1):
        raise ValueError('Not a valid combination of global coefficients vector, P, N_ex, and N_ey')

    u_e = np.zeros((N_ex, N_ey, P+1, P+1))
    for m in range(N_ex):
        for n in range(N_ey):
            for i in range(P+1):
                for j in range(P+1):
                    u_e[m, n, i, j] = u[global_index(P, N_ex, N_ey, m, n, i, j)]
    return u_e


def global_mass_matrix(P: int, N_ex: int, N_ey: int, dx: float, dy: float) -> sp_sparse.csr_matrix:
    """
    Returns global mass matrix in SciPy-CSR format.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    """
    M_s = GLL.standard_mass_matrix(P)
    M_ex = np.multiply.outer(np.full(N_ex, dx/2), M_s)
    M_ey = np.multiply.outer(np.full(N_ey, dy/2), M_s)
    M_e = np.einsum('mik,njl->mnijkl', M_ex, M_ey, optimize=True)
    return assemble(M_e)


def global_stiffness_matrix(P: int, N_ex: int, N_ey: int, dx: float, dy: float) -> sp_sparse.csr_matrix:
    """
    Returns global stiffness matrix in SciPy-CSR format.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    """
    M_s = GLL.standard_mass_matrix(P)
    K_s = GLL.standard_stiffness_matrix(P)
    M_ex = np.multiply.outer(np.full(N_ex, dx/2), M_s)
    M_ey = np.multiply.outer(np.full(N_ey, dy/2), M_s)
    K_ex = np.multiply.outer(np.full(N_ex, 2/dx), K_s)
    K_ey = np.multiply.outer(np.full(N_ey, 2/dy), K_s)
    K_e = np.einsum('mik,njl->mnijkl', K_ex, M_ey, optimize=True)\
        + np.einsum('mik,njl->mnijkl', M_ex, K_ey, optimize=True)
    return assemble(K_e)


def global_gradient_matrices(P: int, N_ex: int, N_ey: int, dx: float, dy: float) \
        -> typing.Tuple[sp_sparse.csr_matrix, sp_sparse.csr_matrix]:
    """
    Returns global gradient matrices in SciPy-CSR format.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    :return: G_x, G_y
    """
    M_s = GLL.standard_mass_matrix(P)
    G_s = GLL.standard_gradient_matrix(P)
    M_ex = np.multiply.outer(np.full(N_ex, dx/2), M_s)
    M_ey = np.multiply.outer(np.full(N_ey, dy/2), M_s)
    G_x_e = np.einsum('m,ik,njl->mnijkl', np.ones(N_ex), G_s, M_ey, optimize=True)
    G_y_e = np.einsum('mik,n,jl->mnijkl', M_ex, np.ones(N_ey), G_s, optimize=True)
    return assemble(G_x_e), assemble(G_y_e)


def global_convection_matrices(P: int, N_ex: int, N_ey: int, dx: float, dy: float) \
        -> typing.Tuple[sparse.COO, sparse.COO]:
    """
    Returns global convection matrices in Sparse-COO format.\n
    To return (C_x @ u) in SciPy-CSR format perform 'sparse.tensordot(C_x,u,(2,0),return_type=sparse.COO).tocsr()'.\n
    To return (u @ C_x) in SciPy-CSR format perform 'sparse.tensordot(C_x,u,(1,0),return_type=sparse.COO).tocsr()'.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    :return: C_x, C_y
    """
    F_s = GLL.standard_product_matrix(P)
    C_s = GLL.standard_convection_matrix(P)
    F_ex = np.multiply.outer(np.full(N_ex, dx/2), F_s)
    F_ey = np.multiply.outer(np.full(N_ey, dy/2), F_s)
    C_x_e = np.einsum('m,irk,njsl->mnijrskl', np.ones(N_ex), C_s, F_ex, optimize=True)
    C_y_e = np.einsum('mirk,n,jsl->mnijrskl', F_ey, np.ones(N_ey), C_s, optimize=True)
    return assemble(C_x_e), assemble(C_y_e)


def eval_interpolation(u_e: np.ndarray, points_e: np.ndarray, points_plot: typing.Tuple[np.ndarray, np.ndarray]):
    """
    Returns the evaluation of u in the points points_plot.\n
    :param u_e: element coefficients array uᵐⁿₖₗ[m,n,i,j]
    :param points_e: element nodes array [xᵐⁿₖₗ[m,n,k,l],yᵐⁿₖₗ[m,n,k,l]]
    :param points_plot: evaluation points (xᵢⱼ[i,j],yᵢⱼ[i,j]) as ij-indexed meshgrid
    :return: u(xᵢⱼ,yᵢⱼ)[i,j]
    """
    N_ex = u_e.shape[0]
    N_ey = u_e.shape[1]
    P = u_e.shape[2]-1
    x_e = points_e[0, :, 0, :, 0]
    y_e = points_e[1, 0, :, 0, :]
    dx = x_e[0, -1] - x_e[0, 0]
    dy = y_e[0, -1] - y_e[0, 0]
    m_plot, xi_plot = x2xi(points_plot[0][:, 0], dx)
    n_plot, eta_plot = x2xi(points_plot[1][0, :], dy)
    val = np.zeros((points_plot[0].shape[0], points_plot[0].shape[1]))
    for m in range(N_ex):
        for n in range(N_ey):
            np.place(val, np.outer(m_plot == m, n_plot == n),
                     np.einsum('kl,ik,jl->ij',
                               u_e[m, n],
                               GLL.standard_evaluation_matrix(P, xi_plot[m_plot == m]),
                               GLL.standard_evaluation_matrix(P, eta_plot[n_plot == n])))
    return val
