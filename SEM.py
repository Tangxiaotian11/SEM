"""
Continuous GALERKIN spectral element method
"""
import typing
import numpy as np
import GLL
import scipy.interpolate as sp_interp
import scipy.sparse as sp_sparse
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices


def xi2x(e: int, xi: float, dx: float) -> float:
    """
    Returns physical coordinate x from standard coordinate xi in element e. This function is vectorized.\n
    :param e: element number
    :param xi: standard coordinate
    :param dx: element width
    """
    return dx/2 * (xi+1) + dx*e


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
    return np.unique(np.ravel(x_e))  # TODO check round off error


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
        coords = global_index(P, N_ex, N_ey, m, n, i, j)
        data = A_e[m, n, i, j]
        A = sp_sparse.coo_matrix((data, coords), shape=((P*N_ex+1)*(P*N_ey+1)))
        A = A.toarray()
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


def element_mass_matrix_1d(P: int, N_ex: int, dx: float):
    """
    Returns element mass matrix for one dimension.\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param dx: element width
    :return: Mᵐᵢₖ[m,i,k]
    """
    M_s = GLL.standard_mass_matrix(P)
    return np.einsum('m,ik->mik', dx/2*np.ones(N_ex), M_s)


def element_stiffness_matrix_1d(P: int, N_ex: int, dx: float):
    """
    Returns element stiffness matrix for one dimension.\n
    :param P: polynomial order
    :param N_ex: num of elements
    :param dx: element width
    :return: Kᵐᵢₖ[m,i,k]
    """
    M_s = GLL.standard_mass_matrix(P)
    D_s = GLL.standard_differentiation_matrix(P)
    return np.einsum('m,ik->mik', 2/dx*np.ones(N_ex), D_s.transpose() @ M_s @ D_s)


def global_mass_matrix(P: int, N_ex: int, N_ey: int, dx: float, dy: float) -> sp_sparse.csr_matrix:
    """
    Returns global mass matrix in SciPy-CSR format.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    """
    M_ex = element_mass_matrix_1d(P, N_ex, dx)
    M_ey = element_mass_matrix_1d(P, N_ey, dy)
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
    M_ex = element_mass_matrix_1d(P, N_ex, dx)
    M_ey = element_mass_matrix_1d(P, N_ey, dy)
    K_ex = element_stiffness_matrix_1d(P, N_ex, dx)
    K_ey = element_stiffness_matrix_1d(P, N_ey, dy)
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
    M_ex = element_mass_matrix_1d(P, N_ex, dx)
    M_ey = element_mass_matrix_1d(P, N_ey, dy)
    G_s = GLL.standard_gradient_matrix(P)
    G_x_e = np.einsum('m,ik,njl->mnijkl', np.ones(N_ex), G_s, M_ey, optimize=True)
    G_y_e = np.einsum('mik,n,jl->mnijkl', M_ex, np.ones(N_ey), G_s, optimize=True)
    return assemble(G_x_e), assemble(G_y_e)


def global_convection_matrices(P: int, N_ex: int, N_ey: int, dx: float, dy: float) \
        -> typing.Tuple[sparse.COO, sparse.COO]:
    """
    Returns global gradient matrices in Sparse-COO format.\n
    To return (C_x @ u) in SciPy-CSR format perform 'sparse.tensordot(C_x,u,(2,0),return_type=sparse.COO).tocsr()'.\n
    To return (u @ C_x) in SciPy-CSR format perform 'sparse.tensordot(C_x,u,(1,0),return_type=sparse.COO).tocsr()'.\n
    :param P: polynomial order
    :param N_ex: num of elements in x direction
    :param N_ey: ... in y direction
    :param dx: element width in x direction
    :param dy: ... in y direction
    :return: C_x, C_y
    """
    M_ex = element_mass_matrix_1d(P, N_ex, dx)
    M_ey = element_mass_matrix_1d(P, N_ey, dy)
    C_s = GLL.standard_convection_matrix(P)
    C_x_e = np.einsum('m,irk,sl,njl->mnijrskl', np.ones(N_ex), C_s, np.identity(P+1), M_ey, optimize=True)
    C_y_e = np.einsum('rk,nik,m,jsl->mnijrskl', np.identity(P+1), M_ex, np.ones(N_ey), C_s, optimize=True)
    return assemble(C_x_e), assemble(C_y_e)


def eval_interpolation(u_e: np.ndarray, points_e: np.ndarray, points_plot: typing.Tuple[np.ndarray, np.ndarray]):
    """
    Returns the evaluation of u in the points points_plot.\n
    :param u_e: element coefficients array uᵐⁿₖₗ[m,n,i,j]
    :param points_e: element nodes array [xᵐⁿₖₗ[m,n,k,l],yᵐⁿₖₗ[m,n,k,l]]
    :param points_plot: evaluation points (xᵢⱼ[i,j],yᵢⱼ[i,j])
    :return: u(xᵢⱼ,yᵢⱼ)[i,j]
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
