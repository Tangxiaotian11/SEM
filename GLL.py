"""
GAUSS-LEGENDRE-LOBATTO nodal LAGRANGE polynomial base
"""
import numpy as np


def standard_nodes(P: int):
    """
    Returns quadrature nodes ∈[−1,1], weights, and VANDERMONDE matrix of LEGENDRE polynomials\n
    :param P: polynomial order
    :return: ξᵢ[i], wᵢ[i], Pⱼ(ξᵢ)[i,j]
    """
    # NEWTON method to find nodes
    # use GAUSS-CHEBYSHEV nodes as first guess
    nodes = -np.cos(np.pi * np.arange(P+1) / P)
    # initialize VANDERMONDE matrix
    vandermonde = np.zeros((P+1, P+1), dtype=np.float64)
    # iterate NEWTON methode for root finding
    update = 1
    while np.max(np.abs(update)) > np.finfo(np.float64).eps:
        # iteratively fill VANDERMONDE matrix
        vandermonde[:, 0] = np.ones(P+1)
        vandermonde[:, 1] = nodes
        for k in range(2, P+1):
            vandermonde[:, k] = ((2*k-1) * nodes * vandermonde[:, k-1] - (k-1) * vandermonde[:, k-2])/k
        # update on guess
        update = - (nodes * vandermonde[:, P] - vandermonde[:, P-1]) / ((P+1) * vandermonde[:, P])
        nodes = nodes + update

    # weights
    weights = 2./(P*(P+1) * vandermonde[:, P]**2)

    return nodes, weights, vandermonde


def standard_mass_matrix(P: int):
    """
    Returns standard mass matrix Mˢᵢⱼ=∫ℓᵢℓⱼdξ\n
    :param P: polynomial order
    :return: Mˢᵢⱼ[i,j]
    """
    return np.diag(standard_nodes(P)[1])


def standard_differentiation_matrix(P: int):
    """
    Returns standard differentiation matrix Dˢᵢⱼ=ℓ'ⱼ(ξᵢ)\n
    :param P: polynomial order
    :return: Dˢᵢⱼ[i,j]
    """
    nodes, _, vandermonde = standard_nodes(P)
    D_s = np.zeros((P+1, P+1))
    for i in range(P+1):
        for j in range(P+1):
            if i != j:
                D_s[i, j] = vandermonde[i, -1]/vandermonde[j, -1] * 1/(nodes[i]-nodes[j])
    D_s[0, 0] = -P*(P+1)/4
    D_s[-1, -1] = P*(P+1)/4
    return D_s


def standard_gradient_matrix(P: int):
    """
    Returns standard convection matrix Gˢᵢⱼ=∫ℓᵢℓ'ⱼdξ\n
    :param P: polynomial order
    :return: Gˢᵢⱼ[i,j]
    """
    D_s = standard_differentiation_matrix(P)
    M_s = standard_mass_matrix(P)
    return M_s @ D_s


def standard_convection_matrix(P: int):
    """
    Returns standard convection matrix Cˢᵢⱼₖ=∫ℓᵢℓⱼℓ'ₖdξ\n
    :param P: polynomial order
    :return: Cˢᵢⱼₖ[i,j,k]
    """
    D_s = standard_differentiation_matrix(P)
    M_s = standard_mass_matrix(P)
    return np.einsum('ij,ik->ijk', M_s, D_s)

