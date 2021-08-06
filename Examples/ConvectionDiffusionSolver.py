import sys, os
sys.path.append(os.getcwd() + '/..')
import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as linalg
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import SEM
import time
import matplotlib.pyplot as plt


class ConvectionDiffusionSolver:
    def __init__(self, L_x: float, L_y: float, Pe: float, P: int, N_ex: int, N_ey: int,
                 T_W: float = None, T_E: float = None, T_S: float = None, T_N: float = None,
                 mtol=1e-7, iprint: list[str] = []):
        """
        Solves the steady-state convection-diffusion equation for T(x,y) given
        the convection velocities u(x,y) and v(x,y)\n
        Pe [u, v]∘∇T = ∇²T ∀(x,y)∈[0,L_x]×[0,L_y]\n
        with either DIRICHLET or homogenous NEUMANN conditions\n
        T(0,y)   = T_W or ∂ₙT(0,y)   = 0 ∀y∈[0,L_y]\n
        T(L_x,y) = T_E or ∂ₙT(L_x,y) = 0 ∀y∈[0,L_y]\n
        T(x,0)   = T_S or ∂ₙT(x,0)   = 0 ∀x∈[0,L_x]\n
        T(x,L_y) = T_N or ∂ₙT(x,L_y) = 0 ∀x∈[0,L_x]\n
        :param L_x: length in x direction
        :param L_y: length in y direction
        :param Pe: PECLET number
        :param P: polynomial order
        :param N_ex: num of elements in x direction
        :param N_ey: num of elements in y direction
        :param T_W: DIRICHLET value or None for homogeneous NEUMANN
        :param T_E: DIRICHLET value or None for homogeneous NEUMANN
        :param T_S: DIRICHLET value or None for homogeneous NEUMANN
        :param T_N: DIRICHLET value or None for homogeneous NEUMANN
        :param mtol: tolerance on root mean square residuals for JACOBI inversion
        :param iprint: list of infos to print TODO desc
        """
        self._iprint = iprint

        self._Pe = Pe
        self._mtol = mtol

        # grid
        self._L_x = L_x
        self._L_y = L_y
        self.points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
        self.N = (N_ex*P+1)*(N_ey*P+1)

        # global matrices
        dx = self._L_x / N_ex
        dy = self._L_y / N_ey
        self._M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
        self._K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
        self._C_x, self._C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)

        self._Sys = None  # system matrix from last _calc_system call
        self._Jac_T_u = None  # JACOBIans from last _calc_jacobians call
        self._Jac_T_v = None

        # DIRICHLET values and mask
        self._dirichlet = np.full(self.N, np.nan)
        if T_W is not None:
            self._dirichlet[np.isclose(self.points[0], 0)] = T_W
        if T_E is not None:
            self._dirichlet[np.isclose(self.points[0], self._L_x)] = T_E
        if T_S is not None:
            self._dirichlet[np.isclose(self.points[1], 0)] = T_S
        if T_N is not None:
            self._dirichlet[np.isclose(self.points[1], self._L_y)] = T_N
        self._mask_dir = ~np.isnan(self._dirichlet)

    def _get_residuals(self, T, u, v):
        """
        Returns residual with precalculated system matrices\n
        :param T: temperature as global vector
        :return: res as global vector
        """
        # left-hand-side multiplication of convection velocities, i.e. Pe*(u @ C_x + v @ C_y)
        Conv = self._Pe * (sparse.tensordot(self._C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                         + sparse.tensordot(self._C_y, v, (1, 0), return_type=sparse.COO).tocsr())
        # system matrix
        self._Sys = Conv + self._K

        res = self._Sys @ T

        # apply DIRICHLET conditions
        res[self._mask_dir] = T[self._mask_dir] - self._dirichlet[self._mask_dir]

        return res

    def _calc_jacobians(self, T):
        """
        Precalculates JACOBIans\n
        :param T: temperature as global vector\n
        """
        # JACOBI matrices including chain rule but excluding DIRICHLET BC
        # right-hand-side multiplication of T, i.e. Pe*(C_x @ T)
        self._Jac_T_u = self._Pe * sparse.tensordot(self._C_x, T, (2, 0), return_type=sparse.COO).tocsr()
        self._Jac_T_v = self._Pe * sparse.tensordot(self._C_y, T, (2, 0), return_type=sparse.COO).tocsr()

    def _get_dresiduals(self, dT, du=None, dv=None):
        """
        Returns residual differential with precalculated JACOBIans\n
        :param dT: T differential as global vector
        :param du: u differential as global vector
        :param dv: v differential as global vector
        :return: dres as global vector
        """
        dres = self._Sys @ dT
        if du is not None:
            dres += self._Jac_T_u @ du
        if dv is not None:
            dres += self._Jac_T_v @ dv

        # apply DIRICHLET conditions
        dres[self._mask_dir] = dT[self._mask_dir]

        return dres

    def _get_update(self, dres, dT0=None):
        """
        Returns temperature differential for given residual differential with precalculated JAVOBIans\n
        :param dres: residual differential as global vector
        :param dT0: guess for T differential as global vector
        :return: dT as global vector
        """
        atol = self._mtol * np.sqrt(self.N)

        # LHS
        def lhs_mv(dT):
            lhs_mv.fCount += 1
            return self._get_dresiduals(dT)
        lhs_mv.fCount = 0
        lhs_LO = linalg.LinearOperator((self.N,)*2, lhs_mv, dtype=float)

        # precon
        # def precon_mv(c):
        #     z = np.zeros(self.N)
        #     z[self._mask_dir] = c[self._mask_dir]
        #     c[~self._mask_dir] -= self._K[~self._mask_dir, :][:, self._mask_dir] @ c[self._mask_dir]
        #     z[~self._mask_dir], info = linalg.cg(self._K[~self._mask_dir, :][:, ~self._mask_dir],
        #                                          c[~self._mask_dir], atol=atol*1e-1, tol=0)
        #     if info != 0:
        #         raise RuntimeError(f'ConvectionDiffusion precon CG: Failed to converge in {info} iterations')
        #     return z
        # precon_LO = linalg.LinearOperator((self.N,)*2, precon_mv, dtype=float)

        # solve
        def print_res(xk):
            print_res.iterCount += 1
            resk = np.linalg.norm(lhs_LO.matvec(xk) - dres)
            if 'LGMRES_iter' in self._iprint:
                print(f'ConvectionDiffusion LGMRES: {print_res.iterCount}\t{resk}')
        print_res.iterCount = 0

        dT, info = linalg.lgmres(A=lhs_LO, b=dres, M=None, x0=dT0,
                                 atol=atol, tol=0, inner_m=int(self.N*0.1), callback=print_res)
        if info != 0:
            raise RuntimeError(f'ConvectionDiffusion LGMRES: Failed to converge in {info} iterations')

        if 'LGMRES_suc' in self._iprint:
            res = np.linalg.norm(lhs_LO.matvec(dT) - dres, ord=np.inf)
            print(f'ConvectionDiffusion LGMRES: Converged in {lhs_mv.fCount} evaluations with max-norm {res}')

        return dT

    def get_solution(self, u_func, v_func, T0=None):
        """
        Returns Temperature solution
        :param u_func: u(x,y) as vectorized function
        :param v_func: v(x,y) as vectorized function
        :param T0: guess for T as global vector
        :return: T as global vector
        """
        if T0 is None:
            T0 = np.zeros(self.N)
        u = u_func(self.points[0], self.points[1]) if u_func is not None else np.zeros(self.N)
        v = v_func(self.points[0], self.points[1]) if v_func is not None else np.zeros(self.N)

        res = self._get_residuals(T0, u, v)
        dT = self._get_update(-res)  # single NEWTON step because problem is already linear
        return T0 + dT


if __name__ == "__main__":
    # Example
    # input
    L_x = 1
    L_y = 1
    P = 4
    N_ex = 16
    N_ey = 16
    Pe = 40

    # plotting points
    points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
    points_e = SEM.element_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)

    cd = ConvectionDiffusionSolver(L_x, L_y, Pe, P, N_ex, N_ey, T_E=-0.5, T_W=0.5)

    u = lambda x, y: (y-0.5)
    v = lambda x, y: -(x-0.5)
    T = cd.get_solution(u, v)

    # scatter for plot
    T_e = SEM.scatter(T, P, N_ex, N_ey)

    # plot
    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 51), np.linspace(0, L_y, 51), indexing='ij')
    T_plot = SEM.eval_interpolation(T_e, points_e, (x_plot, y_plot))
    fig = plt.figure(figsize=(L_x*4, L_y*4))
    ax = fig.gca()
    CS = ax.contour(x_plot, y_plot, T_plot, levels=11, colors='k', linestyles='solid')
    ax.streamplot(x_plot.T, y_plot.T, u(x_plot, y_plot).T, v(x_plot, y_plot).T, density=1)
    ax.clabel(CS, inline=True)
    ax.set_title(f"P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={cd._mtol:.0e}", fontsize='small')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()