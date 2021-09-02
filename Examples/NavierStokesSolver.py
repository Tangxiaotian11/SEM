import sys, os
sys.path.append(os.getcwd() + '/..')
import typing
import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as linalg
import sparse  # this package is exclusively used for three dimensional sparse matrices, i.e. the convection matrices
import SEM
import time
import matplotlib.pyplot as plt

class NavierStokesSolver:
    def __init__(self, L_x: float, L_y: float, Re: float, Gr: float, P: int, N_ex: int, N_ey: int,
                 v_W: float = 0, v_E: float = 0, u_S: float = 0, u_N: float = 0,
                 mtol=1e-7, mtol_newton=1e-5, iprint: list = ['NEWTON_suc', 'NEWTON_iter']):
        """
        Solves the steady-state NAVIER-STOKES equation for u(x,y) and v(x,y) given the temperature T(x,y)\n
        Re([u, v]∘∇)[u, v] = -∇p + ∇²[u, v] + Gr/Re [0, T] ∀(x,y)∈[0,L_x]×[0,L_y]\n
        ∇∘[u, v] = 0 ∀(x,y)∈[0,L_x]×[0,L_y]\n
        with no normal flow and tangential DIRICHLET conditions\n
        v(0,y)   = v_W y∈[0,L_y]\n
        v(L_x,y) = v_E y∈[0,L_y]\n
        u(x,0)   = u_S x∈[0,L_x]\n
        u(x,L_y) = u_N x∈[0,L_x]\n
        Artificial NEUMANN boundary condition for the pressure\n
        ∂ₙp = 0 ∀(x,y)∈∂([0,L_x]×[0,L_y])\n
        :param L_x: length in x direction
        :param L_y: length in y direction
        :param Re: REYNOLDS numver
        :param Gr: GRASHOF number
        :param P: polynomial order
        :param N_ex: num of elements in x direction
        :param N_ey: num of elements in y direction
        :param v_W: DIRICHLET value
        :param v_E: DIRICHLET value
        :param u_S: DIRICHLET value
        :param u_N: DIRICHLET value
        :param mtol: tolerance on root mean square residuals for JACOBI inversion
        :param mtol_newton: tolerance on root mean square residuals for NEWTON
        :param iprint: list of infos to print TODO desc
        """
        self._iprint = iprint

        self._Re = Re
        self._Gr = Gr
        if self._Re == 0 and self._Gr != 0:
            raise ValueError('Cannot have Re == 0 and Gr != 0')
        self._Gr_over_Re = self._Gr / self._Re if self._Re != 0 else 0.

        self._mtol = mtol
        self._mtol_newton = mtol_newton

        # grid
        self._L_x = L_x
        self._L_y = L_y
        self._P = P
        self._N_ex = N_ex
        self._N_ey = N_ey
        dx = L_x / N_ex
        dy = L_y / N_ey
        self.points = SEM.global_nodes(P, N_ex, N_ey, L_x/N_ex, L_y/N_ey)
        self.points_e = SEM.element_nodes(P, N_ex, N_ey, dx, dy)
        self.N = (N_ex*P+1)*(N_ey*P+1)

        # global matrices
        self._M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
        self._K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
        self._G_x, self._G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, dx, dy)
        self._C_x, self._C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)

        self._Sys = None  # system matrix from last _calc_system call
        self._Jac_u_u = None  # JACOBIans from last _calc_jacobians call
        self._Jac_u_v = None
        self._Jac_v_u = None
        self._Jac_v_v = None

        # DIRICHLET values and masks
        self._dirichlet_u = np.full(self.N, np.nan)
        self._dirichlet_v = np.full(self.N, np.nan)
        self._dirichlet_p = np.full(self.N, np.nan)
        self._dirichlet_v[np.isclose(self.points[0], 0)] = v_W
        self._dirichlet_u[np.isclose(self.points[0], 0)] = 0
        self._dirichlet_v[np.isclose(self.points[0], self._L_x)] = v_E
        self._dirichlet_u[np.isclose(self.points[0], self._L_x)] = 0
        self._dirichlet_u[np.isclose(self.points[1], 0)] = u_S
        self._dirichlet_v[np.isclose(self.points[1], 0)] = 0
        self._dirichlet_u[np.isclose(self.points[1], self._L_y)] = u_N
        self._dirichlet_v[np.isclose(self.points[1], self._L_y)] = 0
        self._dirichlet_p[int(self.N / 2)] = 0  # reference pressure at approx center
        self._mask_bound = ~np.isnan(self._dirichlet_u)
        self._mask_dir_p = ~np.isnan(self._dirichlet_p)

    def _get_residuals(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, T: np.ndarray)\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns momentum and continuity residuals\n
        :param u: u as global vector
        :param v: v as global vector
        :param p: p as global vector
        :return: res_u, res_v, res_cont as global vectors
        """
        # left-hand-side multiplication of convection velocities, i.e. Re*(u @ C_x + v @ C_y)
        Conv = self._Re * (sparse.tensordot(self._C_x, u, (1, 0), return_type=sparse.COO).tocsr()
                           + sparse.tensordot(self._C_y, v, (1, 0), return_type=sparse.COO).tocsr())
        # system matrix
        self._Sys = self._K + Conv
        self._Bouyancy = self._Gr_over_Re * self._M @ T

        res_u = self._Sys @ u + self._G_x @ p
        res_v = self._Sys @ v + self._G_y @ p - self._Bouyancy
        res_cont = self._G_x @ u + self._G_y @ v

        # apply DIRICHLET conditions
        res_u[self._mask_bound] = u[self._mask_bound] - self._dirichlet_u[self._mask_bound]
        res_v[self._mask_bound] = v[self._mask_bound] - self._dirichlet_v[self._mask_bound]
        res_cont[self._mask_dir_p] = p[self._mask_dir_p] - self._dirichlet_p[self._mask_dir_p]

        # apply artificial homogeneous NEUMANN pressure condition
        res_cont[self._mask_bound] = self._K[self._mask_bound, :] @ p

        return res_u, res_v, res_cont

    def _calc_jacobians(self, u: np.ndarray, v: np.ndarray):
        """
        Precalculates JACOBIans\n
        :param u: u as global vector
        :param v: v as global vector
        """
        # JACOBI matrices including chain rule but excluding DIRICHLET BC
        # right-hand-side multiplication of the velocities, i.e. Re * C_x @ u
        self._Jac_u_u = self._Sys \
                        + self._Re * sparse.tensordot(self._C_x, u, (2, 0), return_type=sparse.COO).tocsr()
        self._Jac_v_v = self._Sys \
                        + self._Re * sparse.tensordot(self._C_y, v, (2, 0), return_type=sparse.COO).tocsr()
        self._Jac_u_v = self._Re * sparse.tensordot(self._C_y, u, (2, 0), return_type=sparse.COO).tocsr()
        self._Jac_v_u = self._Re * sparse.tensordot(self._C_x, v, (2, 0), return_type=sparse.COO).tocsr()

    def _get_dresiduals(self, du: np.ndarray, dv: np.ndarray, dp: np.ndarray, dT: np.ndarray = None)\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns momentum and continuity residual differentials with precalculated JACOBIans\n
        :param du: u differential as global vector
        :param dv: v differential as global vector
        :param dp: p differential as global vector
        :param dT: T differential as global vector
        :return: dres_u, dres_v, dres_cont as global vectors
        """
        dres_u = self._Jac_u_u @ du + self._Jac_u_v @ dv + self._G_x @ dp
        dres_v = self._Jac_v_u @ du + self._Jac_v_v @ dv + self._G_y @ dp
        dres_cont = self._G_x @ du + self._G_y @ dv
        if dT is not None:
            dres_v += -self._Gr_over_Re * self._M @ dT

        # apply DIRICHLET and artificial NEUMANN pressure conditions
        dres_u[self._mask_bound] = du[self._mask_bound]
        dres_v[self._mask_bound] = dv[self._mask_bound]
        dres_cont[self._mask_bound] = self._K[self._mask_bound, :] @ dp
        dres_cont[self._mask_dir_p] = dp[self._mask_dir_p]

        return dres_u, dres_v, dres_cont

    def _get_update(self, dres_u: np.ndarray, dres_v: np.ndarray, dres_cont: np.ndarray,
                    du0: np.ndarray = None, dv0: np.ndarray = None, dp0: np.ndarray = None)\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns velocity and pressure differentials for given momentum and continuity residual differentials with
        precalculated JAVOBIans\n
        :param dres_u: x-momentum residual differential as global vector
        :param dres_v: y-momentum residual differential as global vector
        :param dres_cont: continuity residual differential as global vector
        :param du0: guess for u differential as global vector
        :param dv0: guess for v differential as global vector
        :param dp0: guess for p differential as global vector
        :return: du, dv, dp as global vectors
        """
        # == Jac_velo solver == # TODO replace with QMR
        tStart = time.perf_counter()
        mask = np.hstack((self._mask_bound,) * 2)
        Jac_velo = sp_sparse.bmat([[self._Jac_u_u, self._Jac_u_v],
                                   [self._Jac_v_u, self._Jac_v_v]], format='lil')
        Jac_velo[mask, :] = 0  # apply DIRICHLET condition
        Jac_velo[mask, mask] = 1  # ...
        Jac_velo = Jac_velo.tocsc()
        Jac_velo_lu = linalg.splu(Jac_velo)
        if 'LU_suc' in self._iprint:
            print(f'NavierStokes LU: Succeeded in {time.perf_counter()-tStart:0.2f}sec '
                  f'with fill factor {Jac_velo_lu.nnz/Jac_velo.nnz:0.1f}')

        def solve_jac_velo(dres_u, dres_v):
            dres_uv = np.hstack((dres_u, dres_v))
            duv = Jac_velo_lu.solve(dres_uv)
            return np.split(duv, 2)

        # == solve for pressure ==
        # RHS
        b_schur_u, b_schur_v = solve_jac_velo(dres_u, dres_v)
        b_schur = -(self._G_x @ b_schur_u + self._G_y @ b_schur_v)
        b_schur[self._mask_bound] = 0  # apply NEUMANN pressure conditions
        b_schur[self._mask_dir_p] = 0  # apply DIRICHLET pressure conditions
        b_schur += dres_cont

        # LHS
        def schur_mv(dp):
            schur_mv.fCount += 1
            # apply gradient
            f_x = self._G_x @ dp
            f_y = self._G_y @ dp
            # apply DIRICHLET velocity condition
            f_x[self._mask_bound] = f_y[self._mask_bound] = 0
            # apply inverse
            f_x, f_y = solve_jac_velo(f_x, f_y)
            # apply divergence
            f = -(self._G_x @ f_x + self._G_y @ f_y)
            # apply artificial NEUMANN pressure condition
            f[self._mask_bound] = self._K[self._mask_bound, :] @ dp
            # apply DIRICHLET pressure condition
            f[self._mask_dir_p] = dp[self._mask_dir_p]
            return f
        schur_mv.fCount = 0
        schur_LO = linalg.LinearOperator((self.N,)*2, schur_mv, dtype=float)

        # preconditioner
        def precon_mv(c):  # mass precon
            z = c/self._M.diagonal()
            z[self._mask_dir_p] = c[self._mask_dir_p]
            return z
        precon_LO = linalg.LinearOperator((self.N,)*2, precon_mv, dtype=float)

        # solve
        def print_res(xk):
            print_res.iterCount += 1
            if 'LGMRES_iter' in self._iprint:
                res = np.linalg.norm(schur_LO.matvec(xk) - b_schur)
                print(f'NavierStokes LGMRES: {print_res.iterCount}\t{res}')
        print_res.iterCount = 0

        dp, info = linalg.lgmres(A=schur_LO, b=b_schur, M=precon_LO, x0=dp0,
                                 atol=self._mtol*np.sqrt(self.N), tol=0,
                                 inner_m=int(self.N*0.3), callback=print_res)  # TODO set realistic inner_m
        if info != 0:
            raise RuntimeError(f'NavierStokes LGMRES: Failed to converge in {info} iterations')

        if 'LGMRES_suc' in self._iprint:
            res = np.linalg.norm(schur_LO.matvec(dp) - b_schur, ord=np.inf)
            print(f'NavierStokes LGMRES: Converged in {schur_mv.fCount} evaluations with max-norm {res}')

        # == solve for velocities ==
        # RHS
        b_u = -self._G_x @ dp
        b_v = -self._G_y @ dp
        b_u[self._mask_bound] = 0  # apply DIRICHLET velocity condition
        b_v[self._mask_bound] = 0  # ...
        b_u += dres_u
        b_v += dres_v

        du, dv = solve_jac_velo(b_u, b_v)

        return du, dv, dp

    def _get_solution(self, T: np.ndarray, u0: np.ndarray = None, v0: np.ndarray = None, p0: np.ndarray = None)\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns nonlinear solution\n
        :param T: T as global vector
        :param u0: guess for u as global vector
        :param v0: guess for v as global vector
        :param p0: guess for p as global vector
        :return: u, v, p as global vectors
        """
        u = u0 if u0 is not None else np.zeros(self.N)
        v = v0 if v0 is not None else np.zeros(self.N)
        p = p0 if p0 is not None else np.zeros(self.N)

        self._k = 0
        while True:
            res_u, res_v, res_cont = self._get_residuals(u, v, p, T)
            norm = np.linalg.norm((res_u, res_v, res_cont), ord=2)
            if 'NEWTON_iter' in self._iprint:
                print(f'NavierStokes NEWTON: {self._k}\t{norm}')
            if norm <= self._mtol_newton*np.sqrt(self.N*3):
                if 'NEWTON_suc' in self._iprint:
                    print(f'NavierStokes NEWTON: Converged in {self._k} iterations'
                          f' with max-norm {np.linalg.norm((res_u, res_v, res_cont), ord=np.inf)}')
                break
            self._calc_jacobians(u, v)
            du, dv, dp = self._get_update(-res_u, -res_v, -res_cont)  # single NEWTON
            u += du
            v += dv
            p += dp
            self._k += 1

        return u, v, p

    def _get_vector(self, f_func: typing.Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Returns global vector from f\n
        :param f_func: f as function
        :return: f as global vector
        """
        return f_func(self.points[0], self.points[1])

    def _get_interpol(self, f: np.ndarray, points_plot: typing.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Returns interpolation of f at plotting points\n
        :param f: f as global vector
        :param points_plot: plotting points (xᵢⱼ[i,j],yᵢⱼ[i,j])
        :return: f(xᵢⱼ,yᵢⱼ)[i,j]
        """
        f_e = SEM.scatter(f, self._P, self._N_ex, self._N_ey)
        return SEM.eval_interpolation(f_e, self.points_e, points_plot)

    def run(self, T_func: typing.Callable[[np.ndarray, np.ndarray], np.ndarray],
            points_plot: typing.Tuple[np.ndarray, np.ndarray])\
            -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns solution at plotting points
        :param T_func: T as function
        :param points_plot: plotting points (xᵢⱼ[i,j],yᵢⱼ[i,j])
        :return: u(xᵢⱼ,yᵢⱼ)[i,j], v(xᵢⱼ,yᵢⱼ)[i,j], p(xᵢⱼ,yᵢⱼ)[i,j]
        """
        T = self._get_vector(T_func)
        u, v, p = self._get_solution(T)
        return self._get_interpol(u, points_plot),\
               self._get_interpol(v, points_plot),\
               self._get_interpol(p, points_plot)


if __name__ == "__main__":
    # Example: lid-driven cavity flow
    # input
    L_x = 1
    L_y = 1
    Re = 400
    P = 4
    N_ex = 16
    N_ey = 16
    iprint = ['NEWTON_suc', 'NEWTON_iter', 'LGMRES_suc', 'LGMRES_iter']
    save = False

    for i, arg in enumerate(sys.argv):
        if arg == '-L_x':
            P = float(sys.argv[i+1])
        if arg == '-L_y':
            P = float(sys.argv[i+1])
        if arg == '-P':
            P = int(sys.argv[i+1])
        if arg == '-Ne_x':
            Ne_x = int(sys.argv[i+1])
        if arg == '-Ne_y':
            Ne_y = int(sys.argv[i+1])
        if arg == '-Re':
            Re = float(sys.argv[i+1])
        if arg == '-iprint':
            iprint = sys.argv[i+1].split(',')
        if arg == '-save':
            save = bool(sys.argv[i+1])

    # plotting points
    x_plot, y_plot = np.meshgrid(np.linspace(0, L_x, 101), np.linspace(0, L_y, 101), indexing='ij')

    ns = NavierStokesSolver(L_x, L_y, Re, 0, P, N_ex, N_ey, u_N=1, iprint=iprint)

    u_plot, v_plot, p_plot = ns.run(T_func=lambda x, y: 0*x*y,
                                    points_plot=(x_plot, y_plot))

    fig = plt.figure(figsize=(L_x*4, L_y*4))
    ax = fig.gca()
    ax.streamplot(x_plot.T, y_plot.T, u_plot.T, v_plot.T, density=2)
    ax.set_title(f"Re={Re:.0e}, P={P}, N_ex={N_ex}, N_ey={N_ey}, mtol={ns._mtol_newton:.0e}", fontsize='small')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()
    if save:
        plt.savefig('tmp.png')
