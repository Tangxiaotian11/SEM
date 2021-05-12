# Examples
TODO
## HELMHOLTZ Equation
Solves the dimensionless HELMHOLTZ-Equation equation for 𝑢(x,y) with given paramter λ := 1, function 

𝑓(x,y) := (λ + (π/𝐿ˣ)² + (2π/𝐿ʸ)²) (cos(π 𝑥/𝐿ˣ)cos(2π 𝑦/𝐿ʸ))

in the rectangular domain 𝛺 := [0,𝐿ˣ]×[0,𝐿ʸ], and homogeneous NEUMANN boundary conditions

λ𝑢 = ∇²𝑢 + 𝑓 ∀ (𝑥,𝑦)∈𝛺 and ∂𝑢/∂𝑛 = 0 ∀ (𝑥,𝑦)∈∂𝛺.

The exact solution reads

𝑢ₑₓ = cos(π 𝑥/𝐿ˣ)cos(2π 𝑦/𝐿ʸ).

The discretized equation then reads

𝑯 𝒖 := 𝑴 𝒖 + 𝑲 𝒖 = 𝑴 𝒇.

The example script `Helmholtz_SciPy.py` discretizes the equation and solves them using SciPy.
The main procedure is
1. importing modules and setting the parameters

    ```python
    import numpy as np
    import scipy.sparse.linalg as sp_sparse_linalg
    import SEM   
 
    L_x = 2     # length in x direction
    L_y = 1     # length in y direction
    lam = 1     # HELMHOLTZ parameter != 0
    P = 4       # polynomial order
    N_ex = 2    # num of elements in x direction
    N_ey = 3    # num of elements in y direction
    exact = lambda x, y: np.cos(x/L_x*np.pi)*np.cos(y/L_y*2*np.pi)  # exact solution
    f = lambda x, y: (lam + (np.pi/L_x)**2 + (2*np.pi/L_y)**2)*exact(x, y)  # f(x,y)
    ```
    
1. calling the grid

    ```python
    dx = L_x / N_ex
    dy = L_y / N_ey
    points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)
    ```
    
1. calling the matrices and setting the LHS matrix and the RHS vector
    
    ```python
    M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
    K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
 
    H = lam * M + K
    g = M @ f(points[0], points[1])
    ```
    
1. solving with SciPy
    
    ```python
    u, info = sp_sparse_linalg.cg(H, g)
    ```
## NAVIER-STOKES Equations
TODO
## BOUSSINESQ Equations