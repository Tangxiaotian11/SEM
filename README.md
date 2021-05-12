# SEM

<p align="center">
    <img src="https://raw.githubusercontent.com/SEhrm/SEM/master/Examples/NavierStokes_SciPy.png" width="400" height="400" alt="test"/>
</p>
Continuous GALERKIN spectral element method with GAUSS-LEGENDRE-LOBATTO nodal LAGRANGE polynomial base.

## Description

SEM provides Python scripts of high readability for generating global system matrices of discretized partial differential equations.
Discretization is performed using the continuous GALERKIN spectral element method with GAUSS-LEGENDRE-LOBATTO nodal LAGRANGE polynomial base.

The  main files are
* `GLL.py` provides functions from the GAUSS-LEGENDRE-LOBATTO nodal LAGRANGE polynomial base, e.g. the _standard stiffness matrix_.
* `SEM.py` provides functions from the continuous GALERKIN spectral element method, e.g. the _global stiffness matrix_

The scripts are _not_ meant to be of high performance or of high versatility.

Theory mainly follows the CFD lecture given at [TU Dresden, chair of fluid mechanics](https://tu-dresden.de/ing/maschinenwesen/ism/psm) in 2019.
Additional theoretical background can be read at
* [ KARNIADAKIS, G.; SHERWIN, S. _Spectral/hp element methods for computational
fluid dynamics_](https://doi.org/https://doi.org/10.1093/acprof:oso/9780198528692.001.0001)
* [DEVILLE, M. O.; FISCHER, P. F. _High-Order Methods for Incompressible Fluid Flow_](https://doi.org/https://doi.org/10.1017/CBO9780511546792)
* [GIRALDO, F. X. _An Introduction to Element-Based Galerkin Methods on TensorProduct Bases_](https://doi.org/https://doi.org/10.1007/978-3-030-55069-1)


## Getting Started

### Dependencies
The main files require
* [NumPy](https://www.numpy.org)
* [SciPy](https://scipy.org/)
* [Sparse](https://github.com/pydata/sparse)

The provided examples may additionally require
* [Matplotlib](https://matplotlib.org/)
* [OpenMDAO](https://openmdao.org/)

### Installing

TODO

## Usage

Given an exaplary differential equation for 𝑢 with given functions 𝑣₁, 𝑣₂ and 𝑓 in the domain 𝛺, and homogeneous NEUMANN boundary conditions

𝑢 + ∂𝑢/∂𝑥 + ∂𝑢/∂𝑦 + 𝑣₁ ∂𝑢/∂𝑥 + 𝑣₂ ∂𝑢/∂𝑦 = ∇²𝑢 + 𝑓 ∀ (𝑥,𝑦)∈𝛺 and ∂𝑢/∂𝑛 = 0 ∀ (𝑥,𝑦)∈∂𝛺.

The discretized equation reads

𝑴 𝒖 + 𝑮ˣ 𝒖 + 𝑮ʸ 𝒖 + 𝒗₁ 𝑪ˣ 𝒖 + 𝒗₂ 𝑪ʸ 𝒖 + 𝑲 𝒖 = 𝑴 𝒇.

The vectors 𝒖, 𝒗₁, 𝒗₂, and 𝒇 are the evaluations of the functions 𝑢, 𝑣₁, 𝑣₂ and 𝑓 at the global grid points (𝒙,𝒚).
The matrices 𝑴, 𝑮ˣ/𝑮ʸ, 𝑪ˣ/𝑪ʸ, 𝑲 are called the _global mass_, _gradient_, _convection_ and _stiffness matrices_.

The Cartesian grid is defined by the polynomial order `P` of the base,
the number of elements in 𝑥 and 𝑦 direction `N_ex` and `N_ey`, 
and the element width in 𝑥 and 𝑦 direction `dx` and `dy`

The grid points are called by
```python
points = SEM.global_nodes(P, N_ex, N_ey, dx, dy)
```

The global matrices are likewise called by
```python
M = SEM.global_mass_matrix(P, N_ex, N_ey, dx, dy)
K = SEM.global_stiffness_matrix(P, N_ex, N_ey, dx, dy)
G_x, G_y = SEM.global_gradient_matrices(P, N_ex, N_ey, dx, dy)
C_x, C_y = SEM.global_convection_matrices(P, N_ex, N_ey, dx, dy)
```

## Examples

TODO

## Authors

Simon Ehrmanntraut

## Acknowledgments

Special thanks go to

* [PD. Dr. Stiller (TU Dresden)](https://tu-dresden.de/ing/maschinenwesen/ism/psm/die-professur/beschaeftigte/pd-dr-ing-habil-joerg-stiller)
* [Dr. Stück and Mr. Gottfried (DLR)](https://www.dlr.de/sp/en/desktopdefault.aspx/tabid-12176/21361_read-53975/)