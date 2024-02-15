# helmholtz-x

**helmholtz-x** is a python library which solves a non-homogeneous Helmholtz equation using finite element method (FEM).

In **helmholtz-x**, the nonlinear eigenvalue problem is determined using [PETSc](https://petsc.org/release/overview/), [SLEPc](https://slepc.upv.es/) and [FEniCSx](https://github.com/FEniCS) libraries. 

We specifically address thermoacoustic Helmholtz equation within **helmholtz-x**. In its submodules, **helmholtz-x** exploits extensive parallelization with handled preallocations for the generation of nonlinear part of the thermoacoustic Helmlholtz equation.

## Thermoacoustic Helmholtz equation

The thermoacoustic Helmholtz equation reads;

$$ \nabla\cdot\left( c^2 \nabla  \hat{p}_1 \right) + \omega^2\hat{p}_1  = -i\omega (\gamma-1)\hat{q}_1  $$

In matrix form;

$$ A\textbf{P} + \omega B\textbf{P} + \omega^2C\textbf{P} = D(\omega)\textbf{P} $$

where 
$\textbf{P}$ is the eigenvector or eigenfunction, $\omega$ is the eigenvalue ( $f$ = $\omega$ / $2\pi$ is eigenfrequency) and the discretized matrices are;

$$ A_{jk} = -\int_\Omega c^2\nabla \phi_j \cdot\nabla \phi_k dx   $$

$$ B_{jk} = \int_{\partial \Omega} \left( \frac{  ic}{Z}\right)  \phi_j\phi_k d{\sigma}   $$

$$ C_{jk} = \int_\Omega\phi_j\phi_k\ dx   $$

$$ D_{jk} = (\gamma-1) |FTF|\frac{ q_{tot}  }{ U_{bulk}} \int_{\Omega} \phi_i h(\textbf{x}) e^{i \omega \tau(\textbf{x})} d\textbf{x}  \int_{\Omega} \frac{w(\chi)}{\rho_0 (\chi)}  \nabla{\phi_j} \cdot \textbf{n}_{ref} d\chi $$

This library has the following capabilities;

- Distributed and pointwise heat release rate fields are available.
- Distributed and constant time delay fields are available.
- The Bloch boundary condition for cheap calculation of azimuthal eigenmodes is available.
- Advanced operations for mesh generation is available in standalone [gmsh-x](https://github.com/ekremekc/gmsh-x) library.

## Installation

We advise to install those packages from source in Ubuntu/Linux OS, but there is an another option: *Docker*.  

### Docker images
You can get packaged installation of dependencies counted above using Docker images. The **helmholtz-x** runs with the v0.7.3 of [DOLFINx](https://github.com/FEniCS/dolfinx). And easiest way of getting the latest DOLFINx is docker containers;

```shell
docker run -ti dolfinx/dolfinx:v0.7.3
```
for Bloch boundary condition, user should also use [DOLFINX MPC](https://github.com/jorgensd/dolfinx_mpc);

```shell
docker run -ti -v $(pwd):/root/shared -w /root/shared ghcr.io/jorgensd/dolfinx_mpc:v0.7.2
```

The code should also utilize complex builds of DOLFINx/PETSc/SLEPc and it can be activated running;

```shell
source /usr/local/bin/dolfinx-complex-mode
```

### Conda
Users may install the dependencies with conda. Here is the livestock for [PETSc/SLEPc/FEniCSx](https://fenicsproject.discourse.group/t/error-when-trying-to-solve-complex-eigenvalue-problem-in-parallel/13546/3);
```shell
conda create -n helmholtzx-env
conda activate helmholtzx-env
conda install -c conda-forge python=3.10 mpich fenics-dolfinx hdf5=1.14.2 petsc=*=complex* slepc=*=complex*
```
Then **helmholtz-x** can be installed within this conda environment.
