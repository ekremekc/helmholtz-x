# helmholtz-x

This python library implements the complex number version of Helmholtz solver using finite element method.

It is using extensive parallelization with handled preallocations for generation of nonlinear part of the thermoacoustic Helmlholtz equation.

The nonlinear eigenvalue problem is solved using [PETSc](https://petsc.org/release/overview/), [SLEPc](https://slepc.upv.es/) and [FEniCSx](https://github.com/FEniCS/dolfinx) libraries. You can get packaged installation of those using Docker. 

### Docker images

To run a Docker image with the latest release of DOLFINx:

```shell
docker run -ti dolfinx/dolfinx:stable
```

To switch between real and complex builds of DOLFINx/PETSc.

```shell
source /usr/local/bin/dolfinx-complex-mode
source /usr/local/bin/dolfinx-real-mode

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