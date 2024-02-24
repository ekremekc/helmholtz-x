# helmholtz-x

**helmholtz-x** is a python library which solves a non-homogeneous Helmholtz equation using finite element method (FEM).

In **helmholtz-x**, the nonlinear eigenvalue problem is determined using [PETSc](https://petsc.org/release/overview/), [SLEPc](https://slepc.upv.es/) and [FEniCSx](https://github.com/FEniCS) libraries. 

We specifically address thermoacoustic Helmholtz equation within **helmholtz-x**. In its submodules, **helmholtz-x** exploits extensive parallelization with handled preallocations for the generation of nonlinear part of the thermoacoustic Helmlholtz equation.

## Thermoacoustic Helmholtz equation

The thermoacoustic Helmholtz equation reads;

$$ \nabla\cdot\left( c^2 \nabla  \hat{p}_1 \right) + \omega^2\hat{p}_1  = -i\omega (\gamma-1)\hat{q}_1  $$

In matrix form;

$$ \textbf{A}\textbf{P} + \omega \textbf{B}\textbf{P} + \omega^2 \textbf{C} \textbf{P} = \textbf{D}(\omega)\textbf{P} $$

where 

$\textbf{P}$ is the eigenvector or eigenfunction, $\omega$ is the eigenvalue ( $f$ = $\omega$ / $2\pi$ is eigenfrequency) and the discretized matrices are;

$$ \mathbf{A_{jk}} = -\int_\Omega c^2\nabla \phi_j \cdot\nabla \phi_k dx $$

$$ \mathbf{B_{jk}} = \int_{\partial \Omega} \left( \frac{  ic}{Z}\right)  \phi_j\phi_k d{\sigma}   $$

$$ \mathbf{C_{jk}} = \int_\Omega\phi_j\phi_k\ dx   $$

$$ \mathbf{D_{jk}} = (\gamma-1) \text{|FTF|}\frac{ q_{tot}  }{ U_{bulk}} \int_{\Omega} \phi_i h(\textbf{x}) e^{i \omega \tau(\textbf{x})} d\textbf{x}  \int_{\Omega} \frac{w(\chi)}{\rho_0 (\chi)}  \nabla{\phi_j} \cdot \textbf{n}_{ref} d\chi $$

This library has the following capabilities;

- Distributed and pointwise heat release rate fields are available.
- Distributed and constant time delay fields are available.
- The Bloch boundary condition for cheap calculation of azimuthal eigenmodes is available.
- Advanced operations for mesh generation is available in standalone [gmsh-x](https://github.com/ekremekc/gmsh-x) library.

## Installation

We advise to install those packages from source in Ubuntu/Linux OS, but there is a very simple option to run **helmholtz-x**: *Docker*.  

### Docker images

You can get packaged installation of dependencies counted above using Docker images. To install the docker into your system, you can find the instructions in this [link](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). 

The **helmholtz-x** runs with the v0.7.3 of [DOLFINx](https://github.com/FEniCS/dolfinx). And easiest way of getting the latest DOLFINx is docker containers;

First, you need to clone the **helmholtz-x** repository by typing;

```
git clone https://github.com/ekremekc/helmholtz-x.git
```
then `cd` into the **helmholtz-x** with

```
cd helmholtz-x
```
Now, you need to pull the docker container for DOLFINx. You can make a docker environment for **helmholtz-x** by typing;

```
sudo docker run -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/root/shared -w /root/shared --name=helmholtz-x dolfinx/dolfinx:v0.7.3
```
It might take some time (5-10 min) depending on your system and internet connection. Then you will be in the new terminal within the docker container. At the present working directory, you should run;

```
pip3 install -e .
```
in order to install **helmholtz-x** within the docker container. Then, lastly you need to activate the complex number mode of DOLFINx, as the **helmholtz-x** utilizes complex builds of DOLFINx/PETSc/SLEPc. It can be activated running;

```shell
source /usr/local/bin/dolfinx-complex-mode
```

Now, you should be able to run the demos in the *numerical_example* directory.

When you exit the terminal by typing `Ctrl+D`, you can always login back into the docker container for helmholtz-x by typing;

```
sudo docker start -i helmholtz-x
```
then you are again in the **helmholtz-x** directory in the fresh terminal. It is important to note that, you always have to run

```shell
source /usr/local/bin/dolfinx-complex-mode
```
to activate the complex build of DOLFINx, upon your every new login to the docker container.

### Conda

It is advised to use **helmholtz-x** using *docker* images. But, users may install the dependencies with the conda, which generally takes much longer time to install, compared to the installation of *docker*. Here is the livestock for [PETSc/SLEPc/FEniCSx](https://fenicsproject.discourse.group/t/error-when-trying-to-solve-complex-eigenvalue-problem-in-parallel/13546/3);

```shell
conda create -n helmholtzx-env
conda activate helmholtzx-env
conda install -c conda-forge python=3.10 mpich fenics-dolfinx hdf5=1.14.2 petsc=*=complex* slepc=*=complex*
```
Then **helmholtz-x** can be installed within this conda environment by typing.

```
pip3 install -e .
```
in the **helmholtz-x** directory.
