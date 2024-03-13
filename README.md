# helmholtz-x

**helmholtz-x** is a python library which solves a non-homogeneous Helmholtz equation using finite element method (FEM).

In **helmholtz-x**, the nonlinear eigenvalue problem is determined using [PETSc](https://petsc.org/release/overview/), [SLEPc](https://slepc.upv.es/) and [FEniCSx](https://github.com/FEniCS) libraries. 

We specifically address thermoacoustic Helmholtz equation within **helmholtz-x**. In its submodules, **helmholtz-x** exploits extensive parallelization with handled preallocations for the generation of nonlinear part of the thermoacoustic Helmlholtz equation. The thermoacoustic Helmholtz equation reads;

$$ \nabla\cdot\left( c^2 \nabla  \hat{p}_1 \right) + \omega^2\hat{p}_1  = i\omega (\gamma-1)\hat{q}_1  $$

In matrix form;

$$ \textbf{A}\textbf{P} + \omega \textbf{B}\textbf{P} + \omega^2 \textbf{C} \textbf{P} = \textbf{D}(\omega)\textbf{P} $$

and we solve this matrix system with **helmholtz-x**.

## Citation

We are currently writing a journal paper to describe internal mechanics of **helmholtz-x**, and the .bibtex will be available here after publication.

## Installation

We advise to install those packages from source in Ubuntu/Linux OS, but there is a very simple option to run **helmholtz-x**: *Docker*.  

### Docker images

You can get packaged installation of dependencies such as [PETSc](https://petsc.org/release/overview/), [SLEPc](https://slepc.upv.es/) and [FEniCSx](https://github.com/FEniCS) using Docker images. To install the docker into your system, you can find the instructions in this [link](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). 

The **helmholtz-x** runs with the v0.7.3 of [DOLFINx](https://github.com/FEniCS/dolfinx). First, you need to clone the **helmholtz-x** repository by typing;

```
git clone https://github.com/ekremekc/helmholtz-x.git
```
in your Linux/Ubuntu terminal then `cd` into the **helmholtz-x** with

```
cd helmholtz-x
```
Now, you need to pull the docker container for DOLFINx. You can make a docker environment for **helmholtz-x** by typing;

```
sudo docker run -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/root/shared -w /root/shared --name=helmholtz-x dolfinx/dolfinx:v0.7.3
```
Pulling the image might take some time (5-10 min) depending on your system and internet connection. Then you will be in the new terminal within the docker container. At the present working directory (helmholtz-x), you should run;

```
pip3 install -e .
```
to install **helmholtz-x** within the docker container environment. Then, lastly you need to activate the complex number mode of DOLFINx, as the **helmholtz-x** utilizes complex builds of DOLFINx/PETSc/SLEPc. It can be activated running;

```shell
source /usr/local/bin/dolfinx-complex-mode
```

Now, you should be able to run the demos in the *numerical_example* directory.

When you exit the terminal by typing `Ctrl+D`, you can always log back into the docker container for **helmholtz-x** by typing;

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

## How to use helmholtz-x?

You just need to navigate the example case you like in */numerical_examples* folder and type 

```
python3 -u file.py
```
in the terminal. Each example case includes `runAll.sh`, which includes the example commands for running files in serial as well as in parallel. It can be executed with;
```
source runAll.sh
```
to perform testing for serial and parallel runs and writes the log file in the */Results* directory.

The flowchart and duties of the components within **helmholtz-x** are visualized below;

![alt text](https://github.com/ekremekc/helmholtz-x/blob/main/docs/flowchart.svg?raw=true)

