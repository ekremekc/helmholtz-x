from helmholtz_x.eigensolvers import eps_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import OneDimensionalSetup
from helmholtz_x.io_utils import xdmf_writer
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import c_step
import matplotlib.pyplot as plt
import params
import numpy as np
import sys

start = start_time()

# approximation space polynomial degree
degree = 1

# number of elements in 1D mesh
n_elem = 300
mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions
boundary_conditions = {1: {'Neumann'},
                       2: {'Neumann'}}

# Introduce Passive Flame Matrices
c = c_step(mesh, params.x_f, params.c_u, params.c_u)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=degree)

# Introduce solver object and start
target = 200 * 2 * np.pi
E = eps_solver(matrices.A, matrices.C, target, nev=2, print_results= True)

# Extract eigenvalue and normalized eigenvector 
omega, p_passive = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')
xdmf_writer("Results/Passive/p", mesh, p_passive)

# We plot eigenvectors when running in serial
from mpi4py import MPI
size = MPI.COMM_WORLD.Get_size()
if size ==1:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, figsize=(12, 6))
    x_coords = mesh.geometry.x[:,0]
    ax[0].plot(x_coords, p_passive.x.array.real)
    ax[1].plot(x_coords, p_passive.x.array.imag)
    plt.savefig("Results/Passive/"+"Passive"+".png")
    if '-nopopup' not in sys.argv:
        plt.show()

execution_time(start)