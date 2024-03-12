from helmholtz_x.dolfinx_utils import OneDimensionalSetup
from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, rho_step
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.io_utils import xdmf_writer
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params as params
import sys

start = start_time()

# approximation space polynomial degree
degree = 1

# number of elements in 1D mesh
n_elem = 3000
mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions
boundary_conditions = {1: {'Neumann'},
                       2: {'Neumann'}}

# Introduce Passive Flame Matrices
T = temperature_step(mesh, params.x_f, params.T_u, params.T_d)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T, degree=degree)

# Introduce Flame Matrix parameters
FTF = nTau(params.n, params.tau)
rho = rho_step(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
D = DistributedFlameMatrix(mesh, w, h, rho, T, params.q_0, params.u_b, FTF, degree=degree)
D.assemble_submatrices()

# Introduce solver object and start
target = 200 * 2 * np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False)

# Extract eigenvalue and normalized eigenvector 
omega, p_active = normalize_eigenvector(mesh, E, i=0, degree=degree, which='right')
v = velocity_eigenvector(mesh, p_active, omega, rho, degree=degree)

# Save Eigenvector
xdmf_writer("Results/Active/p", mesh, p_active)

# We plot eigenvectors when running in serial
from mpi4py import MPI
size = MPI.COMM_WORLD.Get_size()
if size ==1 and degree==1:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, figsize=(6, 8))
    x_coords = mesh.geometry.x[:,0]
    ax[0].plot(x_coords, p_active.x.array.real)
    ax[0].set_ylabel(r"$real(\hat{p}_1)$")
    ax[1].plot(x_coords, p_active.x.array.imag)
    ax[1].set_ylabel(r"$imag(\hat{p}_1)$")

    ax[2].plot(x_coords, v.x.array.real)
    ax[2].set_ylabel(r"$real(\hat{u}_1)$")
    ax[3].plot(x_coords, v.x.array.imag)
    ax[3].set_ylabel(r"$imag(\hat{u}_1)$")
    ax[3].set_xlabel(r"$x$")

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    fig.tight_layout()

    plt.savefig("Results/Active/"+"Active"+".pdf")

    if '-nopopup' not in sys.argv:
        plt.show()

execution_time(start)