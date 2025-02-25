from helmholtz_x.dolfinx_utils import OneDimensionalSetup
from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, c_step
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.io_utils import xdmf_writer
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params
import sys

start = start_time()

# approximation space polynomial degree
degree = 1

# number of elements in 1D mesh
n_elem = 300
mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions
boundary_conditions = {1: {'Robin': params.R_in},  # inlet
                       2: {'Robin': params.R_out}} # outlet

# Introduce Passive Flame Matrices
c = c_step(mesh, params.x_f, params.c_u, params.c_d)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=degree)

# Introduce Flame Matrix parameters
FTF = nTau(params.n, params.tau)
rho = params.rho_func(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
T = temperature_step(mesh, params.x_f, params.T_u, params.T_d)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
D = DistributedFlameMatrix(mesh, w, h, rho, T, params.q_0, params.u_b, FTF, degree=degree, gamma=params.gamma)
D.assemble_submatrices()

# Introduce solver object and start
target = np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False)

# Extract eigenvalue and normalized eigenvector 
omega, p_active = normalize_eigenvector(mesh, E, i=0, degree=degree, which='right')
v = velocity_eigenvector(mesh, p_active, omega, rho, degree=degree)

# Save Eigenvector
xdmf_writer("Results/Active/p_dir", mesh, p_active)

# Introduce adjoints
D.assemble_submatrices('adjoint')

# Introduce solver object and start
target = np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, problem_type='adjoint', print_results= False)

# Extract eigenvalue and normalized eigenvector 
omega_adj, p_adjoint = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector
xdmf_writer("Results/Active/p_adj", mesh, p_adjoint)

# We plot eigenvectors when running in serial
from mpi4py import MPI
size = MPI.COMM_WORLD.Get_size()
if size ==1 and degree==1:
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    direct_mat = loadmat('data/direct_data.mat')
    adjoint_mat = loadmat('data/adjoint_data.mat')

    x_matlab = direct_mat['emode_FEW_DA_nonlin'][0][0][1]
    p_direct_matlab = direct_mat['emode_FEW_DA_nonlin'][0][0][3]
    p_adjoint_matlab = adjoint_mat['emode_FEW_DA'][0][0][4]

    fig, ax = plt.subplots(4, figsize=(6, 8))
    x_coords = mesh.geometry.x[:,0]
    ax[0].plot(x_coords, p_active.x.array.real, label='helmholtz-x')
    ax[0].set_ylabel(r"$real(\hat{p}_1)$")
    ax[1].plot(x_coords, p_active.x.array.imag)
    ax[1].set_ylabel(r"$imag(\hat{p}_1)$")

    ax[2].plot(x_coords, p_adjoint.x.array.real)
    ax[2].set_ylabel(r"$real(\hat{p}_1^{\dagger})$")
    ax[3].plot(x_coords, p_adjoint.x.array.imag)
    ax[3].set_ylabel(r"$imag(\hat{p}_1^{\dagger})$")
    ax[3].set_xlabel(r"$x$")

    # Plot PRF data
    ax[0].plot(x_matlab, p_direct_matlab.real ,marker='s', markersize=2, label='[19]')
    ax[1].plot(x_matlab, -p_direct_matlab.imag ,marker='s', markersize=2)
    ax[2].plot(x_matlab, p_adjoint_matlab.real ,marker='s', markersize=2)
    ax[3].plot(x_matlab, -p_adjoint_matlab.imag ,marker='s', markersize=2)

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()

    ax[0].legend()
    fig.tight_layout()

    plt.savefig("Results/Active/"+"A3_PRF_modes"+".pdf")

    if '-nopopup' not in sys.argv:
        plt.show()

execution_time(start)