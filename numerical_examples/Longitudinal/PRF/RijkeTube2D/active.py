from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, c_step
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.io_utils import xdmf_writer, XDMFReader
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params
import sys

start = start_time()

# approximation space polynomial degree
degree = 1

rijke2d = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = rijke2d.getAll()
rijke2d.getInfo()

# Define the boundary conditions
boundary_conditions = {4: {'Robin': params.R_out}, # outlet
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Robin': params.R_in}} # inlet

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
omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector
xdmf_writer("Results/Active/p", mesh, uh)

# Introduce adjoints
D.assemble_submatrices('adjoint')

# Introduce solver object and start
target = np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, problem_type='adjoint', print_results= False)

# Extract eigenvalue and normalized eigenvector 
omega_adj, p_adjoint = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector
xdmf_writer("Results/Active/p_adj", mesh, p_adjoint)

execution_time(start)