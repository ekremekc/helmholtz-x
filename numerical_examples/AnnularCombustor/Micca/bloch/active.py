from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.bloch_operator import Blochifier
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.io_utils import XDMFReader, xdmf_writer
from petsc4py import PETSc
import numpy as np
import datetime
import params

start_time = datetime.datetime.now()

# approximation space polynomial degree
degree = 1

# Read mesh 
micca = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = micca.getAll()
micca.getInfo()

# Define the boundary conditions
boundary_conditions = {1: 'Neumann',
                       2: 'Neumann',
                       3: 'Neumann',
                       4: 'Neumann',
                       5: 'Neumann',
                       6: 'Neumann',
                       7: 'Neumann',
                       8: 'Neumann',
                       9: 'Neumann',
                       10: 'Neumann',
                       11: 'Dirichlet',
                       12: 'Master',
                       13: 'Slave'}

# Introduce Passive Flame Matrices
c = params.c(mesh)
matrices =  AcousticMatrices(mesh, facet_tags, boundary_conditions,c, degree=degree)

# set the bloch elements
N = 16 # Bloch number
bloch_matrices = Blochifier(geometry=micca, boundary_conditions=boundary_conditions, N=N, passive_matrices=matrices)

# Introduce Flame Matrix parameters
FTF = state_space(params.S1, params.s2, params.s3, params.s4)
D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree, bloch_object=bloch_matrices)
D.assemble_submatrices('direct')
D.blochify()

# Introduce solver object and start
target_dir = PETSc.ScalarType(3200+500j) #1st mode 
# target_dir = PETSc.ScalarType(4479+140j) #2nd mode 
E = fixed_point_iteration(bloch_matrices, D, target_dir, nev=3, i=0, tol=1e-3)

# Extract eigenvalue and normalized eigenvector 
omega_1_dir, p_1_dir = normalize_eigenvector(mesh, E, i=0, degree=degree, BlochRemapper=bloch_matrices.remapper)

# Save eigenvectors
xdmf_writer("Results/Active/p_1_dir", mesh, p_1_dir)

# ADJOINT PROBLEM

# D.assemble_submatrices('adjoint')
# D.blochify('adjoint')

# target_adj = PETSc.ScalarType(3200-500j)
# E_adj = fixed_point_iteration_eps(bloch_matrices, D, target_adj**2, problem_type='adjoint', nev=3, i=0, tol=1e-3)

# omega_1_adj, p_1_adj = normalize_eigenvector(mesh, E_adj, i=0, degree=degree, mpc=bloch_matrices.remapper)
# omega_2_adj, p_2_adj = normalize_eigenvector(mesh, E_adj, i=1, degree=degree, mpc=bloch_matrices.remapper)
# omega_3_adj, p_3_adj = normalize_eigenvector(mesh, E_adj, i=2, degree=degree, mpc=bloch_matrices.remapper)


# print(f"Eigenvalues 1 -> {omega_1_adj:.3f} | Eigenfrequencies ->  {omega_1_adj/(2*np.pi):.3f}")
# print(f"Eigenvalues 2 -> {omega_2_adj:.3f} | Eigenfrequencies ->  {omega_2_adj/(2*np.pi):.3f}")
# print(f"Eigenvalues 3 -> {omega_3_adj:.3f} | Eigenfrequencies ->  {omega_3_adj/(2*np.pi):.3f}")

# # Save eigenvectors

# xdmf_writer("Results/p_1_adj", mesh, p_1_adj)
# xdmf_writer("Results/p_2_adj", mesh, p_2_adj)

print("Total Execution Time: ", datetime.datetime.now()-start_time)