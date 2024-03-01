from helmholtz_x.eigensolvers import eps_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.bloch_operator import Blochifier
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
                       11: 'Neumann',
                       12: 'Master',
                       13: 'Slave'}

# Introduce Passive Flame Matrices
c = params.c(mesh)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=degree)

# set the bloch elements
N = 16
bloch_matrices = Blochifier(geometry=micca, boundary_conditions=boundary_conditions, N=N, passive_matrices=matrices)

# Introduce solver object and start
target_dir = PETSc.ScalarType(3000)
eigensolver = eps_solver(bloch_matrices.A, bloch_matrices.C, target_dir, nev = 5, print_results=False)

# Extract eigenvalue and normalized eigenvector 
BN_petsc = bloch_matrices.remapper
omega_1, p_1 = normalize_eigenvector(mesh, eigensolver, 0, BlochRemapper=BN_petsc)
omega_2, p_2 = normalize_eigenvector(mesh, eigensolver, 1, BlochRemapper=BN_petsc)
omega_3, p_3 = normalize_eigenvector(mesh, eigensolver, 2, BlochRemapper=BN_petsc)

# Save eigenvectors
xdmf_writer("Results/Passive/p_1", mesh, p_1)
xdmf_writer("Results/Passive/p_2", mesh, p_2)
xdmf_writer("Results/Passive/p_3", mesh, p_3)

print("Total Execution Time: ", datetime.datetime.now()-start_time)