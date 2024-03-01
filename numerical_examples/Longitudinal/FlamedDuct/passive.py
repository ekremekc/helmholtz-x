from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.io_utils import XDMFReader, xdmf_writer
from helmholtz_x.parameters_utils import temperature_uniform
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params

start = start_time()

# approximation space polynomial degree
degree = 1

# Read mesh 
tube = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = tube.getAll()
tube.getInfo()

# Define the boundary conditions
boundary_conditions = {3:{"ChokedInlet":params.M0},
                       8:{"ChokedOutlet":params.M1}}

# Introduce Passive Flame Matrices
T = temperature_uniform(mesh, params.T_passive)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T, degree=degree)

# Introduce solver object and start
target_dir = 262 * 2 * np.pi
E = pep_solver(matrices.A, matrices.B, matrices.C, target_dir, nev=10, print_results=True)

# Extract eigenvalue and normalized eigenvector 
omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')
xdmf_writer("Results/Passive/p", mesh, p)

execution_time(start)