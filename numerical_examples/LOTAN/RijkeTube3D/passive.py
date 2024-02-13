from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, XDMFReader
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import c_step
start = start_time()
import numpy as np
import params

# approximation space polynomial degree
degree = 1

# number of elements in each direction of mesh
rijke3d = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = rijke3d.getAll()
rijke3d.getInfo()

# Define the boundary conditions
boundary_conditions = {1: {'Neumann'},
                       2: {'Neumann'},
                       3: {'Neumann'}}

# Define Speed of sound
c = c_step(mesh, params.x_f, params.c_u, params.c_u)

# Introduce Passive Flame Matrices
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=degree)
matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# Introduce solver object and start
target = 200 * 2 * np.pi 
E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= True)

# Extract eigenvalue and normalized eigenvector 
omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector
xdmf_writer("Results/Passive/p", mesh, p)

execution_time(start)