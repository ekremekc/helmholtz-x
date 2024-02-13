from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
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
rijke3d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke3d.getAll()
rijke3d.getInfo()

# Define the boundary conditions
boundary_conditions = {3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Define Speed of sound

c = c_step(mesh, params.x_f, params.c_u, params.c_u)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target = 200 * 2 * np.pi 
E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= True)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p_passive", mesh, uh)

execution_time(start)