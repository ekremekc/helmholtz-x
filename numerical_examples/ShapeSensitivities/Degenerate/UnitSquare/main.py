import numpy as np
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
from helmholtz_x.eigenvectors import normalize_eigenvector, normalize_adjoint
from helmholtz_x.eigensolvers import pep_solver,eps_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.dolfinx_utils import XDMFReader

# Read mesh 
geometry = XDMFReader("MeshDir/square")
mesh, cell_tags, facet_tags = geometry.getAll()
# Define the boundary conditions

boundary_conditions = {1: 'Neumann',
                       2: 'Neumann',
                       3: 'Neumann',
                       4: 'Neumann'}

# Define Speed of sound
c = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))

deg = 2

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree =deg)

matrices.assemble_A()
matrices.assemble_C()

A = matrices.A
C = matrices.C

target = np.pi**2

eigensolver = eps_solver(A, C, target, 3, two_sided=True,print_results=False)

omega1, p_dir1 = normalize_eigenvector(mesh, eigensolver, 0,degree=deg)
omega2, p_dir2 = normalize_eigenvector(mesh, eigensolver, 1,degree=deg)

print("omega1", omega1,"omega2", omega2)

omega1adj, p1adj = normalize_eigenvector(mesh, eigensolver, 0,which='left',degree=deg)
omega2adj, p2adj = normalize_eigenvector(mesh, eigensolver, 1,which='left',degree=deg)

from helmholtz_x.shape_derivatives_x import ShapeDerivativesDegenerate

results = ShapeDerivativesDegenerate(geometry, boundary_conditions, omega1, 
                               p_dir1, p_dir2, p1adj, p2adj, c)

print(results[1])

import dolfinx.io
p_dir1.name = "Direct1"
p_dir2.name = "Direct2"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "eigenvectors.xdmf", "w") as xdmf:
    # use extract block in paraview to visualize
    xdmf.write_mesh(mesh)
    xdmf.write_function(p_dir1)
    xdmf.write_function(p_dir2)
