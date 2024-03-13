from mpi4py import MPI
from helmholtz_x.flame_matrices import PointwiseFlameMatrix
from helmholtz_x.flame_transfer_function import stateSpace
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.io_utils import XDMFReader, xdmf_writer, dict_writer
from helmholtz_x.parameters_utils import Q_multiple
from helmholtz_x.dolfinx_utils import absolute
from petsc4py import PETSc
import params
from math import pi
import datetime
import sys
start_time = datetime.datetime.now()

# approximation space polynomial degree
degree = 1

# Read mesh 
Micca = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = Micca.getAll()
Micca.getInfo()

# Define the boundary conditions
boundary_conditions = {11: {'Robin':params.R_outlet}}

# Introduce Passive Flame Matrices
c = params.c(mesh)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=degree)

# Introduce Flame Matrix parameters
FTF = stateSpace(params.S1, params.s2, params.s3, params.s4)
h = Q_multiple(mesh, subdomains, params.N_sector)
D = PointwiseFlameMatrix(mesh, subdomains, params.x_r, h, params.rho_xr, params.q_0, params.u_b, FTF, degree=degree)

## Solving direct problem

D.assemble_submatrices('direct')

if '-target' not in sys.argv:
    target = PETSc.ScalarType(+1000)
else:
    ind = sys.argv.index('-target')
    target = PETSc.ScalarType(sys.argv[ind+1])

# Introduce direct solver object and start
E = fixed_point_iteration(matrices, D, target, i=0, nev=4, tol=1e-3)

# Extract direct eigenvalues and normalized eigenvectors 
omega, p = normalize_eigenvector(mesh, E, i=0, degree=degree)

# Calculate absolute eigenfunctions
p = absolute(p)

# Save eigenvectors
xdmf_writer("Results/Active/Modes/p_abs_"+str(int(omega.real/2/pi))+"Hz", mesh, p)

# Save eigenvalues
omega_dict = {'direct':omega/2/pi}      
dict_writer("Results/Active/Modes/eigenvalues_"+str(int(omega.real/2/pi))+"Hz", omega_dict)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time for Direct Modes: ", datetime.datetime.now()-start_time, "\n")