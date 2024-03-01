from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.io_utils import XDMFReader, xdmf_writer, dict_writer
import params
import datetime
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
FTF = state_space(params.S1, params.s2, params.s3, params.s4)
D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree)

## Solving direct problem

D.assemble_submatrices('direct')

# Introduce direct solver object and start
target_dir = PETSc.ScalarType(+3225.120  +481.0j)
E = fixed_point_iteration(matrices, D, target_dir, i=0, nev=4, tol=1e-3)

# Extract direct eigenvalues and normalized eigenvectors 
omega_1_dir, p_1_dir = normalize_eigenvector(mesh, E, i=0, degree=degree)
omega_2_dir, p_2_dir = normalize_eigenvector(mesh, E, i=1, degree=degree)
u_1_dir = velocity_eigenvector(mesh, p_1_dir, omega_1_dir, params.rho_amb)

# Save eigenvectors
xdmf_writer("Results/Active/FPI/p_1_dir", mesh, p_1_dir)
xdmf_writer("Results/Active/FPI/p_2_dir", mesh, p_2_dir)
xdmf_writer("Results/Active/FPI/u_1_dir", mesh, u_1_dir)

# Save eigenvalues
omega_dir_dict = {'direct_1':omega_1_dir, 'direct_2':omega_2_dir}      
dict_writer("Results/Active/FPI/eigenvalues_dir", omega_dir_dict)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time for Direct Modes: ", datetime.datetime.now()-start_time, "\n")

## Solving adjoint problem

D.assemble_submatrices('adjoint')

# Introduce adjoint solver object and start
target_adj = PETSc.ScalarType(+3225.120  -481.0j)
E_adj = fixed_point_iteration(matrices, D, target_adj, i=0, tol=1e-3, problem_type='adjoint',print_results=False)

# Extract adjoint eigenvalues and normalized eigenvectors 
omega_1_adj, p_1_adj = normalize_eigenvector(mesh, E_adj, i=0, degree=degree)
omega_2_adj, p_2_adj = normalize_eigenvector(mesh, E_adj, i=1, degree=degree)

# Save eigenvectors
xdmf_writer("Results/Active/FPI/p_1_adj", mesh, p_1_adj)
xdmf_writer("Results/Active/FPI/p_2_adj", mesh, p_2_adj)

# Save eigenvalues
omega_adj_dict = {'adjoint_1':omega_1_adj, 'adjoint_2':omega_2_adj}     
dict_writer("Results/Active/FPI/eigenvalues_adj", omega_adj_dict)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)