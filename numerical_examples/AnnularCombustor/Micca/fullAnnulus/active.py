from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import fixed_point_iteration_pep
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
matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# Introduce Flame Matrix parameters
FTF = state_space(params.S1, params.s2, params.s3, params.s4)
D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree)
D.assemble_submatrices('direct')

# Introduce solver object and start
target_dir = PETSc.ScalarType(+3225.120  +481.0j)
E = fixed_point_iteration_pep(matrices, D, target_dir, i=0, nev=4, tol=1e-3)

# Extract eigenvalue and normalized eigenvector 
omega_1, p_1 = normalize_eigenvector(mesh, E, i=0, degree=degree)
omega_2, p_2 = normalize_eigenvector(mesh, E, i=1, degree=degree)
u_1 = velocity_eigenvector(mesh, p_1, omega_1, params.rho_amb)

# Save eigenvectors
xdmf_writer("Results/Active/p_1", mesh, p_1)
xdmf_writer("Results/Active/p_2", mesh, p_2)
xdmf_writer("Results/Active/u_1", mesh, u_1)

# Save eigenvalues
omega_dict = {'direct_1':omega_1, 'direct_2':omega_2}      
dict_writer("Results/Active/eigenvalues", omega_dict)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time for Direct Modes: ", datetime.datetime.now()-start_time)

# ________________________________________________________________________________

# D.assemble_submatrices('adjoint')

# E_adj = fixed_point_iteration_eps(matrices, D, target_adj**2, i=0, tol=1e-3, problem_type='adjoint',print_results=False)

# omega_adj_1, p_adj_1 = normalize_eigenvector(mesh, E_adj, i=0, degree=degree)
# omega_adj_2, p_adj_2 = normalize_eigenvector(mesh, E_adj, i=1, degree=degree)

# if MPI.COMM_WORLD.rank == 0:
#     print("Adjoint Eigenvalues -> ", omega_adj_1," =? ", omega_adj_2)

# p_adj_norm_1 = normalize_adjoint(omega_1, p_1, p_adj_1, matrices, D)
# p_adj_norm_2 = normalize_adjoint(omega_2, p_2, p_adj_2, matrices, D)

# # Save eigenvectors

# xdmf_writer("Results/"+case_name+"/p_adj_1", mesh, p_adj_1)
# xdmf_writer("Results/"+case_name+"/p_adj_2", mesh, p_adj_2)

# if MPI.COMM_WORLD.rank == 0:
#     print("Total Execution Time: ", datetime.datetime.now()-start_time)