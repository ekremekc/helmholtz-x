from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import PointwiseFlameMatrix
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import newton_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
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
D = PointwiseFlameMatrix(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree)
D.assemble_submatrices('direct')

# Introduce newton solver object and extract eigenvalues and normalized eigenvectors 
target_dir = PETSc.ScalarType(+3250 + 500j)
omega_1, p_1 = newton_solver(matrices, D, target_dir, i=0, nev=4, tol=1e-3)
omega_2, p_2 = newton_solver(matrices, D, target_dir, i=1, nev=4, tol=1e-3)

# Save eigenvectors
xdmf_writer("Results/Active/NewtonSolver/p_1", mesh, p_1)
xdmf_writer("Results/Active/NewtonSolver/p_2", mesh, p_2)

# Save eigenvalues
omega_dict = {'direct_1':omega_1, 'direct_2':omega_2}      
dict_writer("Results/Active/NewtonSolver/eigenvalues", omega_dict)

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time for Direct Modes: ", datetime.datetime.now()-start_time)