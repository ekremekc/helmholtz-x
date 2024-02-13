from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import fixed_point_iteration_eps
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer

import datetime
start_time = datetime.datetime.now()

import params

# Read mesh 
Micca = XDMFReader("MeshDir/Micca")
mesh, subdomains, facet_tags = Micca.getAll()
Micca.getInfo()

# FTF = n_tau(params.N3, params.tau)
FTF = state_space(params.S1, params.s2, params.s3, params.s4)

# EVERYWHERE İS NEUMANN EXCEPT OUTLET(COMBUSTION CHAMBER OUTLET)
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
                       11: 'Dirichlet'}

degree = 1

target_dir = PETSc.ScalarType(3281+540.7j)
target_adj = PETSc.ScalarType(3281-540.7j)
c = params.c(mesh)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)
matrices.assemble_A()
matrices.assemble_C()

D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree)

D.assemble_submatrices('direct')

E = fixed_point_iteration_eps(matrices, D, target_dir**2, i=0, tol=1e-3)

omega_1, p_1 = normalize_eigenvector(mesh, E, i=0, degree=degree)
omega_2, p_2 = normalize_eigenvector(mesh, E, i=1, degree=degree)

if MPI.COMM_WORLD.Get_rank()==0:
    print("Direct Eigenvalues -> ", omega_1," =? ", omega_2)

# Save eigenvectors
xdmf_writer("Results/p_1", mesh, p_1)
xdmf_writer("Results/p_2", mesh, p_2)

if MPI.COMM_WORLD.Get_rank()==0:
    print("Total Execution Time for Direct Modes: ", datetime.datetime.now()-start_time)