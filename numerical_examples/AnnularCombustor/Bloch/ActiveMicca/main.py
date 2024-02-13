from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import fixed_point_iteration_eps
from helmholtz_x.bloch_operator import Blochifier
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer
import numpy as np
import datetime
start_time = datetime.datetime.now()

import params

# Read mesh 
micca = XDMFReader("MeshDir/micca")
mesh, subdomains, facet_tags = micca.getAll()
micca.getInfo()

# FTF = n_tau(params.N3, params.tau)
FTF = state_space(params.S1, params.s2, params.s3, params.s4)


# ________________________________________________________________________________
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
                       11: 'Dirichlet', # Dirichlet
                       12: 'Master',
                       13: 'Slave'}

degree = 1

target_dir = PETSc.ScalarType(3200+500j) #2nd mode 4479+14j
target_adj = PETSc.ScalarType(3200-500j)
c = params.c(mesh)

b = 1
BlochNumber = 16 # Bloch number

matrices =  PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions,c, degree=degree)
matrices.assemble_A()
matrices.assemble_C()
bloch_matrices = Blochifier(geometry=micca, boundary_conditions=boundary_conditions, BlochNumber=BlochNumber, passive_matrices=matrices)
bloch_matrices.blochify_A()
bloch_matrices.blochify_C()

D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree, bloch_object=bloch_matrices)
D.assemble_submatrices('direct')
D.blochify()

E = fixed_point_iteration_eps(bloch_matrices, D, target_dir**2, nev=3, i=0, tol=1e-3)

omega_1, p_1 = normalize_eigenvector(mesh, E, i=0, degree=degree,mpc=bloch_matrices.remapper)
omega_2, p_2 = normalize_eigenvector(mesh, E, i=1, degree=degree,mpc=bloch_matrices.remapper)
omega_3, p_3 = normalize_eigenvector(mesh, E, i=2, degree=degree,mpc=bloch_matrices.remapper)


print(f"Eigenvalues 1 -> {omega_1:.3f} | Eigenfrequencies ->  {omega_1/(2*np.pi):.3f}")
print(f"Eigenvalues 2 -> {omega_2:.3f} | Eigenfrequencies ->  {omega_2/(2*np.pi):.3f}")
print(f"Eigenvalues 3 -> {omega_3:.3f} | Eigenfrequencies ->  {omega_3/(2*np.pi):.3f}")

# Save eigenvectors

p_1.name = "P_1_Direct"
p_2.name = "P_2_Direct"

xdmf_writer("Results/p_1", mesh, p_1)
xdmf_writer("Results/p_2", mesh, p_2)
# ________________________________________________________________________________
# ADJOINT PROBLEM


D.assemble_submatrices('adjoint')
D.blochify('adjoint')

E_adj = fixed_point_iteration_eps(bloch_matrices, D, target_adj**2, problem_type='adjoint', nev=3, i=0, tol=1e-3)

omega_1_adj, p_1_adj = normalize_eigenvector(mesh, E_adj, i=0, degree=degree, mpc=bloch_matrices.remapper)
omega_2_adj, p_2_adj = normalize_eigenvector(mesh, E_adj, i=1, degree=degree, mpc=bloch_matrices.remapper)
omega_3_adj, p_3_adj = normalize_eigenvector(mesh, E_adj, i=2, degree=degree, mpc=bloch_matrices.remapper)


print(f"Eigenvalues 1 -> {omega_1_adj:.3f} | Eigenfrequencies ->  {omega_1_adj/(2*np.pi):.3f}")
print(f"Eigenvalues 2 -> {omega_2_adj:.3f} | Eigenfrequencies ->  {omega_2_adj/(2*np.pi):.3f}")
print(f"Eigenvalues 3 -> {omega_3_adj:.3f} | Eigenfrequencies ->  {omega_3_adj/(2*np.pi):.3f}")

# Save eigenvectors

p_1_adj.name = "P_1_Adjoint"
p_2_adj.name = "P_2_Adjoint"

xdmf_writer("Results/p_1_adj", mesh, p_1_adj)
xdmf_writer("Results/p_2_adj", mesh, p_2_adj)

print("Total Execution Time: ", datetime.datetime.now()-start_time)
