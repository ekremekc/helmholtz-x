from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.io_utils import xdmf_writer, XDMFReader
from helmholtz_x.parameters_utils import temperature_step, rho_step, gaussianFunction
from helmholtz_x.solver_utils import start_time, execution_time
start = start_time()
import numpy as np
import  params

# approximation space polynomial degree
degree = 1

rijke2d = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = rijke2d.getAll()
rijke2d.getInfo()

# Define the boundary conditions
boundary_conditions = {4: {'Neumann'},
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Introduce Passive Flame Matrices
T = temperature_step(mesh, params.x_f, params.T_u, params.T_d) 
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T, degree=degree)

rho = rho_step(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
FTF = nTau(params.n, params.tau)
D = DistributedFlameMatrix(mesh, w, h, rho, T, params.q_0, params.u_b, FTF, degree=degree)
D.assemble_submatrices()

# Introduce solver object and start
target = 200 * 2 * np.pi # 150 * 2 * np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False)

# Extract eigenvalue and normalized eigenvector 
omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector
xdmf_writer("Results/Active/p", mesh, uh)

execution_time(start)