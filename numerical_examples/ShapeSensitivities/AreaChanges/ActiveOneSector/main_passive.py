from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector, velocity_eigenvector_holes
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer, ParallelMeshVisualizer
from helmholtz_x.parameters_utils import rho_ideal, temperature_uniform
from helmholtz_x.solver_utils import start_time, execution_time
from mpi4py import MPI
start = start_time()
import params
import numpy as np

# Read mesh 
LPP = XDMFReader("MeshDir/thinAnnulus")
mesh, subdomains, facet_tags = LPP.getAll()
LPP.getInfo()

ParallelMeshVisualizer("MeshDir/thinAnnulus")
boundary_conditions = params.boundary_conditions

degree = 1

# T = params.temperature(mesh, params.T_ambient, params.T_ambient, params.T_ambient)
T = temperature_uniform(mesh, params.T_ambient)
rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)
matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, holes=params.holes)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target_dir = 380*2*np.pi # 390.62*2*np.pi

E = pep_solver(matrices.A, matrices.B, matrices.C, target_dir, nev=2, print_results=False)

omega2, p_damped2 = normalize_eigenvector(mesh, E, 0, absolute=True, degree=degree, which='right')
xdmf_writer("Results/p_damped_axial", mesh, p_damped2)

omega_u, p_u = normalize_eigenvector(mesh, E, 0, absolute=False, degree=degree, which='right')

u = velocity_eigenvector_holes(mesh, subdomains, p_u, omega_u, rho, params.holes, absolute=True)
xdmf_writer("Results/u", mesh, u)

execution_time(start)
