from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.io_utils import XDMFReader, xdmf_writer
from helmholtz_x.dolfinx_utils import absolute, phase
from helmholtz_x.parameters_utils import rho_ideal, gaussianFunction, temperature_step, halfGaussianFunction
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params

start = start_time()

# approximation space polynomial degree
degree = 1

# Read mesh 
tube = XDMFReader("MeshDir/mesh")
mesh, subdomains, facet_tags = tube.getAll()
tube.getInfo()

# Define the boundary conditions
boundary_conditions = {3:{"ChokedInlet":params.M0},
                       8:{"ChokedOutlet":params.M1}}

# Introduce Passive Flame Matrices
T = temperature_step(mesh, params.x_flame, params.T_passive, params.T_flame)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T, degree=degree)

# Introduce Flame Matrix parameters
rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)
w = gaussianFunction(mesh, params.x_ref, params.a_ref)
h= halfGaussianFunction(mesh, params.x_flame, params.a_flame)
D = ActiveFlameNT(mesh, w, h, rho, T, params.eta, params.tau, degree=1)

# Introduce solver object and start
target_dir = 250 * 2 * np.pi
E = fixed_point_iteration(matrices, D, target_dir, nev=2)

# Extract eigenvalue and normalized eigenvector 
omega, p = normalize_eigenvector(mesh, E, 0, absolute=False, degree=degree, which='right')
u = velocity_eigenvector(mesh, p, omega, rho, absolute=True)
p_abs = absolute(p)
p_phase = phase(p, deg=True)

# Save Eigenvector
xdmf_writer("Results/Active/p", mesh, p)
xdmf_writer("Results/Active/p_abs", mesh, p_abs)
xdmf_writer("Results/Active/p_phase", mesh, p_phase)
xdmf_writer("Results/Active/u", mesh, u)

execution_time(start)