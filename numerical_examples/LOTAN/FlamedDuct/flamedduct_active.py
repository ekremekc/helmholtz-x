from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer
from helmholtz_x.parameters_utils import sound_speed_variable_gamma, rho_ideal, gaussianFunction, temperature_step, halfGaussianFunction
from helmholtz_x.solver_utils import start_time, execution_time
import params
import numpy as np

start = start_time()

# Read mesh 
tube = XDMFReader("MeshDir/FlamedDuct/tube")

mesh, subdomains, facet_tags = tube.getAll()
tube.getInfo()

M0 = 9.2224960671405849E-003
M1 = 1.1408306741423997E-002

boundary_conditions = {3:{"ChokedInlet":M0},
                       8:{"ChokedOutlet":M1}}
# boundary_conditions = {}

degree = 1

target_dir = 250 * 2 * np.pi
T = temperature_step(mesh, params.x_flame, params.T_mean, params.T_flame)
c = sound_speed_variable_gamma(mesh, T)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, degree=degree)
matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)
w = gaussianFunction(mesh, params.x_ref, params.a_ref)
h= halfGaussianFunction(mesh, params.x_flame, params.a_flame)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params.eta, params.tau, degree=1)

E = fixed_point_iteration_pep(matrices, D, target_dir, nev=2)

omega1, p1 = normalize_eigenvector(mesh, E, 0, absolute=False, degree=degree, which='right')

u = velocity_eigenvector(mesh, p1, omega1, rho, absolute=True)

from helmholtz_x.dolfinx_utils import absolute, phase
p_abs = absolute(p1)
p_phase = phase(p1, deg=True)

xdmf_writer("Results/p1_active", mesh, p1)
xdmf_writer("Results/p1_active_abs", mesh, p_abs)
xdmf_writer("Results/p1_active_phase", mesh, p_phase)
xdmf_writer("Results/u", mesh, u)
execution_time(start)
