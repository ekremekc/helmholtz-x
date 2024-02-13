from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector, normalize_adjoint
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer, derivatives_visualizer, dict_writer
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import gaussianFunction, sound_speed_variable_gamma, rho_ideal
from helmholtz_x.shape_derivatives_x import ShapeDerivativesAxial
start = start_time()
import params
import numpy as np

# Read mesh 
LPP = XDMFReader("MeshDir/thinAnnulus")
mesh, subdomains, facet_tags = LPP.getAll()
LPP.getInfo()

boundary_conditions = params.boundary_conditions

degree = 2

T = params.temperature(mesh, params.T_ambient, params.T_flame, params.T_cooling_duct)
c = sound_speed_variable_gamma(mesh, T)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, holes=params.holes, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

w = gaussianFunction(mesh,params.x_r,params.a_r)
h = gaussianFunction(mesh,params.x_f,params.a_f)
rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params.eta, params.tau, degree=degree)

target_dir = 350*2*np.pi 
target_adj = 350*2*np.pi 

E_dir = fixed_point_iteration_pep(matrices, D, target_dir, nev=2, print_results= False, problem_type='direct')
omega_dir, p_dir = normalize_eigenvector(mesh, E_dir, 0, absolute=False, degree=degree)
xdmf_writer("ShapeDerivatives/p_axial_dir", mesh, p_dir)

E_adj = fixed_point_iteration_pep(matrices, D, target_adj, nev=2, print_results= False, problem_type='adjoint')
omega_adj, p_adj = normalize_eigenvector(mesh, E_adj, 0, absolute=False, degree=degree)
xdmf_writer("ShapeDerivatives/p_axial_adj", mesh, p_adj)

p_adj_normalized = normalize_adjoint(omega_dir, p_dir, p_adj, matrices, D)
shape_derivatives = ShapeDerivativesAxial(LPP, boundary_conditions, omega_dir, p_dir, p_adj_normalized, c)

derivatives_visualizer("ShapeDerivatives/Derivatives", shape_derivatives, LPP, normalize=False)
dict_writer("ShapeDerivatives/Derivatives", shape_derivatives)

execution_time(start)
