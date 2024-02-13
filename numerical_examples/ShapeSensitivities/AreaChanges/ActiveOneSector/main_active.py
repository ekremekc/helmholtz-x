from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector_holes
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer
from helmholtz_x.parameters_utils import rho_ideal
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import gaussianFunction
start = start_time()
import params
import numpy as np

# Read mesh 
LPP = XDMFReader("MeshDir/thinAnnulus")
mesh, subdomains, facet_tags = LPP.getAll()
LPP.getInfo()

M_inlet  =  3.0741653557135287E-002
M_outlet = 7.0200419278181339E-002

boundary_conditions = {32:{"ChokedInlet":M_inlet},
                       39:{"ChokedOutlet":M_outlet}}

degree = 1

T = params.temperature(mesh, params.T_ambient, params.T_flame, params.T_cooling_duct)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, holes=params.holes)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

w = gaussianFunction(mesh,params.x_r,params.a_r)
h = gaussianFunction(mesh,params.x_f,params.a_f)
rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params.eta, params.tau, degree=degree)

target_dir = 400*2*np.pi # 390.62*2*np.pi

E = fixed_point_iteration_pep(matrices, D, target_dir, nev=2, print_results= False)

omega2, p_damped2 = normalize_eigenvector(mesh, E, 0, absolute=True, degree=degree, which='right')
xdmf_writer("Results/p_damped_axial", mesh, p_damped2)

omega_u, p_u = normalize_eigenvector(mesh, E, 0, absolute=False, degree=degree, which='right')

u = velocity_eigenvector_holes(mesh, subdomains, p_u, omega_u, rho, params.holes, absolute=True)
xdmf_writer("Results/u", mesh, u)

execution_time(start)