from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import rho_ideal, temperature_uniform

start = start_time()
import params

# Read mesh 
LPP = XDMFReader("MeshDir/thinAnnulus")
mesh, subdomains, facet_tags = LPP.getAll()
LPP.getInfo()

boundary_conditions = {}

degree = 1

target_dir = 400
T = temperature_uniform(mesh, params.T_ambient)
rho = rho_ideal(mesh, T, params.p_gas, params.r_gas)
matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

E = pep_solver(matrices.A, matrices.B, matrices.C, target_dir, nev=10, print_results=False)

omega, p_undamped = normalize_eigenvector(mesh, E, 4, absolute=True, degree=degree, which='right')
xdmf_writer("Results/p_undamped", mesh, p_undamped)

u = velocity_eigenvector(mesh, p_undamped, omega, rho, absolute=True)
xdmf_writer("Results/u_undamped", mesh, u)

execution_time(start)
