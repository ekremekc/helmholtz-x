from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer
from helmholtz_x.parameters_utils import temperature_uniform
from helmholtz_x.solver_utils import start_time, execution_time
import params

start = start_time()

# Read mesh 
tube = XDMFReader("MeshDir/FlamedDuct/tube")

mesh, subdomains, facet_tags = tube.getAll()
tube.getInfo()

M0 = 9.2224960671405849E-003
M1 = 9.2230258764948152E-003

boundary_conditions = {3:{"ChokedInlet":M0},
                       8:{"ChokedOutlet":M1}}

degree = 1

T = temperature_uniform(mesh, params.Temperature)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, degree=degree)
matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target_dir = 250
E = pep_solver(matrices.A, matrices.B, matrices.C, target_dir, nev=10, print_results=True)

omega1, p1 = normalize_eigenvector(mesh, E, 2, degree=degree, which='right')
xdmf_writer("Results/p1_passive", mesh, p1)

execution_time(start)