from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, XDMFReader
from helmholtz_x.parameters_utils import temperature_step, c_step,rho, gaussianFunction
from helmholtz_x.solver_utils import start_time, execution_time
start = start_time()
import numpy as np
import  params

# approximation space polynomial degree
degree = 1

rijke2d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke2d.getAll()
rijke2d.getInfo()

# Define the boundary conditions
boundary_conditions = {4: {'Neumann'},
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Introduce Passive Flame Matrices
T = temperature_step(mesh, params.x_f, params.T_u, params.T_d) 

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, T, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

rho = rho(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params.eta, params.tau, degree=degree)

target = 200 * 2 * np.pi # 150 * 2 * np.pi
E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

execution_time(start)