from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer,  XDMFReader, phase, absolute
from helmholtz_x.parameters_utils import temperature_step, c_step, gaussianFunction, rho
from helmholtz_x.solver_utils import start_time, execution_time
start = start_time()
import numpy as np
import  params

# approximation space polynomial degree
degree = 1

rijke3d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke3d.getAll()
rijke3d.getInfo()

# Define the boundary conditions
boundary_conditions = {3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

# Define Speed of sound

c = c_step(mesh, params.x_f, params.c_u, params.c_d)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

rho = rho(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
T = temperature_step(mesh, params.x_f, params.T_u, params.T_d) 

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params.eta, params.tau, degree=degree)

target = 200 * 2 * np.pi 
E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

p_abs = absolute(uh)
xdmf_writer("Results/p_abs", mesh, p_abs)
p_phase = phase(uh)
xdmf_writer("Results/p_phase", mesh, p_phase)

execution_time(start)