from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, OneDimensionalSetup
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import c_step
import matplotlib.pyplot as plt
import params_dim
import numpy as np
import sys
start = start_time()

# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 300

mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions

# boundary_conditions = {1: {'Robin': params_dim.R_in},  # inlet
#                        2: {'Robin': params_dim.R_out}}  # outlet
# boundary_conditions = {1: {'Dirichlet'},  # inlet
#                        2: {'Dirichlet'}}  # outlet}

boundary_conditions = {}

# Define Speed of sound

c = c_step(mesh, params_dim.x_f, params_dim.c_u, params_dim.c_u)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target = 200 * 2 * np.pi # 150 * 2 * np.pi

E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= True)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

if '-nopopup' not in sys.argv:
    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0].plot(uh.x.array.real)
    ax[1].plot(uh.x.array.imag)
    plt.savefig("Results/"+"1DPassive"+".png")

execution_time(start)