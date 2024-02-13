from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import OneDimensionalSetup
from helmholtz_x.io_utils import xdmf_writer
from helmholtz_x.solver_utils import start_time, execution_time
from helmholtz_x.parameters_utils import c_step
import matplotlib.pyplot as plt
import params_dim
import numpy as np
import sys

start = start_time()

# approximation space polynomial degree
degree = 1

n_elem = 300

mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

boundary_conditions = {}

c = c_step(mesh, params_dim.x_f, params_dim.c_u, params_dim.c_u)

# Introduce Passive Flame Matrices
matrices = AcousticMatrices(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target = 200 * 2 * np.pi

E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= True)

omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, p)

if '-nopopup' not in sys.argv:
    fig, ax = plt.subplots(2, figsize=(12, 6))
    ax[0].plot(p.x.array.real)
    ax[1].plot(p.x.array.imag)
    plt.savefig("Results/"+"1DPassive"+".png")

execution_time(start)