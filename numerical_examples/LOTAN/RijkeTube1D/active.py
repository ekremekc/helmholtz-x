from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_matrices import ActiveFlameNT
from helmholtz_x.eigenvectors import normalize_eigenvector, velocity_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, OneDimensionalSetup
from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, rho
from helmholtz_x.solver_utils import start_time, execution_time
import numpy as np
import params_dim
import sys
start = start_time()

# approximation space polynomial degree
degree = 1

# number of elements in each direction of mesh
n_elem = 3000
mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions

boundary_conditions = {}

# Introduce Passive Flame Matrices

T = temperature_step(mesh, params_dim.x_f, params_dim.T_u, params_dim.T_d)

matrices = AcousticMatrices(mesh, subdomains, facet_tags, boundary_conditions, T, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# Introduce Active Flame Matrix parameters

rho = rho(mesh, params_dim.x_f, params_dim.a_f, params_dim.rho_d, params_dim.rho_u)
w = gaussianFunction(mesh, params_dim.x_r, params_dim.a_r)
h = gaussianFunction(mesh, params_dim.x_f, params_dim.a_f)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params_dim.eta, params_dim.tau, degree=degree)

# Introduce solver object and start

target = 200 * 2 * np.pi
E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

# Extract eigenvalue and normalized eigenvector 

omega, p = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector

v = velocity_eigenvector(mesh, p, omega, rho, degree=2)

xdmf_writer("Results/p", mesh, p)

if '-nopopup' not in sys.argv:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, figsize=(12, 12))
    ax[0].plot(p.x.array.real)
    ax[0].set_ylabel(r"$P_r$")
    ax[1].plot(p.x.array.imag)
    ax[1].set_ylabel(r"$P_i$")

    ax[2].plot(v.x.array.real)
    ax[2].set_ylabel(r"$U_r$")
    ax[3].plot(v.x.array.imag)
    ax[3].set_ylabel(r"$U_i$")

plt.show()
plt.savefig("Results/"+"1DActive"+".png")

execution_time(start)