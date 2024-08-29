from helmholtz_x.dolfinx_utils import RectangleSetup
from helmholtz_x.parameters_utils import c_uniform
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.eigenvectors import normalize_eigenvector
import matplotlib.pyplot as plt
import numpy as np

def calc_R(Z):
    return (Z-1)/(Z+1)

L, h = 0.4, 0.1
nx, ny = 160, 40

mesh, subdomains, facet_tags = RectangleSetup(nx, ny, L, h)

c0 = 450
c = c_uniform(mesh, c0)

def helmholtzx_2d(Z,target):
    # Define the boundary conditions
    boundary_conditions = {4: {'Robin': calc_R(Z)}} # top
    # Introduce Passive Flame Matrices
    matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, c, degree=1)
    E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= False)
    # Extract angular eigenfrequency
    omega, _ = normalize_eigenvector(mesh, E, i=0, degree=1, which='right')
    return omega

# Number of numerical eigenmode calculations
N_numeric = 10

# PURE IMAGINARY CASE
Z_helmholtz_b = np.linspace(-10j, 10j, N_numeric)
f_helmholtz_b = np.zeros_like(Z_helmholtz_b)
omega_init = 600*2*np.pi
for ind, impedance in enumerate(Z_helmholtz_b):
    if ind==int(N_numeric/2):
        omega_init = 250*2*np.pi
    omega_helmx = helmholtzx_2d(impedance,omega_init)
    f_helmholtz_b[ind]= omega_helmx/2/np.pi
    omega_init = omega_helmx*1.1

# PURE REAL CASE
Z_helmholtz_a = np.linspace(-10, 10, N_numeric)
f_helmholtz_a = np.zeros_like(Z_helmholtz_a,dtype=np.complex128)
omega_init = 600*2*np.pi
for ind, impedance in enumerate(Z_helmholtz_a):
    if ind==int(N_numeric/2):
        omega_init = 600*2*np.pi
    omega_helmx = helmholtzx_2d(impedance,omega_init)
    f_helmholtz_a[ind]= omega_helmx/2/np.pi
    omega_init = omega_helmx

# Analytical results from MATLAB
analytic_b_real, analytic_b_imag = [], []
analytic_a_real, analytic_a_imag = [], []

f=open('matlab_data/analytical.txt',"r")
lines=f.readlines()
for x in lines:
    analytic_b_real.append(float(x.split(' ')[0]))
    analytic_b_imag.append(float(x.split(' ')[1]))
    analytic_a_real.append(float(x.split(' ')[2]))
    analytic_a_imag.append(float(x.split(' ')[3]))
f.close()
Z_analytic = np.linspace(-10, 10, len(analytic_a_real))

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8,2))
ax1.plot(Z_analytic, analytic_b_real)
ax1.plot(Z_helmholtz_b.imag, f_helmholtz_b.real,'^')
ax1.set_xlabel(r"$b$")
ax1.set_ylabel(r"Re{$f$}")

ax2.plot(Z_analytic, analytic_b_imag, label="analytical")
ax2.plot(Z_helmholtz_b.imag, f_helmholtz_b.imag,'^', label='helmholtz-x')
ax2.set_ylim([-0.1, 0.1])
ax2.set_xlabel(r"$b$")
ax2.set_ylabel(r"Im{$f$}")
ax2.legend()

fig.tight_layout()
plt.savefig("A2_manufacturedSolution_a.pdf", bbox_inches='tight')
plt.show()

fig, ((ax3, ax4)) = plt.subplots(1, 2, figsize=(8,2))

ax3.plot(Z_analytic, analytic_a_real)
ax3.plot(Z_helmholtz_a.real, f_helmholtz_a.real,'^')
ax3.set_xlabel(r"$a$")
ax3.set_ylabel(r"Re{$f$}")

ax4.plot(Z_analytic, analytic_a_imag, label="analytical")
ax4.plot(Z_helmholtz_a.real, f_helmholtz_a.imag,'^', label='helmholtz-x')
ax4.set_xlabel(r"$a$")
ax4.set_ylabel(r"Im{$f$}")
ax4.legend()

fig.tight_layout()
plt.savefig("A2_manufacturedSolution_b.pdf", bbox_inches='tight')
plt.show()