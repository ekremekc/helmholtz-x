from math import *
import numpy as np
from scipy.io import loadmat
from helmholtz_x.dolfinx_utils import cyl2cart

# geometric data
l_cc = 0.2

# flame location
r_p = .14  # [m]
d_2 = .035 # [m]
r_f = r_p + d_2  # [m]
theta = np.deg2rad(22.5)  # [rad]
z_f = 0  # [m]

# measurement function location
r_r = r_f
z_r = - 0.02  # [m]

# ambient and other data
r = 287.  # [J/kg/K]
gamma = 1.4  # [/]
p_amb = 101325.  # [Pa]
T_amb = 300.  # [K
rho_amb = p_amb/(r*T_amb)  # [kg/m^3]
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

rho_xr = rho_amb 

T_a = 1521.  # [K] at z = 0
T_b = 1200.  # [K] at z = l_cc

# Experimental flame transfer function data
q_0 = 2080.  # [W] **per burner**
u_b = 0.66  # [m/s]
mat = loadmat('ftf.mat')
S1 = mat['A']
s2 = mat['b']
s3 = mat['c']
s4 = mat['d']

# Locations for flame and measurement functions
N_sector = 16
x_f = np.array([cyl2cart(r_f, i*theta, z_f) for i in range(N_sector)])
x_r = np.array([cyl2cart(r_r, i*theta, z_r) for i in range(N_sector)])

# Outlet reflection coefficient
R_outlet = -0.875-0.2j

from dolfinx.fem import Function, functionspace

# Axial speed of sound field
def c(mesh):
    V = functionspace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    global gamma
    global r
    global l_cc
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[2]< 0:
            c.x.petsc_vec.setValueLocal(i, sqrt(gamma * r * T_amb))
        elif midpoint[2]> 0 and midpoint[2]< l_cc:
            
            c.x.petsc_vec.setValueLocal(i, sqrt(gamma * r * ((T_b - T_a) * (midpoint[2]/l_cc)**2 + T_a)))
        else:
            c.x.petsc_vec.setValueLocal(i, sqrt(gamma * r * T_b))
    # c.x.scatter_forward()
    return c

if __name__ == '__main__':

    from helmholtz_x.flame_transfer_function import stateSpace
    from helmholtz_x.io_utils import XDMFReader,xdmf_writer
    from helmholtz_x.parameters_utils import Q_multiple
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from ufl import Measure
    import sys

    MICCA = XDMFReader("MeshDir/mesh")
    mesh, subdomains, facet_tags = MICCA.getAll()
    dx = Measure("dx", subdomain_data=subdomains)

    h = Q_multiple(mesh, subdomains, N_sector)
    xdmf_writer("InputFunctions/Q", mesh, h)

    speedOfSound = c(mesh)
    xdmf_writer("InputFunctions/c", mesh, speedOfSound)

    mat = loadmat('ftf.mat')
    freq = mat['freq'][0]
    f = mat['f'][0]
    fit = mat['fit'][0]

    A_ = mat['A']
    b_ = mat['b']
    c_ = mat['c']
    d_ = mat['d']

    ftf = stateSpace(A_, b_, c_, d_)

    f_ = np.linspace(0, 1060, 1061)
    omega_ = 2 * np.pi * f_

    z = np.zeros_like(omega_, dtype=np.complex128)
    for i in range(1061):
        z[i] = ftf(omega_[i])

    fig, ax = plt.subplots(2, figsize=(6, 4))
    ax[0].plot(freq, np.abs(f), 's', f_, np.abs(z))
    ax[0].set_ylabel("abs(FTF)")
    ax[1].plot(freq, np.angle(f), 's', f_, -np.angle(z))
    ax[1].set_ylabel("angle(FTF)")
    ax[1].set_xlabel("frequency [1/s]")
    ax[1].set_yticks(np.arange(-pi, pi+pi/2, step=(pi/2)), [r"$\pi$",r"$-\pi$/2",'0',r"$\pi$/2",r"$\pi$"])
    ax[0].grid()
    ax[1].grid()
    fig.tight_layout()
    plt.savefig("InputFunctions/ftf.pdf")

    if '-nopopup' not in sys.argv:
        plt.show()