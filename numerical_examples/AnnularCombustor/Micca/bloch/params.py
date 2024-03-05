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
N_sector = 1
x_f = np.array([cyl2cart(r_f, i*theta, z_f) for i in range(N_sector)])
x_r = np.array([cyl2cart(r_r, i*theta, z_r) for i in range(N_sector)])

# Outlet reflection coefficient
R_outlet = -0.875-0.2j

from dolfinx.fem import Function, FunctionSpace

# Axial speed of sound field
def c(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    global gamma
    global r
    global l_cc
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[2]< 0:
            c.vector.setValueLocal(i, sqrt(gamma * r * 300.))
        elif midpoint[2]> 0 and midpoint[2]< l_cc:
            
            c.vector.setValueLocal(i, sqrt(gamma * r * ((1200. - 1521.) * (midpoint[2]/l_cc)**2 + 1521.)))
        else:
            c.vector.setValueLocal(i, sqrt(gamma * r * 1200.))
    # c.x.scatter_forward()
    return c