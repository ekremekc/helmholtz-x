from math import *
import numpy as np
from scipy.io import loadmat


def cyl2cart(rho, phi, zeta):
    # cylindrical to Cartesian
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = zeta
    return x, y, z


def cart2cyl(x, y, z):
    # cylindrical to Cartesian
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    zeta = z
    return rho, phi, zeta


# ------------------------------------------------------------

r_p = .14  # [m]
R_p = .07
l_p = .07

l_b = .014
l_pp = .006
l_f = .006
h_b = .0165
h_pp = .00945
h_f = .018
d_1 = .025
d_2 = .035

r_cc = .15
R_cc = .05
l_cc = .2

l_ec = .041  # end correction

# flame
r_f = r_p + d_2  # [m]
theta = np.deg2rad(22.5)  # [rad]
z_f = 0  # [m]

# reference
r_r = r_f
z_r = - 0.02  # [m]

# ------------------------------------------------------------

r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 101325.  # [Pa]

T_amb = 300.  # [K]

rho_amb = p_amb/(r*T_amb)  # [kg/m^3]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

T_a = 1521.  # [K] at z = 0
T_b = 1200.  # [K] at z = l_cc

# ------------------------------------------------------------

# Flame transfer function

Q_tot = 2080.  # [W] **per burner**
U_bulk = 0.66  # [m/s]

# n = Q_tot/U_bulk  # [J/m]
N3 = 1  # [/]

tau = 0.003  # [s]

mat = loadmat('ftf.mat')

S1 = mat['A']
s2 = mat['b']
s3 = mat['c']
s4 = mat['d']

# ------------------------------------------------------------

# x_f = np.array([cyl2cart(r_f, 0*theta, z_f)])
x_f = np.array([cyl2cart(r_f, i*theta, z_f) for i in range(16)])

# x_r = np.array([cyl2cart(r_r, 0*theta, z_r)])
x_r = np.array([cyl2cart(r_r, i*theta, z_r) for i in range(16)])

# ------------------------------------------------------------
from dolfinx.fem import Function,FunctionSpace

def c(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    global gamma
    global r
    global l_cc
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        # print(midpoint)
        if midpoint[2]< 0:
            c.vector.setValueLocal(i, sqrt(gamma * r * 300.))
        elif midpoint[2]> 0 and midpoint[2]< l_cc:
            
            c.vector.setValueLocal(i, sqrt(gamma * r * ((1200. - 1521.) * (midpoint[2]/l_cc)**2 + 1521.)))
        else:
            c.vector.setValueLocal(i, sqrt(gamma * r * 1200.))
    return c