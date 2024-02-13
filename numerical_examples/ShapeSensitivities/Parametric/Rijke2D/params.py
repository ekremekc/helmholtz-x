"""
A note on the nomenclature:
dim ~ dimensional quantity
ref ~ reference quantity for the non-dimensionalization
in ~ inlet, same as u ~ upstream (of the flame)
out ~ outlet, same as d ~ downstream (of the flame)
"""

from math import *
import numpy as np

# ------------------------------------------------------------

L_ref = 1.  # [m]

# ------------------------------------------------------------

r = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r*rho_amb)  # [K]

c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_in_dim = rho_amb  # [kg/m^3]
rho_out_dim = 0.85  # [kg/m^3]

# print('rho_in_dim = %f' % rho_in_dim)
# print('rho_out_dim = %f' % rho_out_dim)

T_in_dim = p_amb/(r*rho_in_dim)  # [K]
T_out_dim = p_amb/(r*rho_out_dim)  # [K]

# print('T_in_dim = %f' % T_in_dim)
# print('T_out_dim = %f' % T_out_dim)

c_in_dim = sqrt(gamma*p_amb/rho_in_dim)  # [kg/m^3]
c_out_dim = sqrt(gamma*p_amb/rho_out_dim)  # [kg/m^3]

# print('c_in_dim = %f' % c_in_dim)
# print('c_out_dim = %f' % c_out_dim)

# ------------------------------------------------------------
# Reflection coefficients

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# Acoustic impedance

# Z_in = rho_amb*c_amb*(1 + R_in)/(1 - R_in)
# Z_out = rho_amb*c_amb*(1 + R_out)/(1 - R_out)

# print('Z_in =', Z_in)
# print('Z_out =', Z_out)

# Specific impedance

Z_in = (1 + R_in)/(1 - R_in)
Z_out = (1 + R_out)/(1 - R_out)

# print('Z_in =', Z_in)
# print('Z_out =', Z_out)

# Specific admittance

Y_in = 1/Z_in
Y_out = 1/Z_out

# print('Y_in =', Y_in)
# print('Y_out =', Y_out)

# ------------------------------------------------------------
# Flame transfer function

Q_tot = 200.  # [W]
U_bulk = 0.1  # [m/s]
N = 0.014  # [/]

n_dim = N*Q_tot/U_bulk  # [J/m]

n_dim /= pi/4 * 0.047

"""[n_dim is case dependent]

n_dim = N*Q_tot/U_bulk  # [J/m]

1D - n_dim /= pi/4 * 0.047**2
2D - n_dim /= pi/4 * 0.047
3D - n_dim = n_dim
"""

# print('n_dim = %f' % n_dim)

tau_dim = 0.0015  # [s]

# ------------------------------------------------------------

x_f_dim = np.array([[0.25, 0., 0.]])  # [m]
a_f_dim = 0.025  # [m]
# a_f_dim = 0.0047  # [m]

# print('a_f_dim = %f' % a_f_dim)

x_r_dim = np.array([[0.20, 0., 0.]])  # [m]
# x_r_dim = 0.2  # [m]
# a_r_dim = 0.0047  # [m]

# print('a_r_dim = %f' % a_r_dim)

# ------------------------------------------------------------
# ------------------------------------------------------------
# Non-dimensionalization

U_ref = c_amb  # [m/s]
p_ref = p_amb  # [Pa]

# ------------------------------------------------------------

rho_in = rho_in_dim*U_ref**2/p_ref
rho_out = rho_out_dim*U_ref**2/p_ref

# print('rho_in = %f' % rho_in)
# print('rho_out = %f' % rho_out)

T_in = T_in_dim*r/U_ref**2
T_out = T_out_dim*r/U_ref**2

# print('T_in = %f' % T_in)
# print('T_out = %f' % T_out)

c_in = c_in_dim/U_ref
c_out = c_out_dim/U_ref

# print('c_in = %f' % c_in)
# print('c_out = %f' % c_out)

# ------------------------------------------------------------

n = n_dim/(p_ref*L_ref**2)

tau = tau_dim*U_ref/L_ref

# ------------------------------------------------------------

x_f = x_f_dim/L_ref
x_r = x_r_dim/L_ref

a_f = a_f_dim/L_ref
# a_r = a_r_dim/L_ref

# ------------------------------------------------------------


from dolfinx.fem import Function,FunctionSpace

def c(mesh):
    V = FunctionSpace(mesh, ("DG", 0))
    c = Function(V)
    x = V.tabulate_dof_coordinates()
    global c_in
    global c_out
    global x_f
    global a_f
    x_f = x_f[0][0]
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[0]< x_f:
            c.vector.setValueLocal(i, c_in)
        else:
            c.vector.setValueLocal(i, c_out)
    return c
# c = dolf.Expression('x[0] <= x_f ? c_in : c_out', degree=0, x_f=x_f[0][0], c_in=c_in, c_out=c_out)

# rho = dolf.Expression("rho_u+0.5*(rho_d-rho_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
#                  rho_u = rho_in,
#                  rho_d = rho_out,
#                  x_f = x_f[0][0],
#                  a_f = a_f)

# c_ = dolf.Expression("sqrt(gamma*p_amb/rho)", degree = 1,
#                gamma = gamma,
#                p_amb = p_amb/p_ref,
#                rho = rho) # Variable Speed of sound (m/s)

# # c_ = dolf.Expression("c_u+0.5*(c_d-c_u)*(1+tanh((x[0]-x_f)/a_f))", degree=1,
# #                  c_u = c_in,
# #                  c_d = c_out,
# #                  x_f = x_f[0][0],
# #                  a_f = a_f)


# string_f_1d = '1 / (sigma * sqrt(2*pi)) * exp(- pow(x[0] - x_0, 2) / (2 * pow(sigma, 2)) )'

# v = dolf.Expression(string_f_1d, degree=0, x_0=x_f[0][0], sigma=a_f)
# w = dolf.Expression(string_f_1d, degree=0, x_0=x_r[0][0], sigma=a_f)

# string_w_1d = '1 / (sigma * sqrt(2*pi)) * exp(- pow(x[0] - x_0, 2) / (2 * pow(sigma, 2)) )/rho'
# w_r = dolf.Expression(string_f_1d, degree=0, x_0=x_r[0][0], sigma=a_f, rho = rho)


