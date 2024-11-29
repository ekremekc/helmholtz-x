from helmholtz_x.parameters_utils import c_step, gaussianFunction
from dolfinx.fem import Function, functionspace
from math import sqrt
import numpy as np

r_gas = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r_gas*rho_amb)  # [K]
c_amb = sqrt(gamma*p_amb/rho_amb)  # [m/s]

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

T_in = p_amb/(r_gas*rho_u)  # [K]
T_out = p_amb/(r_gas*rho_d)  # [K]

c_in = sqrt(gamma*p_amb/rho_u)  # [kg/m^3]
c_out = sqrt(gamma*p_amb/rho_d)  # [kg/m^3]

# Reflection coefficients

R_in = - 0.975 - 0.05j  # [/] #\abs(Z} e^{\angle(Z) i} 
R_out = - 0.975 - 0.05j  # [/]

# Flame transfer function

q_0 = 200.  # [W]
u_b = 0.1  # [m/s]
n =  0.014  # [/]

tau = 0.0015 #s

x_f = np.array([[0., 0., 0.25]])  # [m]
a_f = 0.025  # [m]

x_r = np.array([[0., 0., 0.20]])  # [m]
a_r = 0.025  # [m]

def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))

def rho_func(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    V = functionspace(mesh, ("CG", degree))
    rho = Function(V)
    if mesh.geometry.dim == 1 or mesh.geometry.dim == 2:
        x_f = x_f[0]
        x_f = x_f[0]
        rho.interpolate(lambda x: density(x[0], x_f, a_f, rho_d, rho_u))
    elif mesh.geometry.dim == 3:
        x_f = x_f[0]
        x_f = x_f[2]
        rho.interpolate(lambda x: density(x[2], x_f, a_f, rho_d, rho_u))
    return rho

if __name__ == '__main__':
    from helmholtz_x.io_utils import XDMFReader,xdmf_writer

    RijkeTube3D = XDMFReader("MeshDir/Original/mesh")
    mesh, subdomains, facet_tags = RijkeTube3D.getAll()

    rho_func = rho_func(mesh, x_f, a_f, rho_d, rho_u)
    w_func = gaussianFunction(mesh, x_r, a_r)
    h_func = gaussianFunction(mesh, x_f, a_f)
    
    c_func = c_step(mesh, x_f, c_in, c_out)

    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)