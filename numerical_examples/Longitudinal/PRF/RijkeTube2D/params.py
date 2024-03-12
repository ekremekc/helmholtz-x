from helmholtz_x.parameters_utils import gaussianFunction, rho_step
from dolfinx.fem import FunctionSpace, Function
import numpy as np

r_gas = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r_gas*rho_amb)  # [K]
c_amb = 339 # [m/s]

rho_in_dim = rho_amb  # [kg/m^3]
rho_out_dim = 0.85  # [kg/m^3]

c_in_dim = np.sqrt(gamma*p_amb/rho_in_dim)  # [kg/m^3]
c_out_dim = np.sqrt(gamma*p_amb/rho_out_dim)  # [kg/m^3]

T_in_dim = p_amb/(r_gas*rho_in_dim)  # [K]
T_out_dim = p_amb/(r_gas*rho_out_dim)  # [K]

# Reflection coefficients
R_in  = - 0.975 - 0.05j  # [/] 
R_out = - 0.975 - 0.05j  # [/]

# Flame transfer function
n =  0.014  # [/]
q_0 = 200.  # [W]
u_b = 0.1  # [m/s]

# For 2D dimensional consistency
d_tube = 0.047
S_c = (np.pi/4) * d_tube
n /=  S_c

tau_dim = 0.0015

x_f_dim = np.array([[0.25, 0., 0.]])  # [m]
a_f_dim = 0.025  # [m]

x_r_dim = np.array([[0.20, 0., 0.]])  # [m]
a_r_dim = 0.025

# Non-dimensionalization
L_ref = 1.  # [m]
U_ref = c_amb  # [m/s]
p_ref = p_amb  # [Pa]

rho_u = rho_in_dim*U_ref**2/p_ref
rho_d = rho_out_dim*U_ref**2/p_ref

c_u = c_in_dim/U_ref
c_d = c_out_dim/U_ref

T_u = T_in_dim*r_gas/U_ref**2
T_d = T_out_dim*r_gas/U_ref**2

# ------------------------------------------------------------
n = n/(p_ref*L_ref**2)
tau = tau_dim*U_ref/L_ref

# ------------------------------------------------------------
x_f = x_f_dim/L_ref
x_r = x_r_dim/L_ref

a_f = a_f_dim/L_ref
a_r = a_r_dim/L_ref

def density(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))

def rho_func(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    V = FunctionSpace(mesh, ("CG", degree))
    rho = Function(V)
    # x = V.tabulate_dof_coordinates()   
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
    from helmholtz_x.parameters_utils import c_step, gaussianFunction

    RijkeTube3D = XDMFReader("MeshDir/mesh")
    mesh, subdomains, facet_tags = RijkeTube3D.getAll()

    rho_func = rho_func(mesh, x_f, a_f, rho_d, rho_u)
    w_func = gaussianFunction(mesh, x_r, a_r)
    h_func = gaussianFunction(mesh, x_f, a_f)
    
    c_func = c_step(mesh, x_f, c_u, c_d)

    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)