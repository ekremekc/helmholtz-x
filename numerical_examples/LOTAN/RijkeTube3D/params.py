import numpy as np

r_gas = 287.  # [J/kg/K]
gamma = 1.4  # [/]

p_amb = 1e5  # [Pa]
rho_amb = 1.22  # [kg/m^3]

T_amb = p_amb/(r_gas*rho_amb)  # [K]

c_amb = np.sqrt(gamma*p_amb/rho_amb)  # [m/s]

# ------------------------------------------------------------

rho_u = rho_amb  # [kg/m^3]
rho_d = 0.85  # [kg/m^3]

c_u = np.sqrt(gamma*p_amb/rho_u) 
c_d = np.sqrt(gamma*p_amb/rho_d) 

T_u = c_u**2/(gamma*r_gas)
T_d = c_d**2/(gamma*r_gas)

d_tube = 0.047

# ------------------------------------------------------------
# Flame transfer function

FTF_mag =  0.1  # [/]
Q_tot = -27.008910380099735 # [W]
U_bulk = 0.10066660027273297

eta = FTF_mag * Q_tot / U_bulk

tau = 0.0015

x_f = np.array([0., 0., 0.25])  # [m]
a_f = 0.025  # [m]

x_r = np.array([0., 0., 0.20])  # [m]
a_r = 0.025

if __name__ == '__main__':

    from helmholtz_x.io_utils import XDMFReader,xdmf_writer
    from helmholtz_x.parameters_utils import c_step, rho_step, gaussianFunction

    rijke3d = XDMFReader("MeshDir/mesh")
    mesh, subdomains, facet_tags = rijke3d.getAll()

    rho_func = rho_step(mesh, x_f, a_f, rho_d, rho_u)
    w_func = gaussianFunction(mesh, x_r, a_r)
    h_func = gaussianFunction(mesh, x_f, a_f)
    
    c_func = c_step(mesh, x_f, c_u, c_d)

    xdmf_writer("InputFunctions/rho", mesh, rho_func)
    xdmf_writer("InputFunctions/w", mesh, w_func)
    xdmf_writer("InputFunctions/h", mesh, h_func)
    xdmf_writer("InputFunctions/c", mesh, c_func)
