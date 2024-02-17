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

# Flame transfer function

FTF_mag =  0.1  # [/]
Q_tot = -27.008910380099735 # [W]
U_bulk = 0.10066660027273297

eta = FTF_mag * Q_tot / U_bulk
tau = 0.0015

# For dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4
eta /=  S_c

x_f = np.array([0.25, 0., 0.])  # [m]
a_f = 0.025  # [m]

x_r = np.array([0.20, 0., 0.])  # [m]
a_r = 0.025

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from helmholtz_x.parameters_utils import c_step, rho_step, gaussianFunction
    from helmholtz_x.dolfinx_utils import OneDimensionalSetup
    n_elem = 3000
    mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)
    
    fig = plt.figure(figsize=(8,3))
    
    w_plot = gaussianFunction(mesh, x_r, a_r)
    plt.plot(mesh.geometry.x[:,0], w_plot.x.array.real, color="blue")
    plt.xlabel("x")
    plt.ylabel(r"$w$")
    plt.tight_layout()
    plt.savefig("InputFunctions/w.png")
    plt.clf()
    
    h_plot = gaussianFunction(mesh, x_f, a_f)
    plt.plot(mesh.geometry.x[:,0], h_plot.x.array.real, color="red")
    plt.xlabel("x")
    plt.ylabel("h")
    plt.tight_layout()
    plt.savefig("InputFunctions/h.png")
    plt.clf()

    rho_plot = rho_step(mesh, x_f, a_f, rho_d, rho_u)
    plt.plot(mesh.geometry.x[:,0], rho_plot.x.array.real)
    plt.xlabel("x")
    plt.ylabel(r"$\rho$")
    plt.tight_layout()
    plt.savefig("InputFunctions/rho.png")
    plt.clf()
    
    c_plot = c_step(mesh, x_f, c_u, c_d)
    plt.plot(mesh.geometry.x[:,0], c_plot.x.array.real)
    plt.xlabel("x")
    plt.ylabel(r"$c$")
    plt.tight_layout()
    plt.savefig("InputFunctions/c.png")
    plt.clf()