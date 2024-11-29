from helmholtz_x.parameters_utils import gaussianFunction, rho_step
from dolfinx.fem import functionspace, Function
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

# For 1D dimensional consistency
d_tube = 0.047
S_c = np.pi * d_tube **2 / 4
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
    V = functionspace(mesh, ("CG", degree))
    rho = Function(V)
    # x = V.tabulate_dof_coordinates()   
    if mesh.geometry.dim == 1 or mesh.geometry.dim == 2:
        x_f = x_f[0]
        x_f = x_f[0]
        rho.interpolate(lambda x: density(x[0], x_f, a_f, rho_d, rho_u))
    elif mesh.geometry.dim == 3:
        x_f = x_f[2]
        rho.interpolate(lambda x: density(x[2], x_f, a_f, rho_d, rho_u))
    return rho

if __name__ == '__main__':
    from helmholtz_x.dolfinx_utils import OneDimensionalSetup
    import matplotlib.pyplot as plt
    n_elem = 100
    mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)
    x_coords = np.linspace(0,1,n_elem+1)

    w_plot = gaussianFunction(mesh, x_r, a_r)
    plt.plot(x_coords, w_plot.x.array.real)
    plt.ylabel("w")
    plt.savefig("InputFunctions/w.png")
    plt.clf()

    h_plot = gaussianFunction(mesh, x_f, a_f)
    plt.plot(x_coords, h_plot.x.array.real)
    plt.ylabel("h")
    plt.savefig("InputFunctions/h.png")
    plt.clf()

    rho_plot = rho_func(mesh, x_f, a_f, rho_d, rho_u)
    plt.plot(x_coords, rho_plot.x.array.real)
    plt.ylabel(r"$\rho$")
    plt.savefig("InputFunctions/rho.png")
    plt.clf()