import numpy as np

p_gas = 100000 # Pa
r_gas = 287.1

# Passive Flame Data
T_passive = 1000 # K

# Active Flame Data
T_flame = 1500 #K

x_flame = np.array([0., 0., 0.50]) #m
a_flame = 0.025 
x_ref   = np.array([0., 0., 0.35]) #m
a_ref   = 0.025

# Flame transfer function data
FTF_mag = 1
tau = 0.2E-3
Q_tot = -57015.232012607579  # W
U_bulk = 11.485465769828917 # m/s

eta = FTF_mag * Q_tot / U_bulk

# Choked boundary condition data
M0 = 9.2224960671405849E-003
M1 = 1.1408306741423997E-002

if __name__ == '__main__':

    from helmholtz_x.parameters_utils import rho_ideal, gaussianFunction, gamma_function, temperature_step, halfGaussianFunction
    from helmholtz_x.io_utils import XDMFReader, xdmf_writer

    tube = XDMFReader("MeshDir/mesh")
    mesh, subdomains, facet_tags = tube.getAll()
    tube.getInfo()

    temp = temperature_step(mesh, x_flame, T_passive, T_flame)
    xdmf_writer("InputFunctions/T_active",mesh, temp)

    gamma = gamma_function(temp)
    xdmf_writer("InputFunctions/gamma",mesh, gamma)

    w = gaussianFunction(mesh, x_ref, a_ref)
    xdmf_writer("InputFunctions/w",mesh, w)

    h = gaussianFunction(mesh, x_flame, a_flame)
    xdmf_writer("InputFunctions/h",mesh, h)

    h_half = halfGaussianFunction(mesh, x_flame, a_flame)
    xdmf_writer("InputFunctions/h_half",mesh, h_half)

    rho = rho_ideal(mesh, temp, p_gas, r_gas)
    xdmf_writer("InputFunctions/rho",mesh, rho)