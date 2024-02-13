import numpy as np

p_gas = 100000 # Pa
r_gas = 287.1

# Passive Flame Data
Temperature = 1000 # K

# Active Flame Data
T_mean = Temperature
T_flame = 1500 #K

x_flame = np.array([0., 0., 0.50]) #m
a_flame = 0.025 
x_ref   = np.array([0., 0., 0.35]) #m
a_ref   = 0.025

FTF_mag = 1
tau = 0.2E-3
Q_tot = -57015.232012607579  # W
U_bulk = 11.485465769828917 # m/s

eta = FTF_mag * Q_tot / U_bulk

if __name__ == '__main__':

    from helmholtz_x.parameters_utils import rho_ideal, gaussianFunction, gamma_function, temperature_step, halfGaussianFunction
    from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer

    tube = XDMFReader("MeshDir/FlamedDuct/tube")
    mesh, subdomains, facet_tags = tube.getAll()
    tube.getInfo()

    gamma_in = gamma_function(mesh,Temperature)
    print("Gamma in:", gamma_in)
    gamma_out = gamma_function(mesh,T_flame)
    print("Gamma out:", gamma_out)

    temp = temperature_step(mesh, x_flame, T_mean, T_flame)
    xdmf_writer("InputFunctions/T_active",mesh, temp)

    gamma = gamma_function(mesh, temp)
    xdmf_writer("InputFunctions/gamma",mesh, gamma)

    w = gaussianFunction(mesh, x_ref, a_ref)
    xdmf_writer("InputFunctions/w",mesh, w)

    h = gaussianFunction(mesh, x_flame, a_flame)
    xdmf_writer("InputFunctions/h",mesh, h)

    h_half = halfGaussianFunction(mesh, x_flame, a_flame)
    xdmf_writer("InputFunctions/h_half",mesh, h_half)

    rho = rho_ideal(mesh, temp, p_gas, r_gas)
    xdmf_writer("InputFunctions/rho",mesh, rho)