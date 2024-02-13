from dolfinx.fem import Function, FunctionSpace
from helmholtz_x.dolfinx_utils import XDMFReader, cart2cyl,  xdmf_writer
from helmholtz_x.parameters_utils import gaussianFunction, halfGaussianFunction
import numpy as np

z_injector = 0.15
z_flame = 0.2
z_cooling = 0.35
z_cc_end = 0.5

r_cc_inner = 0.243908
r_cc_outer = 0.353552
r_flame = (r_cc_inner + r_cc_outer)/2
r_injector = r_flame

T_ambient = 1000 # K
p_gas = 50e5 # Pa
r_gas = 287.1

T_flame = 2500 #K
T_cooling_duct = 1990 #K

x_r = np.array([0., r_injector, z_injector])
a_r = 0.005

x_f = np.array([0., r_flame, z_flame])
a_f = 0.025

FTF_mag = 1.0
tau = 0.2E-3
Q_tot = -116160941.07282934/20  # W
U_bulk = 193.17303635406606 # m/s

eta = FTF_mag * Q_tot / U_bulk

L = 0.02706 # m
U = 178.84296066993633 # m/s

h_vc = 0.33
h_g = 1.
sigma = h_vc/h_g # contraction coefficient
holes = {1:{'L':L, 'U':U, 'sigma':sigma, 'tag':2}}

def temperature(mesh, T_ambient, T_flame, T_cooling_duct):
    V = FunctionSpace(mesh, ("CG", 1))
    temp = Function(V)
    temp.name="temperature"
    x_tab = V.tabulate_dof_coordinates()
    temp.x.array[:] = T_ambient
    
    for i in range(x_tab.shape[0]):
        midpoint = x_tab[i,:]
        x = midpoint[0]
        y = midpoint[1]
        z = midpoint[2]
        
        r, theta, z_cyc = cart2cyl(x,y,z)
        
        if z_cyc>=z_flame and r>r_cc_inner and r<r_cc_outer and z_cyc<z_cooling:
            value = T_flame

        elif z_cyc>=z_cooling and r>r_cc_inner and r<r_cc_outer :
            value = T_cooling_duct

        else:
            value = T_ambient

        temp.vector.setValueLocal(i, value)

    return temp

M_inlet  =  3.0741653557135287E-002
M_outlet = 7.0200419278181339E-002

boundary_conditions = { 30:{'Master'},42:{'Master'},
                        1030:{'Slave'},1042:{'Slave'},
                        32:{"ChokedInlet":M_inlet},
                        39:{"ChokedOutlet":M_outlet},
                        17 : {'Neumann'},
                        #18 : {'Neumann'},
                        #19 : {'Neumann'},
                        #20 : {'Neumann'},
                        21 : {'Neumann'},
                        #22 : {'Neumann'},
                        #23 : {'Neumann'},
                        24 : {'Neumann'},
                        #25 : {'Neumann'},
                        #26 : {'Neumann'},
                        27 : {'Neumann'},
                        #28 : {'Neumann'},
                        29 : {'Neumann'},
                        31 : {'Neumann'},
                        32 : {'Neumann'},
                        33 : {'Neumann'},
                        34 : {'Neumann'},
                        35 : {'Neumann'},
                        36 : {'Neumann'},
                        37 : {'Neumann'},
                        38 : {'Neumann'},
                        39 : {'Neumann'},
                        40 : {'Neumann'},
                        41 : {'Neumann'}}

if __name__ == '__main__':

    LPP = XDMFReader("MeshDir/thinAnnulus")
    mesh, subdomains, facet_tags = LPP.getAll()
    LPP.getInfo()

    temp = temperature(mesh, T_ambient, T_flame, T_cooling_duct)
    xdmf_writer("InputFunctions/T",mesh, temp)

    h = halfGaussianFunction(mesh,x_f,a_f)
    xdmf_writer("InputFunctions/h",mesh, h)

    w = gaussianFunction(mesh,x_r,a_r)
    xdmf_writer("InputFunctions/w",mesh, w)