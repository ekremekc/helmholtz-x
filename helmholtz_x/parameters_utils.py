from dolfinx.fem import functionspace, Function, form, Constant, assemble_scalar, locate_dofs_topological
from .dolfinx_utils import normalize, unroll_dofmap
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from ufl import Measure

def gaussian(x, x_ref, sigma, n):
    """Implements Gaussian function which integrates to 1 over the domain.
    # https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    Args:
        x (_type_): _Spatial coordinates
        x_ref (_type_): Reference point that Gaussian function implemented around_
        sigma (_type_): _description_
        n (_type_): spatial dimension that we want to implement

    Returns:
        _type_: numpy array of gaussian function
    """
    if len(x_ref)==1:
        x_ref = x_ref[0]

    if   n==1:
        spatial = (x[0]-float(x_ref[0]))**2
    elif n==2:
        spatial = (x[0]-x_ref[0])**2 + (x[1]-x_ref[1])**2
    elif n==3:
        spatial = (x[0]-x_ref[0])**2 + (x[1]-x_ref[1])**2 + (x[2]-x_ref[2])**2
    
    amplitude = 1/(sigma**n*(2*np.pi)**(n/2))
    spatial_term = np.exp(-1*spatial/(2*sigma**2))

    return amplitude*spatial_term

def gaussianFunction(mesh, x_r, a_r, degree=1):
    V = functionspace(mesh, ("CG", degree))
    w = Function(V)
    
    ndim = mesh.geometry.dim
    w.interpolate(lambda x: gaussian(x,x_r,a_r,ndim))

    w = normalize(w)
    return w

def halfGaussianFunction(mesh,x_flame,a_flame,degree=1):
    V = functionspace(mesh, ("CG", degree))
    h = gaussianFunction(mesh, x_flame, a_flame, degree=degree)
    if len(x_flame)==1:
        x_flame = x_flame[0]
    x_tab = V.tabulate_dof_coordinates()
    for i in range(x_tab.shape[0]):
        midpoint = x_tab[i,:]
        z = midpoint[2]
        if z<x_flame[2]:
            value = 0.
        else:
            value = h.x.array[i]
        h.x.petsc_vec.setValueLocal(i, value)
    h = normalize(h)
    return h

def gamma_function(temperature):

    r_gas =  287.1
    if isinstance(temperature, Function):
        V = temperature.function_space
        cp = Function(V)
        cv = Function(V)
        gamma = Function(V)
        cp.x.array[:] = 973.60091+0.1333*temperature.x.array[:]
        cv.x.array[:] = cp.x.array - r_gas
        gamma.x.array[:] = cp.x.array/cv.x.array
        gamma.x.scatter_forward()
    else:    
        cp = 973.60091+0.1333*temperature
        cv= cp - r_gas
        gamma = cp/cv
    return gamma

def sound_speed_variable_gamma(mesh, temperature, degree=1):
    # https://www.engineeringtoolbox.com/air-speed-sound-d_603.html
    # V = functionspace(mesh, ("CG", degree))
    c = Function(temperature.function_space)
    c.name = "soundspeed"
    r_gas = 287.1
    if isinstance(temperature, Function):
        gamma = gamma_function(temperature)
        c.x.array[:] = np.sqrt(gamma.x.array[:]*r_gas*temperature.x.array[:])
    else:
        gamma_ = gamma_function(temperature)
        c.x.array[:] = np.sqrt(gamma_ * r_gas * temperature)
    c.x.scatter_forward()
    return c

def sound_speed(temperature):
    # https://www.engineeringtoolbox.com/air-speed-sound-d_603.html
    c = Function(temperature.function_space)
    c.name = "soundspeed"
    if isinstance(temperature, Function):
        c.x.array[:] =  20.05 * np.sqrt(temperature.x.array)
    else:
        c.x.array[:] =  20.05 * np.sqrt(temperature)
    c.x.scatter_forward()
    return c

def density_step(x, x_f, sigma, rho_d, rho_u):
    return rho_u + (rho_d-rho_u)/2*(1+np.tanh((x-x_f)/(sigma)))

def rho_step(mesh, x_f, a_f, rho_d, rho_u, degree=1):
    V = functionspace(mesh, ("CG", degree))
    rho = Function(V)
    # x = V.tabulate_dof_coordinates()   
    if mesh.geometry.dim == 1 or mesh.geometry.dim == 2:
        x_f = x_f[0]
        x_f = x_f[0]
        rho.interpolate(lambda x: density_step(x[0], x_f, a_f, rho_d, rho_u))
    elif mesh.geometry.dim == 3:
        x_f = x_f[0]
        x_f = x_f[2]
        rho.interpolate(lambda x: density_step(x[2], x_f, a_f, rho_d, rho_u))
    return rho

def rho_ideal(temperature, p_0, r_gas):
    density = Function(temperature.function_space)
    density.x.array[:] =  p_0 /(r_gas * temperature.x.array)
    density.x.scatter_forward()
    return density

def c_step(mesh, x_f, c_u, c_d):
    V = functionspace(mesh, ("CG", 1))
    c = Function(V)
    c.name = "soundspeed"
    x = V.tabulate_dof_coordinates()
    if mesh.geometry.dim == 1:
        x_f = x_f[0]
        x_f = x_f[0]
        axis = 0
    elif mesh.geometry.dim == 2:
        x_f = x_f[0]
        x_f = x_f[0]
        axis = 0
    elif mesh.geometry.dim == 3:
        x_f = x_f[0]
        x_f = x_f[2]
        axis = 2
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[axis]< x_f:
            c.x.petsc_vec.setValueLocal(i, c_u)
        else:
            c.x.petsc_vec.setValueLocal(i, c_d)
    c.x.scatter_forward()
    return c

def c_uniform(mesh, sos, degree=1):
    V = functionspace(mesh, ("CG", degree))
    c = Function(V)
    c.name = "soundspeed"
    c.x.array[:]= sos
    c.x.scatter_forward()
    return c

def temperature(mesh, soundSpeed):
    # https://www.engineeringtoolbox.com/air-speed-sound-d_603.html
    V = functionspace(mesh, ("CG", 1))
    T = Function(V)
    T.name = "temperature"
    r_gas =  287.1
    gamma =  1.4
    if isinstance(soundSpeed, Function):
        T.x.array[:] =  np.square(soundSpeed.x.array) /(r_gas*gamma)
    else:
        T.x.array[:] =  np.square(soundSpeed) / (r_gas*gamma)
    T.x.scatter_forward()
    return T

def temperature_uniform(mesh, temp):
    V = functionspace(mesh, ("CG", 1))
    T = Function(V)
    T.name = "temperature"
    T.x.array[:]=temp
    T.x.scatter_forward()
    return T

def temperature_step(mesh, x_f, T_u, T_d, degree=1):
    V = functionspace(mesh, ("CG", degree))
    T = Function(V)
    T.name = "temperature"
    x = V.tabulate_dof_coordinates()
    if mesh.geometry.dim == 1:
        x_f = x_f[0]
        x_f = x_f[0]
        axis = 0
    elif mesh.geometry.dim == 2:
        x_f = x_f[0]
        x_f = x_f[0]
        axis = 0
    elif mesh.geometry.dim == 3:
        x_f = x_f[0]
        x_f = x_f[2]
        axis = 2
    for i in range(x.shape[0]):
        midpoint = x[i,:]
        if midpoint[axis]< x_f:
            T.x.petsc_vec.setValueLocal(i, T_u)
        else:
            T.x.petsc_vec.setValueLocal(i, T_d)
    return T

def Q_volumetric(mesh, subdomains, Q_total, flame_tag=0, degree=0):
    # volumetric Heat release rate field using subdomains which integrates to 1.
    V = functionspace(mesh, ("DG", degree))
    q = Function(V)
    dx = Measure("dx", subdomain_data=subdomains)
    volume_form = form(Constant(mesh, PETSc.ScalarType(1))*dx(flame_tag))
    V_flame = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
    q_tot = Q_total/V_flame

    cells_flame = subdomains.find(flame_tag)
    dofs_flame = locate_dofs_topological(V, mesh.topology.dim, cells_flame)
    flame_array = unroll_dofmap(dofs_flame, V.dofmap.bs)

    q.x.array[flame_array] = q_tot
    q.x.scatter_forward()

    return q

def Q_multiple(mesh, subdomains, N_sector, degree=0):

    V = functionspace(mesh, ("DG", degree))
    q = Function(V)
    dx = Measure("dx", subdomain_data=subdomains)

    for flame in range(N_sector):
        volume_form = form(Constant(mesh, PETSc.ScalarType(1))*dx(flame))
        V_flame = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        q_tot = 1/V_flame

        cells_flame = subdomains.find(flame)
        dofs_flame = locate_dofs_topological(V, mesh.topology.dim, cells_flame)
        flame_array = unroll_dofmap(dofs_flame, V.dofmap.bs)

        q.x.array[flame_array] = q_tot

    q.x.scatter_forward()

    return q