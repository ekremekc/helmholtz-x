from dolfinx.mesh import meshtags,locate_entities,create_unit_interval, create_unit_square
from dolfinx.fem import Function, FunctionSpace, form, locate_dofs_topological
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.io import XDMFFile
from mpi4py import MPI
import ufl
import numpy as np

def cyl2cart(rho, phi, zeta):
    # cylindrical to Cartesian
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = zeta
    return x, y, z

def cart2cyl(x, y, z):
    # cylindrical to Cartesian
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    zeta = z
    return rho, phi, zeta 

import numba
@numba.njit
def unroll_dofmap(dofs, bs):
    dofs_unrolled = np.zeros(bs*len(dofs), dtype=np.int32)
    for i, dof in enumerate(dofs):
        for b in range(bs):
            dofs_unrolled[i*bs+b] = dof*bs+b

    return dofs_unrolled

def normalize(func):
    """Normalizes dolfinx function such that it integrates to 1 over the domain.

    Args:
        func (dolfinx.fem.function.Function): Dolfinx Function

    Returns:
        dolfinx.fem.function.Function: Normalized dolfinx function
    """

    integral_form = form(func*ufl.dx)
    integral= MPI.COMM_WORLD.allreduce(assemble_scalar(integral_form), op=MPI.SUM)

    func.x.array[:] /= integral
    func.x.scatter_forward()

    return func

def absolute(func):

    abs_temp = abs(func.x.array)
    max_temp = func.function_space.mesh.comm.allreduce(np.amax(abs_temp), op=MPI.MAX)
    temp = abs_temp/max_temp

    p_absoulute = Function(func.function_space) # Required for Parallel runs
    p_absoulute.x.array[:]=temp
    p_absoulute.x.scatter_forward()

    return p_absoulute

def phase(func, deg=True):

    angle_array = np.angle(func.x.array,deg=deg)
    
    p_angle = Function(func.function_space) # Required for Parallel runs
    p_angle.name = "P_angle"
    p_angle.x.array[:] = angle_array
    p_angle.x.scatter_forward()

    return p_angle

def transient(func, omega, t_start, t_end, step, path):

    p_t = Function(func.function_space)

    xdmf = XDMFFile(func.function_space.mesh.comm, path+".xdmf", "w")
    xdmf.write_mesh(func.function_space.mesh)

    time = np.linspace(t_start,t_end,step)

    for t in time:
        # print("t = {0} seconds.".format(t))
        p_t.x.array[:] = func.x.array[:]*np.exp(-1j*omega*t)
        p_t.x.scatter_forward()
        xdmf.write_function(p_t, t)
        
    xdmf.close()

def derivatives_visualizer(filename, shape_derivatives, geometry):
    """ This function generates a .xdmf file which can visualize shape derivative values on boundaries of the geometry.
        Filename should specify the path excluding extension (don't write .xdmf)
        Geometry should be object that is built by XDMF Reader class

    Args:
        filename (str): file name (or path )
        shape_derivatives (dict): Should have the shape derivatives as a dictionary
        geometry (XDMFReader): geometry object
    """

    V = FunctionSpace(geometry.mesh, ("CG",1))
    fdim = geometry.mesh.topology.dim - 1
    U = Function(V)

    # print(shape_derivatives)
    for tag, derivative in shape_derivatives.items():
        # print(tag, derivative)           
        facets = np.array(geometry.facet_tags.indices[geometry.facet_tags.values == tag])
        dofs = locate_dofs_topological(V, fdim, facets)
        U.x.array[dofs] = derivative #first element of boundary
    
    U.x.scatter_forward()

    with XDMFFile(MPI.COMM_WORLD, filename+".xdmf", "w") as xdmf:
        xdmf.write_mesh(geometry.mesh)
        xdmf.write_function(U)

def derivatives_normalizer(shape_derivatives, normalize=True):
    """ This function normalizes the shape derivative values on boundaries of the geometry.

    Args:
        shape_derivatives (dict): Should have shape derivatives as a dictionary
        normalize('bool'): Normalize or not (default is True)
    """
    shape_derivatives_real = shape_derivatives.copy()
    shape_derivatives_imag = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        
        if type(value) == complex: # Axial mode derivatives
            shape_derivatives_real[key] = value.real
            shape_derivatives_imag[key] = value.imag 
            shape_derivatives[key] = value  
        elif type(value) == list: # Azimuthal mode derivatives
            shape_derivatives_real[key] = value[0].real
            shape_derivatives_imag[key] = value[0].imag 
            shape_derivatives[key] = value[0]  # get the first eigenvalue of each list

    if normalize:
        max_key_real = max(shape_derivatives_real, key=lambda y: abs(shape_derivatives_real[y]))
        max_value_real = abs(shape_derivatives_real[max_key_real])
        max_key_imag = max(shape_derivatives_imag, key=lambda y: abs(shape_derivatives_imag[y]))
        max_value_imag = abs(shape_derivatives_imag[max_key_imag])

        normalized_derivatives = shape_derivatives.copy()

        for key, value in shape_derivatives.items():
            normalized_derivatives[key] =  value.real/max_value_real + 1j*value.imag/max_value_imag

        shape_derivatives = normalized_derivatives

    return shape_derivatives

def OneDimensionalSetup(n_elem, x_f = 0.25, a_f = 0.025, tag=0):
    """ This function builds one dimensional setup.
        For boundaries, Tag 1 specifies left end and Tag 2 specifies right end. 
    Args:
        n_elem (int): Number of elements for 1D setup
        x_f (float): Specifies the position of the flame. Default is 0.25
    Returns:
        mesh, subdomains, facet_tags
    """
    
    mesh = create_unit_interval(MPI.COMM_WORLD, n_elem)

    def fl_subdomain_func(x, x_f=x_f, a_f=a_f, eps=1e-16):
        x = x[0]
        return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
    tdim = mesh.topology.dim
    marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
    fl = tag
    subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

    boundaries = [(1, lambda x: np.isclose(x[0], 0)),
                (2, lambda x: np.isclose(x[0], 1))]

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return mesh, subdomains, facet_tags

def SquareSetup(n_elem, x_f = 0.25, a_f = 0.025,):
    """ This function builds two dimensional setup.
        For boundaries, Tag 1 specifies left boundary, 
                        Tag 2 specifies right boundary,
                        Tag 3 specifies bottom boundary,
                        Tag 4 specifies top boundary.  
    Args:
        n_elem (int): Number of elements for 2D setup
        x_f (float): Specifies the position of the flame. Default is 0.25
    Returns:
        mesh, subdomains, facet_tags
    """
    
    mesh = create_unit_square(MPI.COMM_WORLD, n_elem, n_elem)

    def fl_subdomain_func(x, x_f=x_f,a_f = a_f, eps=1e-16):
        x = x[0]
        return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
    tdim = mesh.topology.dim
    marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
    fl = 0
    subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

    boundaries = [(1, lambda x: np.isclose(x[0], 0)), # Left
                  (2, lambda x: np.isclose(x[0], 1)), # Right
                  (3, lambda x: np.isclose(x[1], 0)), # Bottom
                  (4, lambda x: np.isclose(x[1], 1))] # Top

    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tags = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return mesh, subdomains, facet_tags

def distribute_vector_as_chunks(vector):
    vector = MPI.COMM_WORLD.gather(vector, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        vector = [j for i in vector for j in i]
        chunks = [[] for _ in range(MPI.COMM_WORLD.Get_size())]
        for i, chunk in enumerate(vector):
            chunks[i % MPI.COMM_WORLD.Get_size()].append(chunk)
    else:
        vector = None
        chunks = None
    vector = MPI.COMM_WORLD.scatter(chunks, root=0)
    return vector

def broadcast_vector(vector):
    vector = MPI.COMM_WORLD.gather(vector, root=0)
    if vector:
        vector = [j for i in vector for j in i]
    else:
        vector=[]
    vector = MPI.COMM_WORLD.bcast(vector,root=0)
    return vector