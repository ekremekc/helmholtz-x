from dolfinx.mesh import meshtags,locate_entities,create_interval, create_unit_square, create_rectangle
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.fem import Function, form
from mpi4py import MPI
import numpy as np
import ufl

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

def OneDimensionalSetup(n_elem, x_f = 0.25, a_f = 0.025, x_end=1.0, tag=0):
    """ This function builds one dimensional setup.
        For boundaries, Tag 1 specifies left end and Tag 2 specifies right end. 
    Args:
        n_elem (int): Number of elements for 1D setup
        x_f (float): Specifies the position of the flame. Default is 0.25
    Returns:
        mesh, subdomains, facet_tags
    """
    mesh = create_interval(MPI.COMM_WORLD, n_elem, [0.0, 1.0])

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

def RectangleSetup(nx, ny, L, h,  x_f = 0.25, a_f = 0.025):
    """ This function builds two dimensional setup.
        For boundaries, Tag 1 specifies left boundary, 
                        Tag 2 specifies right boundary,
                        Tag 3 specifies bottom boundary,
                        Tag 4 specifies top boundary.  

    """
    
    mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L, h])],
                        [nx, ny])

    def fl_subdomain_func(x, x_f=x_f,a_f = a_f, eps=1e-16):
        x = x[0]
        return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
    tdim = mesh.topology.dim
    marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
    fl = 0
    subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

    boundaries = [(1, lambda x: np.isclose(x[0], 0)), # Left
                  (2, lambda x: np.isclose(x[0], L)), # Right
                  (3, lambda x: np.isclose(x[1], 0)), # Bottom
                  (4, lambda x: np.isclose(x[1], h))] # Top

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
    if vector:
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