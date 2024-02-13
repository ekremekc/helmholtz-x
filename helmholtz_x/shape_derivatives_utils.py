from dolfinx.fem import locate_dofs_topological, VectorFunctionSpace,Function
from geomdl.helpers import basis_function_one
from math import comb
import numpy as np
import gmsh

import numba
@numba.njit
def unroll_dofmap(dofs, bs):
    dofs_unrolled = np.zeros(bs*len(dofs), dtype=np.int32)
    for i, dof in enumerate(dofs):
        for b in range(bs):
            dofs_unrolled[i*bs+b] = dof*bs+b

    return dofs_unrolled

def calculate_displacement_field(gmsh_model, mesh, facet_tags, elementary_facet_tag, physical_facet_tag, span_u, span_v, knots_array_u, knots_array_v, deg):
    
    gmsh.model = gmsh_model

    Q = VectorFunctionSpace(mesh, ("CG", 1))

    facets = facet_tags.find(physical_facet_tag)
    fdim = mesh.topology.dim-1 
    indices = locate_dofs_topological(Q, fdim, facets)
    x0 = mesh.geometry.x

    

    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)

    surface_coordinates = x0[indices]

    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(2, elementary_facet_tag, includeBoundary=True)
    gmsh.model.occ.synchronize()

    norm = gmsh.model.getNormal(elementary_facet_tag,t_coords)
    norm = norm.reshape(-1,3)
    coords = coords.reshape(-1, 3) 
    t_coords = t_coords.reshape(-1,2) 
    us_dolfinx = t_coords[:,0]
    vs_dolfinx = t_coords[:,1]

    if span_u==0 or span_u==8:
        V_u_0 = [basis_function_one(deg, knots_array_u, 0, knot_u) for knot_u in us_dolfinx]
        V_u_8 = [basis_function_one(deg, knots_array_u, 8, knot_u) for knot_u in us_dolfinx]
        V_u = [u_0 + u_8 for u_0, u_8 in zip(V_u_0, V_u_8)]
    else:
        V_u = [basis_function_one(deg, knots_array_u, span_u, knot_u) for knot_u in us_dolfinx]
    
    V_v = [basis_function_one(deg, knots_array_v, span_v, knot_v) for knot_v in vs_dolfinx]
    
    V_func = Function(Q)

    tolerance = 1e-6
    counter = 0

    dofs_Q_array = dofs_Q.reshape(-1,3)

    for dofs_node, node in zip(dofs_Q_array, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tolerance).all(axis=1))[0]
        if len(itemindex) != 0 : 
            u_val = V_u[itemindex[0]]
            v_val = V_v[itemindex[0]]
            xyz_val = norm[itemindex[0]] * u_val*v_val 
            V_func.x.array[dofs_node] = xyz_val
            counter +=1

    V_func.x.scatter_forward()

    print(counter, len(coords), len(surface_coordinates))

    return V_func

def calculate_perpendicular_displacement_field(gmsh_model, mesh, facet_tags, elementary_facet_tag, physical_facet_tag):
    
    gmsh.model = gmsh_model

    Q = VectorFunctionSpace(mesh, ("CG", 1))

    facets = facet_tags.find(physical_facet_tag)
    fdim = mesh.topology.dim-1 
    indices = locate_dofs_topological(Q, fdim, facets)
    x0 = mesh.geometry.x

    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)

    surface_coordinates = x0[indices]

    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(2, elementary_facet_tag, includeBoundary=True)
    gmsh.model.occ.synchronize()

    norm = gmsh.model.getNormal(elementary_facet_tag,t_coords)
    norm = norm.reshape(-1,3)
    coords = coords.reshape(-1, 3) 
    V_func = Function(Q)

    tolerance = 1e-6
    counter = 0

    dofs_Q_array = dofs_Q.reshape(-1,3)

    for dofs_node, node in zip(dofs_Q_array, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tolerance).all(axis=1))[0]
        if len(itemindex) != 0 : 
            xyz_val = norm[itemindex[0]]
            V_func.x.array[dofs_node] = xyz_val
            counter +=1

    V_func.x.scatter_forward()

    print(counter, len(coords), len(surface_coordinates))

    return V_func

def ffd_displacement_vector(geometry, FFDLattice, surface_physical_tag, i, j, k,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):

    mesh, _, facet_tags = geometry.getAll()
    
    Q = VectorFunctionSpace(mesh, ("CG", deg))

    facets = facet_tags.find(surface_physical_tag)
    indices = locate_dofs_topological(Q, mesh.topology.dim-1 , facets)
    surface_coordinates = mesh.geometry.x[indices]
    
    surface_elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(2,surface_physical_tag)
    node_tags, coords, t_coords = gmsh.model.mesh.getNodes(2, int(surface_elementary_tag), includeBoundary=includeBoundary, returnParametricCoord=returnParametricCoord)

    norm = gmsh.model.getNormal(int(surface_elementary_tag),t_coords)

    V_func = Function(Q)
    dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)
    dofs_Q = dofs_Q.reshape(-1,3)

    s,t,u = FFDLattice.calcSTU(coords)
    value = comb(FFDLattice.l-1,i)*np.power(1-s, FFDLattice.l-1-i)*np.power(s,i) * \
            comb(FFDLattice.m-1,j)*np.power(1-t, FFDLattice.m-1-j)*np.power(t,j) * \
            comb(FFDLattice.n-1,k)*np.power(1-u, FFDLattice.n-1-k)*np.power(u,k)

    coords = coords.reshape(-1, 3) 
    norm = norm.reshape(-1,3)

    for dofs_node, node in zip(dofs_Q, surface_coordinates):
        itemindex = np.where(np.isclose(coords, node, atol=tol).all(axis=1))[0]
        if len(itemindex) == 1: 
            V_func.x.array[dofs_node] = value[itemindex]*norm[itemindex][0]
        elif len(itemindex) == 2 :
            V_func.x.array[dofs_node] = value[itemindex][0]*norm[itemindex][0]
        else:
            print(value[itemindex])
   
    V_func.x.scatter_forward()     

    return V_func    

def derivatives_normalize(shape_derivatives):
    """Normalizes shape derivative dictionary

    Args:
        shape_derivatives (dict): complex shape derivatives as a dictionary

    Returns:
        dict: Normalized shape derivatives
    """

    shape_derivatives_real = shape_derivatives.copy()
    shape_derivatives_imag = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        
        shape_derivatives_real[key] = value.real
        shape_derivatives_imag[key] = value.imag 
        shape_derivatives[key] = value  

    max_key_real = max(shape_derivatives_real, key=lambda y: abs(shape_derivatives_real[y]))
    max_value_real = abs(shape_derivatives_real[max_key_real])
    max_key_imag = max(shape_derivatives_imag, key=lambda y: abs(shape_derivatives_imag[y]))
    max_value_imag = abs(shape_derivatives_imag[max_key_imag])

    normalized_derivatives = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        normalized_derivatives[key] =  value.real/max_value_real + 1j*value.imag/max_value_imag

    return normalized_derivatives

def nonaxisymmetric_derivatives_normalize(shape_derivatives):
    """Normalizes nonaxisymmetric 2D shape derivative dictionary

    Args:
        shape_derivatives (dict): complex nonaxisymmetric shape derivatives as a dictionary

    Returns:
        dict: Normalized nonaxisymmetric 2D shape derivatives
    """

    shape_derivatives_real = {}
    shape_derivatives_imag = {}

    for key_v, value in shape_derivatives.items():
        shape_derivatives_real[key_v] = {}
        shape_derivatives_imag[key_v] = {}
        for key_u, value_u in shape_derivatives[key_v].items():
            shape_derivatives_real[key_v][key_u] = np.real(value_u)#.real
            shape_derivatives_imag[key_v][key_u] = np.imag(value_u)
            shape_derivatives[key_v][key_u] = value_u  

    # Finding max indices for real and imag 
    u_real_indices = []
    u_imag_indices = []
    for key_v, value in shape_derivatives_real.items():
        max_key_real = max(shape_derivatives_real[key_v], key=lambda y: abs(shape_derivatives_real[key_v][y]))
        u_real_indices.append(max_key_real)
        max_key_imag = max(shape_derivatives_imag[key_v], key=lambda y: abs(shape_derivatives_imag[key_v][y]))
        u_imag_indices.append(max_key_imag)
        # print(max_key_real)
    
    # Finding max values
    max_value_real = 0
    max_value_imag = 0
    for key_v, value in shape_derivatives_real.items():
        # real part
        for key_u_real in u_real_indices:
            if abs(shape_derivatives_real[key_v][key_u_real]) > max_value_real:
                max_value_real = abs(shape_derivatives_real[key_v][key_u_real])
        # imag part
        for key_u_imag in u_imag_indices:
            if abs(shape_derivatives_imag[key_v][key_u_imag]) > max_value_imag:
                max_value_imag = abs(shape_derivatives_imag[key_v][key_u_imag])
    
    # print(max_value_real)
    # print(max_value_imag)

    normalized_derivatives = {}
    for key_v, value in shape_derivatives.items():
        normalized_derivatives[key_v] = {}
        for key_u, value_u in shape_derivatives[key_v].items():
            normalized_derivatives[key_v][key_u] =  value_u.real/max_value_real + 1j*value_u.imag/max_value_imag

    return normalized_derivatives