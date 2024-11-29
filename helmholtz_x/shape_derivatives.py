from .petsc4py_utils import conjugate_function
from .eigenvectors import normalize_adjoint
from .dolfinx_utils import unroll_dofmap
from dolfinx.fem import form, locate_dofs_topological,Function, functionspace
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, inner, Measure, div
from math import comb
import numpy as np
import gmsh
import basix

def shapeDerivativesFFD(geometry, lattice, physical_facet_tag, omega_dir, p_dir, p_adj, c, acousticMatrices, FlameMatrix):
    normal = FacetNormal(geometry.mesh)
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)

    p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, acousticMatrices, FlameMatrix)
    p_adj_conj = conjugate_function(p_adj_norm)

    G_neu = div(p_adj_conj * c**2 * grad(p_dir))

    derivatives = {}

    i = lattice.l-1

    for zeta in range(0,lattice.n):

        derivatives[zeta] = {}

        for phi in range(0,lattice.m):

            V_ffd = ffd_displacement_vector(geometry, lattice, physical_facet_tag, i, phi, zeta, deg=1)
            shape_derivative_form = form(inner(V_ffd, normal) * G_neu * ds(physical_facet_tag))
            eig = assemble_scalar(shape_derivative_form)

            derivatives[zeta][phi] = eig

    return derivatives

def ffd_displacement_vector(geometry, FFDLattice, surface_physical_tag, i, j, k,
                            includeBoundary=True, returnParametricCoord=True, tol=1e-6, deg=1):

    mesh, _, facet_tags = geometry.getAll()
    v_cg = basix.ufl.element("Lagrange", mesh.topology.cell_name(), deg, shape=(mesh.geometry.dim,))
    Q = functionspace(mesh, v_cg)

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