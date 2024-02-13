import gmsh
import sys
import numpy as np

mesh_name = "MeshDir/circle"

gmsh.initialize()

r_outer = 1
z_inlet = 0.
z_outlet = 0.5
lc = 0.1

gmsh.model.occ.addPoint(r_outer, 0, z_inlet,lc)
gmsh.model.occ.addPoint(r_outer,r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(0,r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(-r_outer,r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(-r_outer,0,z_inlet,lc)
gmsh.model.occ.addPoint(-r_outer,-r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(0,-r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(r_outer,-r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(r_outer, 0, z_inlet,lc)

dim = 2

knots = [0, 1/4, 1/2, 3/4, 1]
multiplicities = [3, 2, 2, 2, 3]

weights = [1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1]
knots_array =[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]

deg = 2


C_full = gmsh.model.occ.addBSpline(range(1,10), degree=deg, knots=knots, multiplicities=multiplicities, weights=weights)
W_full = gmsh.model.occ.addWire([C_full])
gmsh.model.occ.addSurfaceFilling(W_full)


gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(1,[1],1) # circle edge
gmsh.model.addPhysicalGroup(2,[1],1) # circle surface

gmsh.model.mesh.generate(dim)

# gmsh.fltk.run()
gmsh.write("{}.msh".format(mesh_name))
# gmsh.finalize()

# import mesh

from dolfinx.fem import locate_dofs_topological, VectorFunctionSpace,FunctionSpace, Function, form
from helmholtz_x.dolfinx_utils import write_xdmf_mesh, XDMFReader, xdmf_writer
write_xdmf_mesh(mesh_name,dimension=2)
circle = XDMFReader(mesh_name)

tag_circle = 1
mesh, subdomains, facet_tags = circle.getAll()

Q = VectorFunctionSpace(mesh, ("CG", 1))

facets = facet_tags.find(tag_circle)
fdim = mesh.topology.dim-1 # facet dimension
indices = locate_dofs_topological(Q, fdim, facets)
x0 = mesh.geometry.x

import numba
@numba.njit
def unroll_dofmap(dofs, bs):
    dofs_unrolled = np.zeros(bs*len(dofs), dtype=np.int32)
    for i, dof in enumerate(dofs):
        for b in range(bs):
            dofs_unrolled[i*bs+b] = dof*bs+b

    return dofs_unrolled

dofs_Q = unroll_dofmap(indices, Q.dofmap.bs)

edge_coordinates = x0[indices]

from geomdl.helpers import basis_function_one

span = 2
ts_dolfinx = [gmsh.model.getParametrization(1,W_full, point) for point in edge_coordinates]
ts_dolfinx = np.asarray(ts_dolfinx)

edge_coordinates = edge_coordinates[:,:2]
xs = edge_coordinates[:,0]
ys = edge_coordinates[:,1]
angle = np.arctan2(ys, xs)

V_dolfinx = [basis_function_one(deg, knots_array, span, knot) for knot in ts_dolfinx]
V_dolfinx = [v if v==0 else v[0] for v in V_dolfinx ]

V_x = V_dolfinx * np.cos(angle)
V_y = V_dolfinx * np.sin(angle)

V_full = np.column_stack((V_x, V_y)).flatten()

V_func = Function(Q)
V_func.x.array[dofs_Q] = V_full
V_func.x.scatter_forward()

xdmf_writer("Functions/V",mesh,V_func)

# Mode calculation

from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigensolvers import pep_solver
from helmholtz_x.eigenvectors import normalize_eigenvector, normalize_adjoint
from helmholtz_x.parameters_utils import c_uniform

c_value = 343
c = c_uniform(mesh, c_value)

degree = 2 

boundary_conditions = {1:{'Dirichlet'}}

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

target = 100 * 2 * np.pi 

E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=3, print_results= False)
omega_dir, p_dir = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')
xdmf_writer("Results/p_dir", mesh, p_dir)

E_adj = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=3, print_results= False)
omega_adj, p_adj = normalize_eigenvector(mesh, E_adj, 0, degree=degree, which='left')
xdmf_writer("Results/p_adj", mesh, p_adj)

# Shape derivative calculation

from helmholtz_x.shape_derivatives_x import _shape_gradient_Dirichlet
from helmholtz_x.petsc4py_utils import conjugate_function
from dolfinx.fem import assemble_scalar
from ufl import inner, Measure, FacetNormal

p_adj_normalize = normalize_adjoint(omega_dir, p_dir, p_adj, matrices)

p_adj_conj = conjugate_function(p_adj)
G_dir = _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
n = FacetNormal(mesh)
ds = Measure('ds', domain=mesh, subdomain_data=facet_tags)
omega_prime_real =  assemble_scalar( form(inner(V_func, n) * G_dir * ds(tag_circle)) )

print("Shape Derivative: ", omega_prime_real)