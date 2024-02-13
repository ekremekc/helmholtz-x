import gmsh
import sys
import numpy as np

mesh_name = "MeshDir/circle"

gmsh.initialize()

r_outer = 1
z_inlet = 0.
z_outlet = 0.5
lc = 0.05

gmsh.model.occ.addPoint(r_outer, 0, z_inlet,lc)
gmsh.model.occ.addPoint(r_outer,r_outer,z_inlet,lc)
gmsh.model.occ.addPoint(0,1.01*r_outer,z_inlet,lc)
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

from dolfinx.fem import locate_dofs_topological, VectorFunctionSpace,FunctionSpace, Function
from helmholtz_x.dolfinx_utils import write_xdmf_mesh, XDMFReader, xdmf_writer
write_xdmf_mesh(mesh_name,dimension=2)
circle = XDMFReader(mesh_name)

tag_circle = 1
mesh, subdomains, facet_tags = circle.getAll()

Q = VectorFunctionSpace(mesh, ("CG", 1))

facets = facet_tags.indices[facet_tags.values == tag_circle]
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