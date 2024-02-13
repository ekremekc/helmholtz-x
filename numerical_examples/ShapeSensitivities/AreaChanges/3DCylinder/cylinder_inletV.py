import gmsh
import sys
import numpy as np

mesh_name = "MeshDir/cylinder"

r_outer_base = 0.25
z_inlet = 0.
z_outlet = 0.5
lc = 0.01

deg = 2

N_u = 9

knots_u= [0, 1/4, 1/2, 3/4, 1]
knots_array =[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]

knots_v= [0, 1]
multiplicities_u = [3, 2, 2, 2, 3]
multiplicities_v = [2, 2]

angle_jump = 2*np.pi/N_u

N_wall = 2
weights_surface = [1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1]
weights = np.tile(weights_surface,N_wall)
z_wall = np.linspace(z_inlet,z_outlet,N_wall)

# Mesh generation
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# Lateral Surface Mesh
angle = 0

for z in z_wall:
    if z==z_outlet/2:
        r_outer=1.5*r_outer_base
    else:
        r_outer = r_outer_base
    gmsh.model.occ.addPoint(r_outer, 0, z,lc)
    gmsh.model.occ.addPoint(r_outer,r_outer,z,lc)
    gmsh.model.occ.addPoint(0,r_outer,z,lc)
    gmsh.model.occ.addPoint(-r_outer,r_outer,z,lc)
    gmsh.model.occ.addPoint(-r_outer,0,z,lc)
    gmsh.model.occ.addPoint(-r_outer,-r_outer,z,lc)
    gmsh.model.occ.addPoint(0,-r_outer,z,lc)
    gmsh.model.occ.addPoint(r_outer,-r_outer,z,lc)
    gmsh.model.occ.addPoint(r_outer, 0, z,lc)

gmsh.model.occ.addBSplineSurface(range(1,(N_u)*N_wall+1), N_u, degreeU=2, degreeV=2,
                                 weights=weights, knotsU=knots_u,
                                 multiplicitiesU=multiplicities_u)

# gmsh.model.occ.removeAllDuplicates()

# inlet surface
W_inlet = gmsh.model.occ.addWire([2])
gmsh.model.occ.addSurfaceFilling(W_inlet)

# outlet surface
W_outlet = gmsh.model.occ.addWire([3])
gmsh.model.occ.addSurfaceFilling(W_outlet)

# Volume of the pipe
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

surfaces = gmsh.model.occ.getEntities(2)
surface_tags = [surface[1] for surface in surfaces]
# print(surface_tags)
gmsh.model.occ.addSurfaceLoop(surface_tags, 1)
gmsh.model.occ.addVolume([1], 1)

gmsh.model.occ.synchronize()

# # Add physical tags

lateral_tag = 1
inlet_tag = 2
outlet_tag = 3


gmsh.model.addPhysicalGroup(1,[1],lateral_tag) # Lateral
gmsh.model.addPhysicalGroup(1,[2],inlet_tag) # Inlet
gmsh.model.addPhysicalGroup(1,[3],outlet_tag) # Outlet

gmsh.model.addPhysicalGroup(2,[1],lateral_tag) # Lateral
gmsh.model.addPhysicalGroup(2,[2],inlet_tag) # Inlet
gmsh.model.addPhysicalGroup(2,[3],outlet_tag) # Outlet

gmsh.model.addPhysicalGroup(3,[1],1) # Pipe

# gmsh.option.setNumber('Mesh.MeshSizeMin', 0.15)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

gmsh.model.mesh.generate(3)

gmsh.model.occ.removeAllDuplicates()

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write("{}.msh".format(mesh_name))

from helmholtz_x.dolfinx_utils import write_xdmf_mesh, XDMFReader, xdmf_writer
write_xdmf_mesh(mesh_name,dimension=3)

from dolfinx.fem import locate_dofs_topological, VectorFunctionSpace,FunctionSpace, Function
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices

circle = XDMFReader(mesh_name)
circle.getInfo()
mesh, subdomains, facet_tags = circle.getAll()

Q = VectorFunctionSpace(mesh, ("CG", 1))

edge_tags = circle.edge_tags
edge_facets = edge_tags.find(inlet_tag)
fdim = mesh.topology.dim-2 # edge facet dimension
indices = locate_dofs_topological(Q, fdim, edge_facets)
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
# print(edge_coordinates)
from geomdl.helpers import basis_function_one

span = 2
inlet_element_tag = 2
ts_dolfinx = [gmsh.model.getParametrization(1,inlet_element_tag, point) for point in edge_coordinates]
ts_dolfinx = np.asarray(ts_dolfinx)
print(ts_dolfinx)

edge_coordinates = edge_coordinates[:,:2]
xs = edge_coordinates[:,0]
ys = edge_coordinates[:,1]
angle = np.arctan2(ys, xs)

V_dolfinx = [basis_function_one(deg, knots_array, span, knot) for knot in ts_dolfinx]
V_dolfinx = [v if v==0 else v[0] for v in V_dolfinx ]

V_x = V_dolfinx * np.cos(angle)
V_y = V_dolfinx * np.sin(angle)
V_z = np.zeros(len(V_dolfinx))

V_full = np.column_stack((V_x, V_y, V_z)).flatten()

V_func = Function(Q)
V_func.x.array[dofs_Q] = V_full
V_func.x.scatter_forward()

xdmf_writer("Functions/V_inlet",mesh,V_func)

gmsh.finalize()
