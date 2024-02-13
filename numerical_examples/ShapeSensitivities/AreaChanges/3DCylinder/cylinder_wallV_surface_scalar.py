import gmsh
import sys
import numpy as np

mesh_name = "MeshDir/cylinder"

r_outer_base = 0.25
r_mid = r_outer_base *1.0
z_inlet = 0.
z_outlet = 0.4
lc = 0.01

deg = 2

N_u = 9
N_wall = 3

knots_u= [0, 1/4, 1/2, 3/4, 1]
multiplicities_u = [3, 2, 2, 2, 3]
knots_array_u =[0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]

# For 3 cpt in the v direction
# knots_v= [0, 1]
# multiplicities_v = [3,3]# # [2, 2]# #
# knots_array_v = [0,0,0,1,1,1]

# For 5 cpt in the v direction
# knots_v = [0, 0.33, 0.66, 1]
# multiplicities_v = [3, 1, 1, 3]
# knots_array_v = [0,0,0,0.33,0.66,1,1,1]

knots_v = np.linspace(0,1,N_wall-1)
multiplicities_v = np.ones(len(knots_v))
multiplicities_v[0] = deg+1
multiplicities_v[-1] = deg+1
knots_array_v = []
for i in range(len(knots_v)):
    knots_array_v += [knots_v[i]] * int(multiplicities_v[i])

print(knots_v)
print(multiplicities_v)

angle_jump = 2*np.pi/N_u


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
        r_outer=r_mid
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
                                 weights=weights, knotsU=knots_u, multiplicitiesU=multiplicities_u,
                                 knotsV=knots_v, multiplicitiesV=multiplicities_v)

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


gmsh.model.addPhysicalGroup(1,[2],inlet_tag) # Inlet
gmsh.model.addPhysicalGroup(1,[3],outlet_tag) # Outlet
gmsh.model.addPhysicalGroup(1,[1],lateral_tag) # Lateral

gmsh.model.addPhysicalGroup(2,[2],inlet_tag) # Inlet
gmsh.model.addPhysicalGroup(2,[3],outlet_tag) # Outlet
gmsh.model.addPhysicalGroup(2,[1],lateral_tag) # Lateral

gmsh.model.addPhysicalGroup(3,[1],1) # Pipe

gmsh.model.occ.synchronize()

gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01) # 0.05 working
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.25)

gmsh.model.occ.removeAllDuplicates()

gmsh.model.mesh.generate(3)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write("{}.msh".format(mesh_name))

from helmholtz_x.dolfinx_utils import write_xdmf_mesh, XDMFReader, xdmf_writer
write_xdmf_mesh(mesh_name,dimension=3)

from dolfinx.fem import locate_dofs_topological, VectorFunctionSpace,FunctionSpace, Function

cylinder = XDMFReader(mesh_name)
cylinder.getInfo()
mesh, subdomains, facet_tags = cylinder.getAll()

Q = FunctionSpace(mesh, ("CG", 1))

facet_tags = cylinder.facet_tags
facets = facet_tags.find(lateral_tag)
fdim = mesh.topology.dim-1 # edge facet dimension
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

surface_coordinates = x0[indices]
# print(surface_coordinates)

from geomdl.helpers import basis_function_one

wall_element_tag = lateral_tag

node_tags, coords, t_coords = gmsh.model.mesh.getNodes(2,lateral_tag)

coords = coords.reshape(-1, 3) 
t_coords = t_coords.reshape(-1,2) 

us_dolfinx = t_coords[:,0]
vs_dolfinx = t_coords[:,1]

span_u = 2
span_v = 1 # This is the mid of the lateral surface
V_u = [basis_function_one(deg, knots_array_u, span_u, knot_u) for knot_u in us_dolfinx]
V_v = [basis_function_one(deg, knots_array_v, span_v, knot_v) for knot_v in vs_dolfinx]

V_func = Function(Q)

tolerance = 1e-6
counter = 0

for index, node in zip(indices, surface_coordinates):
    itemindex = np.where(np.isclose(coords, node, atol=tolerance).all(axis=1))[0]
    # itemindex = np.where((coords == node).all(axis=1))[0]
    # print(itemindex)
    if len(itemindex) != 0 : 
        # print(itemindex)
        # print(node, coords[itemindex[0]])
        u_val = V_u[itemindex[0]]
        v_val = V_v[itemindex[0]]
        V_func.x.array[index] = u_val*v_val
        counter +=1

V_func.x.scatter_forward()


print(counter, len(coords), len(surface_coordinates))

xdmf_writer("Functions/V_wall_surface",mesh,V_func)

gmsh.finalize()