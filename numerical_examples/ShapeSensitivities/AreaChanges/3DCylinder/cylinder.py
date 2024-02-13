import gmsh
import sys
import numpy as np

gmsh.initialize()

r_outer_base = 0.25
z_inlet = 0.
z_outlet = 0.5
lc = 0.01

deg = 2

N_u = 9

knots_u= [0, 1/4, 1/2, 3/4, 1]
knots_v= [0, 1]
multiplicities_u = [3, 2, 2, 2, 3]
multiplicities_v = [2, 2]

angle_jump = 2*np.pi/N_u

N_wall = 3
weights_surface = [1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1, 2**0.5/2, 1]
weights = np.tile(weights_surface,N_wall)
z_wall = np.linspace(z_inlet,z_outlet,N_wall)

# inlet
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
tolerance = 0.01
inlet_points = gmsh.model.occ.getEntitiesInBoundingBox(-r_outer-tolerance,-r_outer-tolerance,z_inlet-tolerance,
                                                        r_outer+tolerance,r_outer+tolerance,z_inlet+tolerance,
                                                        0)
inlet_points_tags = [point[1] for point in inlet_points]
print(inlet_points_tags)

C_inlet = gmsh.model.occ.addBSpline(inlet_points_tags[:9], degree=deg, knots=knots_u, multiplicities=multiplicities_u, weights=weights_surface)
W_inlet = gmsh.model.occ.addWire([C_inlet])
gmsh.model.occ.addSurfaceFilling(W_inlet)


# # outlet surface
outlet_points = gmsh.model.occ.getEntitiesInBoundingBox(-r_outer-tolerance,-r_outer-tolerance,z_outlet-tolerance,
                                                        r_outer+tolerance,r_outer+tolerance,z_outlet+tolerance,
                                                        0)
outlet_points_tags = [point[1] for point in outlet_points]
print(outlet_points_tags)

C_outlet = gmsh.model.occ.addBSpline(outlet_points_tags[:9], degree=deg, knots=knots_u, multiplicities=multiplicities_u, weights=weights_surface)
W_outlet = gmsh.model.occ.addWire([C_outlet])
gmsh.model.occ.addSurfaceFilling(W_outlet)

# Volume of the pipe
gmsh.model.occ.removeAllDuplicates()

gmsh.model.occ.synchronize()

surfaces = gmsh.model.occ.getEntities(2)
surface_tags = [surface[1] for surface in surfaces]
print(surface_tags)
# gmsh.model.occ.remove([(2, 2)])
# gmsh.model.occ.remove([(2, 4)])
gmsh.model.occ.addSurfaceLoop(surface_tags, 1)
gmsh.model.occ.addVolume([1], 1)

gmsh.model.occ.synchronize()

# # Add physical tags

gmsh.model.addPhysicalGroup(2,[1],1) # Lateral
gmsh.model.addPhysicalGroup(2,[2],2) # Inlet
gmsh.model.addPhysicalGroup(2,[3],3) # Outlet
gmsh.model.addPhysicalGroup(1,[1],1) # Pipe

gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

gmsh.model.mesh.generate(3)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()