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

N_wall = 5
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

gmsh.model.occ.synchronize()

gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

gmsh.model.mesh.generate(3)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()