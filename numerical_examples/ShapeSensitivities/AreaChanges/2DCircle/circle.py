from time import clock_settime_ns
import gmsh
import sys
import numpy as np

gmsh.initialize()

r_outer = 1
z_inlet = 0.
z_outlet = 0.5
lc = 0.05

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

(point_indices, curve_points, ts) = gmsh.model.mesh.getNodes(1,W_full)
# print(ts)
# gmsh.fltk.run()

from geomdl.helpers import basis_function_one

print("knotvector: ", knots_array)
span = 2 # SPAN IS THE INDEX OF THE CONTROL POINT
knot = ts[0]

for knot in ts:
    print("knot: ", knot) 
    span = 2
    print("span: ", span)

    V = basis_function_one(deg, knots_array, span, knot)
    print(V)

import matplotlib.pyplot as plt

for span in range(9):
    V_array = [basis_function_one(deg, knots_array, span, knot) for knot in ts]
    plt.plot(V_array)
plt.savefig("Results/disp_field.pdf")
plt.show()