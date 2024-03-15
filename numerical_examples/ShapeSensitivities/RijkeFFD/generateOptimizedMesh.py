from helmholtz_x.shape_derivatives_utils import FFDCylindrical, deformCylindricalFFD, getMeshdata
from helmholtz_x.io_utils import write_xdmf_mesh, dict_loader
import numpy as np
import gmsh
import sys
import os 

if not os.path.exists('MeshDir'):
    os.makedirs('MeshDir')

path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/MeshDir/Optimized"
mesh_name = "/mesh"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add(__name__)

R = 0.047/2

L_total = 1

gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_total, R, tag=1)
gmsh.model.occ.synchronize()

# Physical tags
surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    gmsh.model.addPhysicalGroup(2, [surface[1]])

gmsh.model.addPhysicalGroup(3, [1], tag=1) # Geometry tag 

lc = 0.005 #0.005 

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(3)

# Retrieve mesh data before it disappears
mesh_data = getMeshdata(gmsh.model)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

### Introducing FFD 
l, m, n = 2, 4, 9
CylindricalLattice = FFDCylindrical(gmsh.model, l, m , n, 3, tag=-1, includeBoundary=True, parametric=False) 
CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFDinitial")

derivatives_normalized = dict_loader(path+"/ShapeDerivatives/normalized")

step = 0.01
for zeta, value_zeta in derivatives_normalized.items():
    for phi, value_phi in derivatives_normalized[zeta].items():
        CylindricalLattice.Pr[l-1][phi][zeta] -= np.round(step*value_phi.imag,10)
    
CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFDupdated")

# Start deformation
gmsh.model.add('deformedModel')
gmsh.model = deformCylindricalFFD(gmsh.model, mesh_data, CylindricalLattice)

# Physical tags - using original tags
for surface in surfaces:
    gmsh.model.addPhysicalGroup(2, [surface[1]])

gmsh.model.addPhysicalGroup(3, [1], tag=1) # Geometry tag 

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

mesh_name = "/mesh"

gmsh.write("{}.msh".format(path+mesh_dir+mesh_name))

write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=3)