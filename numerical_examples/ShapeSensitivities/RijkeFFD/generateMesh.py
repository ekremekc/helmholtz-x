from helmholtz_x.shape_derivatives_utils import FFDCylindrical, deformCylindricalFFD, getMeshdata
from helmholtz_x.io_utils import write_xdmf_mesh
import gmsh
import sys
import os 

path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/MeshDir/Original"
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

lc = 0.007 #0.005 

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(3)

# Retrieve mesh data before it disappears
mesh_data = getMeshdata(gmsh.model)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write("{}.msh".format(path+mesh_dir+mesh_name))

write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=3)

### Introducing FFD 
l, m, n = 2, 4, 5
CylindricalLattice = FFDCylindrical(gmsh.model, l, m , n, 3, tag=-1, includeBoundary=True, parametric=False) 
CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFDinitial")

# Example Radial Deformation
for i in range(m):
    # CylindricalLattice.Pr[1, i, 0] -= 0.02
    CylindricalLattice.Pr[1, i, 1] += 0.02
    CylindricalLattice.Pr[1, i, 2] -= 0.02
    # CylindricalLattice.Pr[1, i, 3] += 0.02
    # CylindricalLattice.Pr[1, i, 4] -= 0.02

CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFDchanged")

# Start deformation
gmsh.model.add('deformedModel')

gmsh.model = deformCylindricalFFD(gmsh.model, mesh_data, CylindricalLattice)

# Physical tags - using original tags
for surface in surfaces:
    gmsh.model.addPhysicalGroup(2, [surface[1]])

gmsh.model.addPhysicalGroup(3, [1], tag=1) # Geometry tag 

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

deformed_mesh_name = "/meshDeformedExample"

gmsh.write("{}.msh".format(path+mesh_dir+deformed_mesh_name))

write_xdmf_mesh(path+mesh_dir+deformed_mesh_name,dimension=3)