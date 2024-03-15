from helmholtz_x.io_utils import write_xdmf_mesh, XDMFReader, xdmf_writer
from helmholtz_x.shape_derivatives_utils import FFDCylindrical, getMeshdata
import gmsh
import sys
import os 

mesh_dir = "/InputFunctions/V/MeshDir"

if not os.path.exists('MeshDir'):
    os.makedirs('MeshDir')

path = os.path.dirname(os.path.abspath(__file__))
mesh_name = "/mesh"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add(__name__)

R = 0.047/2
L_total = 1.0

gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_total, R, tag=1)
gmsh.model.occ.synchronize()

# Physical tags
surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    gmsh.model.addPhysicalGroup(2, [surface[1]])

gmsh.model.addPhysicalGroup(3, [1], tag=1)

lc = 0.005

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
l, m, n = 2, 4, 9
CylindricalLattice = FFDCylindrical(gmsh.model, l, m , n, 3, tag=-1, includeBoundary=True, parametric=False) 
CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFD")

# Displacement Field

cylinder = XDMFReader(path+mesh_dir+mesh_name)
mesh, subdomains, facet_tags = cylinder.getAll()
cylinder.getInfo()

# Specify the control point in FFD using i,j and k indices
i, j, k = 1, 1, 4
physical_facet_tag = 1
elementary_facet_tag = 1

from helmholtz_x.shape_derivatives import ffd_displacement_vector
              
V_ffd = ffd_displacement_vector(cylinder, CylindricalLattice, physical_facet_tag, i, j, k)
xdmf_writer(path+"/InputFunctions/V/V_"+str(i)+str(j)+str(k), mesh, V_ffd)