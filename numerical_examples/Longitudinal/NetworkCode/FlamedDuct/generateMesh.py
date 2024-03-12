import gmsh
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

meshDirName = 'MeshDir/'
geomDirName = 'geomDir/'
filename = 'tube'
meshname = 'mesh'

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add(filename)
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

# import step file for meshing
gmsh.model.occ.importShapes(os.path.join(path, geomDirName+filename+'.step'))
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

lc = 0.010

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.generate(3)

sur_tags=gmsh.model.getEntities(dim=2)
vol_tags=gmsh.model.getEntities(dim=3)

for i in sur_tags:
    gmsh.model.addPhysicalGroup(2, [i[1]], tag=i[1])

gmsh.model.addPhysicalGroup(3, [1], tag=99)
gmsh.model.occ.synchronize()

gmsh.write("{}.msh".format(dir_path +"/"+meshDirName+meshname))

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

from helmholtz_x.io_utils import write_xdmf_mesh

write_xdmf_mesh(dir_path +"/"+meshDirName+meshname, dimension=3)