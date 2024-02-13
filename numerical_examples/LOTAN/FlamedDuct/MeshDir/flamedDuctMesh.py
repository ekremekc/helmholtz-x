import gmsh
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

filename = 'FlamedDuct/tube'

def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 0)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 2)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)

    gmsh.option.setNumber("Mesh.Lines", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0) # CHANGE THIS FLAG TO 0 TO SEE LABELS

    gmsh.option.setNumber("Mesh.VolumeEdges", 2)
    gmsh.option.setNumber("Mesh.VolumeFaces", 2)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add(filename)
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(os.path.join(path, filename+'.step'))
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

# Mesh refinement
lc = 0.015

# # w_field
# gmsh.model.mesh.field.add("Box", 5)
# gmsh.model.mesh.field.setNumber(5, "VIn", lc / 3)
# gmsh.model.mesh.field.setNumber(5, "VOut", lc)
# gmsh.model.mesh.field.setNumber(5, "XMin", -0.06)
# gmsh.model.mesh.field.setNumber(5, "XMax", 0.06)
# gmsh.model.mesh.field.setNumber(5, "YMin", -0.065)
# gmsh.model.mesh.field.setNumber(5, "YMax", 0.065)
# gmsh.model.mesh.field.setNumber(5, "ZMin", 0.29)
# gmsh.model.mesh.field.setNumber(5, "ZMax", 0.41)
# gmsh.model.mesh.field.setNumber(5, "Thickness", 0.1)

# h_field
# gmsh.model.mesh.field.add("Box", 6)
# gmsh.model.mesh.field.setNumber(6, "VIn", lc / 3)
# gmsh.model.mesh.field.setNumber(6, "VOut", lc)
# gmsh.model.mesh.field.setNumber(6, "XMin", -0.06)
# gmsh.model.mesh.field.setNumber(6, "XMax", 0.06)
# gmsh.model.mesh.field.setNumber(6, "YMin", -0.065)
# gmsh.model.mesh.field.setNumber(6, "YMax", 0.065)
# gmsh.model.mesh.field.setNumber(6, "ZMin", 0.48)
# gmsh.model.mesh.field.setNumber(6, "ZMax", 0.58)
# gmsh.model.mesh.field.setNumber(6, "Thickness", 0.1)

# gmsh.model.mesh.field.add("Min", 7)
# gmsh.model.mesh.field.setNumbers(7, "FieldsList", [5,6])

# gmsh.model.mesh.field.setAsBackgroundMesh(6)

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
# gmsh.model.addPhysicalGroup(3, [2, 1002, 3, 1003] , tag=99)

gmsh.model.occ.synchronize()

gmsh.write("{}.msh".format(dir_path +"/"+filename))

if '-nopopup' not in sys.argv:
    fltk_options()
    gmsh.fltk.run()

gmsh.finalize()

from helmholtz_x.dolfinx_utils import  write_xdmf_mesh

write_xdmf_mesh(dir_path +"/"+filename,dimension=3)
