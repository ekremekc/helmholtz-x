import gmsh
import os
import sys
import numpy as np
from gmsh_x.mesh_utils import  mirror_mesh_x_axis, fltk_options

geom_dir = "/GeomDir/"
mesh_dir = "/MeshDir"
mesh_name = "/thinAnnulus"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

gmsh.model.add("Geom")
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(path+ geom_dir+'Passive-Fusion.step')
gmsh.model.occ.synchronize()
gmsh.model.occ.importShapes(path+ geom_dir+'Passive-Holes.step')
gmsh.model.occ.synchronize()

gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

total_tag = np.arange(1,5)
hole_tags = np.arange(2,5)

other_tags = list(set(total_tag) - set(hole_tags))

lc = 0.009 #0.0035
gmsh.model.mesh.field.add("Constant", 1)
gmsh.model.mesh.field.setNumbers(1, "VolumesList", hole_tags)
gmsh.model.mesh.field.setNumber(1, "VIn", lc / 2)
gmsh.model.mesh.field.setNumber(1, "VOut", lc)

gmsh.model.mesh.field.setAsBackgroundMesh(1)

gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10) 
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
gmsh.model.mesh.generate(3)

surfaces_old = gmsh.model.getEntities(dim=2)

reflection_vector = [-1,1,1]
mirror_mesh_x_axis(reflection_vector)

gmsh.model.occ.synchronize()
print(gmsh.model.getEntities(dim=2))

total_tag_reflected = total_tag + 1000
total_tag = total_tag.tolist() + total_tag_reflected.tolist()

hole_tags_reflected = hole_tags + 1000
hole_tags = hole_tags.tolist() + hole_tags_reflected.tolist()

other_tags = list(set(total_tag) - set(hole_tags))

gmsh.model.addPhysicalGroup(3, hole_tags, tag=2) # Dilution Holes
gmsh.model.addPhysicalGroup(3, other_tags, tag=1) # Other domains

gmsh.model.occ.synchronize()

surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    
    if surface[1]==30: # Assigning Master and slave tags
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1]) # master boundary
        gmsh.model.addPhysicalGroup(2, [surface[1]+1000], tag=surface[1]+1000) # slave boundary
    elif surface[1]==42: # Assigning Master and slave tags
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1]) # master boundary
        gmsh.model.addPhysicalGroup(2, [surface[1]+1000], tag=surface[1]+1000) # slave boundary
    else:
        gmsh.model.addPhysicalGroup(2, [surface[1],surface[1]+1000], tag=surface[1])
        print(surface[1],": {'Neumann'},")

gmsh.model.occ.synchronize()

gmsh.write("{}.msh".format(path+mesh_dir+mesh_name))

if '-nopopup' not in sys.argv:
    fltk_options()
    gmsh.fltk.run()

gmsh.finalize()

from helmholtz_x.dolfinx_utils import  write_xdmf_mesh

write_xdmf_mesh(path +mesh_dir+mesh_name,dimension=3)