import numpy as np
import gmsh
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

meshDirName = 'MeshDir/'
geomDirName = 'geomDir/'
f1name = 'flames'
f2name = 'upstream'
f3name = 'downstream'
meshname = 'mesh'

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

gmsh.model.add("micca")
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(os.path.join(path, geomDirName+f1name+'.step'))
gmsh.model.occ.synchronize()
gmsh.model.occ.importShapes(os.path.join(path, geomDirName+f2name+'.step'))
gmsh.model.occ.synchronize()
gmsh.model.occ.importShapes(os.path.join(path, geomDirName+f3name+'.step'))
gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

flame_vol_tags=gmsh.model.getEntities(dim=3)
for i in range(0, 16):
    gmsh.model.addPhysicalGroup(3, [flame_vol_tags[i][1]], tag=i)

gmsh.model.addPhysicalGroup(3, [flame_vol_tags[-1][1]], tag=16)
gmsh.model.addPhysicalGroup(3, [flame_vol_tags[-2][1]], tag=17)

surfaces = gmsh.model.occ.getEntities(dim=2)

plenum_inlet, plenum_inlet_mark = [], 1
plenum_outer, plenum_outer_mark = [], 2
plenum_inner, plenum_inner_mark = [], 3
plenum_back, plenum_back_mark = [], 4

burner_lateral, burner_lateral_mark = [], 5
burner_back, burner_back_mark = [], 6

injector_lateral, injector_lateral_mark = [], 7

cc_front, cc_front_mark = [], 8
cc_outer, cc_outer_mark = [], 9
cc_inner, cc_inner_mark = [], 10

mass1 = []
mass2 = []
cc_outlet, cc_outlet_mark = [], 11

for surface in surfaces:
    com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
    if np.isclose(com[2], [-0.09]): #PLENUM INLET # TAG1
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], plenum_inlet_mark)
        gmsh.model.setPhysicalName(surface[0], plenum_inlet_mark, "Plenum inlet")

    elif np.isclose(com[2], [-0.055]): #CHAMBER OUTER AND INNER   SURFACE # TAG2 AND TAG 3
        mass1.append(gmsh.model.occ.getMass(surface[0], surface[1]))
        if len(mass1)==2:
            if mass1[0]<mass1[1]:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], plenum_outer_mark)
                gmsh.model.setPhysicalName(surface[0], plenum_outer_mark, "PLENUM OUTER")
                gmsh.model.addPhysicalGroup(other_mark[0], [other_mark[1]], plenum_inner_mark)
                gmsh.model.setPhysicalName(surface[0], plenum_inner_mark, "PLENUM INNER")
            else:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], plenum_inner_mark)
                gmsh.model.setPhysicalName(surface[0], plenum_inner_mark, "PLENUM OUTER")
                gmsh.model.addPhysicalGroup(other_mark[0], [other_mark[1]], plenum_outer_mark)
                gmsh.model.setPhysicalName(surface[0], plenum_outer_mark, "PLENUM INNER")
        other_mark = surface

    elif np.isclose(com[2], [-0.02]): #PLENUM BACK FACE # TAG4
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], plenum_back_mark)
        gmsh.model.setPhysicalName(surface[0], plenum_back_mark, "Plenum Outer")

    elif np.isclose(com[2], [-0.013]): #BURNER LATERAL SURFACE # TAG5
        burner_lateral.append(surface[1])

    elif np.isclose(com[2], [-0.006]): #BURNER BACK SURFACE # TAG6
        burner_back.append(surface[1])

    elif np.isclose(com[2], [-0.003]): #INJECTOR LATERAL  SURFACE # TAG7
        injector_lateral.append(surface[1])

    elif np.allclose(com, [0,0,0]): # CHAMBER FRONT  SURFACE # TAG8
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], cc_front_mark)
        gmsh.model.setPhysicalName(surface[0], cc_front_mark, "Chamber Front")

    elif np.isclose(com[2], [0.1]): #CHAMBER OUTER AND INNER   SURFACE # TAG9 AND TAG 10
        mass2.append(gmsh.model.occ.getMass(surface[0], surface[1]))
        if len(mass2)==2:
            if mass2[0]<mass2[1]:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], cc_outer_mark)
                gmsh.model.setPhysicalName(surface[0], cc_outer_mark, "Chamber OUTER")
                gmsh.model.addPhysicalGroup(other_mark[0], [other_mark[1]], cc_inner_mark)
                gmsh.model.setPhysicalName(surface[0], cc_inner_mark, "Chamber INNER")
            else:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], cc_inner_mark)
                gmsh.model.setPhysicalName(surface[0], cc_inner_mark, "Chamber OUTER")
                gmsh.model.addPhysicalGroup(other_mark[0], [other_mark[1]], cc_outer_mark)
                gmsh.model.setPhysicalName(surface[0], cc_outer_mark, "Chamber INNER")
        other_mark = surface

    elif np.isclose(com[2], [0.2]): #CHAMBER OUTLET   SURFACE # TAG11
        gmsh.model.addPhysicalGroup(surface[0], [surface[1]], cc_outlet_mark)
        gmsh.model.setPhysicalName(surface[0], cc_outlet_mark, "Chamber OUTLET")
        

gmsh.model.addPhysicalGroup(2, burner_lateral, burner_lateral_mark)
gmsh.model.setPhysicalName(2, burner_lateral_mark, "Burner Lateral")

gmsh.model.addPhysicalGroup(2, burner_back, burner_back_mark)
gmsh.model.setPhysicalName(2, burner_back_mark, "Burner Lateral")

gmsh.model.addPhysicalGroup(2, injector_lateral, injector_lateral_mark)
gmsh.model.setPhysicalName(2, injector_lateral_mark, "Injector Lateral")

gmsh.model.occ.synchronize()

# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.008)
gmsh.model.mesh.generate(3)

gmsh.write("{}.msh".format(dir_path +"/"+meshDirName+meshname))

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

from helmholtz_x.io_utils import write_xdmf_mesh

write_xdmf_mesh(dir_path +"/"+meshDirName+meshname, dimension=3)