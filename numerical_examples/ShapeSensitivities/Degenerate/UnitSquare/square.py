import gmsh
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

def geom_rectangle(file="MeshDir/square", fltk=True):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    lc = 5e-2

    L = 1
    h = 0.5

    geom = gmsh.model.geo

    p1 = geom.addPoint(0, -h, 0, lc)
    p2 = geom.addPoint(L, -h, 0, lc)
    p3 = geom.addPoint(L, h, 0, lc)
    p4 = geom.addPoint(0, h, 0, lc)

    l1 = geom.addLine(1, 2)
    l2 = geom.addLine(2, 3)
    l3 = geom.addLine(3, 4)
    l4 = geom.addLine(4, 1)

    ll1 = geom.addCurveLoop([1, 2, 3, 4])
    s1 = geom.addPlaneSurface([1])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(0, [1], 0)
    gmsh.model.addPhysicalGroup(0, list(range(2, 5)), 1)

    gmsh.model.addPhysicalGroup(1, [1], 1)
    gmsh.model.addPhysicalGroup(1, [2], 2)
    gmsh.model.addPhysicalGroup(1, [3], 3)
    gmsh.model.addPhysicalGroup(1, [4], 4)


    gmsh.model.addPhysicalGroup(2, [1], 0)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write("{}.msh".format(file))

    if fltk:
        fltk_options()
        gmsh.fltk.run()

    gmsh.finalize()


def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 1)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 0)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 0)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)

    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.VolumeFaces", 0)


if __name__ == '__main__':

    geom_rectangle(fltk=True)
    from helmholtz_x.geometry_pkgx.xdmf_utils import  write_xdmf_mesh

    write_xdmf_mesh(dir_path +"/MeshDir/square",dimension=2)
