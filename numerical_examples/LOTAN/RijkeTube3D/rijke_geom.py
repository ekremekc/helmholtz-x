from math import pi, cos, sin
import gmsh
import os 
if not os.path.exists('MeshDir'):
    os.makedirs('MeshDir')


def geom_pipe(file="MeshDir/rijke", fltk=False):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    add_elementary_entities()
    add_physical_entities()

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write("{}.msh".format(file))

    if fltk:
        fltk_options()
        gmsh.fltk.run()

    # gmsh.finalize()


def add_elementary_entities():

    
    R = 0.047/2
    lc = 5e-3
    
    L_flame_start = 0.225
    L_flame_end   = 0.275
    L_total = 1.
    
    
    geom = gmsh.model.geo

    # CIRCLE 1 ________________________________________________________________________________

    p1 = geom.addPoint(0, 0, 0, lc)

    p2 = geom.addPoint(R * cos(0), R * sin(0), 0, lc)
    p3 = geom.addPoint(R * cos(pi/2), R * sin(pi/2), 0, lc)
    p4 = geom.addPoint(R * cos(pi), R * sin(pi), 0, lc)
    p5 = geom.addPoint(R * cos(3*pi/2), R * sin(3*pi/2), 0, lc)

    l1 = geom.addCircleArc(p2, p1, p3)
    l2 = geom.addCircleArc(p3, p1, p4)
    l3 = geom.addCircleArc(p4, p1, p5)
    l4 = geom.addCircleArc(p5, p1, p2)

    ll1 = geom.addCurveLoop([l1, l2, l3, l4])

    s1 = geom.addPlaneSurface([ll1])

    # CIRCLE 2 ________________________________________________________________________________

    p6 = geom.addPoint(0, 0, L_flame_start, lc)

    p7 = geom.addPoint(R * cos(0), R * sin(0), L_flame_start, lc)
    p8 = geom.addPoint(R * cos(pi/2), R * sin(pi/2), L_flame_start, lc)
    p9 = geom.addPoint(R * cos(pi), R * sin(pi), L_flame_start, lc)
    p10 = geom.addPoint(R * cos(3*pi/2), R * sin(3*pi/2), L_flame_start, lc)

    l5 = geom.addCircleArc(p7, p6, p8)
    l6 = geom.addCircleArc(p8, p6, p9)
    l7 = geom.addCircleArc(p9, p6, p10)
    l8 = geom.addCircleArc(p10, p6, p7)

    ll2 = geom.addCurveLoop([l5, l6, l7, l8])
    
    s2 = geom.addPlaneSurface([ll2])

    # SHELL 1 AND VOLUME 1 ________________________________________________________________________________

    l9 = geom.addLine(p2, p7)
    l10 = geom.addLine(p3, p8)
    l11 = geom.addLine(p4, p9)
    l12 = geom.addLine(p5, p10)

    ll3 = geom.addCurveLoop([l1, l10, -l5, -l9])
    ll4 = geom.addCurveLoop([l2, l11, -l6, -l10])
    ll5 = geom.addCurveLoop([l3, l12, -l7, -l11])
    ll6 = geom.addCurveLoop([l4, l9, -l8, -l12])

    s3 = geom.addSurfaceFilling([ll3])
    s4 = geom.addSurfaceFilling([ll4])
    s5 = geom.addSurfaceFilling([ll5])
    s6 = geom.addSurfaceFilling([ll6])

    sl1 = geom.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    vol1 = geom.addVolume([sl1])

    # CIRCLE 3 ________________________________________________________________________________

    p11 = geom.addPoint(0, 0, L_flame_end, lc)

    p12 = geom.addPoint(R * cos(0), R * sin(0), L_flame_end, lc)
    p13 = geom.addPoint(R * cos(pi/2), R * sin(pi/2), L_flame_end, lc)
    p14 = geom.addPoint(R * cos(pi), R * sin(pi), L_flame_end, lc)
    p15 = geom.addPoint(R * cos(3*pi/2), R * sin(3*pi/2), L_flame_end, lc)

    l13 = geom.addCircleArc(p12, p11, p13)
    l14 = geom.addCircleArc(p13, p11, p14)
    l15 = geom.addCircleArc(p14, p11, p15)
    l16 = geom.addCircleArc(p15, p11, p12)

    ll7 = geom.addCurveLoop([l13, l14, l15, l16])

    s7 = geom.addPlaneSurface([ll7])

    

    # SHELL 2 AND VOLUME 2 (FLAME) ________________________________________________________________________________

    l17 = geom.addLine(p7, p12)
    l18 = geom.addLine(p8, p13)
    l19 = geom.addLine(p9, p14)
    l20 = geom.addLine(p10, p15)

    ll8 = geom.addCurveLoop([l5, l18, -l13, -l17])
    ll9 = geom.addCurveLoop([l6, l19, -l14, -l18])
    ll10 = geom.addCurveLoop([l7, l20, -l15, -l19])
    ll11 = geom.addCurveLoop([l8, l17, -l16, -l20])

    s8  = geom.addSurfaceFilling([ll8])
    s9  = geom.addSurfaceFilling([ll9])
    s10 = geom.addSurfaceFilling([ll10])
    s11 = geom.addSurfaceFilling([ll11])

    sl2 = geom.addSurfaceLoop([s2, s8, s9, s10, s11, s7])
    vol2 = geom.addVolume([sl2])

    # CIRCLE 4 ________________________________________________________________________________

    p16 = geom.addPoint(0, 0, L_total, lc)

    p17 = geom.addPoint(R * cos(0), R * sin(0), L_total, lc)
    p18 = geom.addPoint(R * cos(pi/2), R * sin(pi/2), L_total, lc)
    p19 = geom.addPoint(R * cos(pi), R * sin(pi), L_total, lc)
    p20 = geom.addPoint(R * cos(3*pi/2), R * sin(3*pi/2), L_total, lc)

    l21 = geom.addCircleArc(p17, p16, p18)
    l22 = geom.addCircleArc(p18, p16, p19)
    l23 = geom.addCircleArc(p19, p16, p20)
    l24 = geom.addCircleArc(p20, p16, p17)

    ll12 = geom.addCurveLoop([l21, l22, l23, l24])

    s12 = geom.addPlaneSurface([ll12])

    # SHELL 3 AND VOLUME 3 (DOWNSTREAM) ________________________________________________________________________________

    l25 = geom.addLine(p12, p17)
    l26 = geom.addLine(p13, p18)
    l27 = geom.addLine(p14, p19)
    l28 = geom.addLine(p15, p20)

    ll13 = geom.addCurveLoop([l13, l26, -l21, -l25])
    ll14 = geom.addCurveLoop([l14, l27, -l22, -l26])
    ll15 = geom.addCurveLoop([l15, l28, -l23, -l27])
    ll16 = geom.addCurveLoop([l16, l25, -l24, -l28])

    s13  = geom.addSurfaceFilling([ll13])
    s14  = geom.addSurfaceFilling([ll14])
    s15 = geom.addSurfaceFilling([ll15])
    s16 = geom.addSurfaceFilling([ll16])

    sl3 = geom.addSurfaceLoop([s7, s13, s14, s15, s16, s12])
    vol3 = geom.addVolume([sl3])

def add_physical_entities():

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.addPhysicalGroup(2, [12], tag=2)
    gmsh.model.addPhysicalGroup(2, [3, 4, 5, 6,
                                    8, 9, 10, 11,
                                    13, 14, 15, 16], tag=3)

    gmsh.model.addPhysicalGroup(3, [1,3], tag=999) # Upstream and Downstream
    gmsh.model.addPhysicalGroup(3, [2], tag=0)     # Flame region


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
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    gmsh.option.setNumber("Mesh.Lines", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0) # CHANGE THIS FLAG TO 0 TO SEE LABELS

    gmsh.option.setNumber("Mesh.VolumeEdges", 1)
    gmsh.option.setNumber("Mesh.VolumeFaces", 1)


if __name__ == '__main__':
    from helmholtz_x.dolfinx_utils import write_xdmf_mesh
    geom_pipe(fltk=False)
    write_xdmf_mesh("MeshDir/rijke",dimension=3)
