import gmsh
import params

def geom_rectangle(file="MeshDir/rijke", fltk=True):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    lc = 3e-3

    L = 1
    h = 0.047/2
    x_f = params.x_f[0]
    a_f = params.a_f
    geom = gmsh.model.geo
    
    # Upstream Domain
    """ p4  _____ p3
           |    |
           |    |
        p1 |____| p2
    
    """
    
    p1 = geom.addPoint(0, -h, 0, lc)
    p2 = geom.addPoint((x_f - a_f), -h, 0, lc)
    p3 = geom.addPoint((x_f - a_f), h, 0, lc)
    p4 = geom.addPoint(0, h, 0, lc)

    l1 = geom.addLine(1, 2)
    l2 = geom.addLine(2, 3)
    l3 = geom.addLine(3, 4)
    l4 = geom.addLine(4, 1)

    ll1 = geom.addCurveLoop([1, 2, 3, 4])
    s1 = geom.addPlaneSurface([1])
    
    # Subdomain (Flame)
    """ p3  _____ p6
           |    |
           |    |
        p2 |____| p5
    
    """
    
    p5 = geom.addPoint((x_f + a_f), -h, 0, lc)
    p6 = geom.addPoint((x_f + a_f), +h, 0, lc)

 
    l5 = geom.addLine(2, 5)
    l6 = geom.addLine(5, 6)
    l7 = geom.addLine(6, 3)


    ll2 = geom.addCurveLoop([5, 6, 7, -2])
    s2 = geom.addPlaneSurface([2])
    
    # Downstream Domain
    """ p6  _____ p8
           |    |
           |    |
        p5 |____| p7
    
    """
    
    p7 = geom.addPoint(L, -h, 0, lc)
    p8 = geom.addPoint(L, +h, 0, lc)

 
    l8  = geom.addLine(5, 7)
    l9  = geom.addLine(7, 8)
    l10 = geom.addLine(8, 6)


    ll3 = geom.addCurveLoop([8 , 9, 10, -6])
    s3 = geom.addPlaneSurface([3])
    
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [1,5,8], 2) # Bottom
    gmsh.model.addPhysicalGroup(1, [9], 3) # Outlet
    gmsh.model.addPhysicalGroup(1, [10,7,3], 4) # Top
    gmsh.model.addPhysicalGroup(1, [4], 1) # Inlet

    #Whole geometry
    gmsh.model.addPhysicalGroup(2, [1,3], 1)
    #Flame Tag
    gmsh.model.addPhysicalGroup(2, [2], 0)

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

    geom_rectangle(file="MeshDir/rijke", fltk=False)
    from helmholtz_x.dolfinx_utils import write_xdmf_mesh
    write_xdmf_mesh("MeshDir/rijke",dimension=2)
    
