import gmsh

from helmholtz_x.dolfinx_utils import write_xdmf_mesh

def make_geometry(points, lcar, elementary_entities, physical_entities, filename, visualization=False):

    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) # Prints log into console
    gmsh.model.add(__name__)

    geom = gmsh.model.geo
    model = gmsh.model


    # points

    geom_points = []
    
    for tag, p in enumerate(points):
        geom_points.append(geom.addPoint(p[0], p[1], 0.0, lcar,tag=tag))

    # lines

    geom_lines = []
    for key in elementary_entities:
        for lists in elementary_entities[key]:
            if type(lists)==int:
                geom_lines.append(geom.addLine(elementary_entities[key][0], elementary_entities[key][1]))
                break
            else:
                geom_lines.append(geom.addLine(lists[0], lists[1]))

    # line loops
    geom_line_loops = geom.addCurveLoop(geom_lines)

    # surfaces
    geom_plane_surfaces = geom.addPlaneSurface([geom_line_loops])


    gmsh.model.geo.synchronize()
    # # Physical Groups
    
    for key in physical_entities:
        if type(physical_entities[key])==int: 
            geom.addPhysicalGroup(1, [physical_entities[key]], key)
        else:
            geom.addPhysicalGroup(1, physical_entities[key], key)

    # # surfaces (subdomains)
    geom.addPhysicalGroup(2, [geom_plane_surfaces],1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    print("Mesh is generated.")
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    
    gmsh.write("{}.msh".format(filename))
    print("Mesh file {} is saved.".format(filename+".msh"))

    write_xdmf_mesh(filename, 2)
    if visualization:
        fltk_options()
        gmsh.fltk.run()

    # gmsh.finalize()

    
    

def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType",2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 2)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 0)


    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    gmsh.option.setNumber("Mesh.Lines", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.VolumeFaces", 0)


if __name__ == '__main__':
    from abstract import *
    
    lcar =0.01

    p0 = [0., + .0235]
    p1 = [0., - .0235]
    p2 = [1., - .0235]
    p3 = [1., + .0235]

    points  = [p0, p1, p2, p3]

    edges = {1:{"points":[points[0], points[1]], "parametrization": False},
             2:{"points":[points[1], points[2]], "parametrization": True, "numctrlpoints":8},
             3:{"points":[points[2], points[3]], "parametrization": False},
             4:{"points":[points[3], points[0]], "parametrization": True, "numctrlpoints":8}}


    geometry = Geometry("ekrem",points,edges,lcar)
    geometry.make_mesh(visualization=True)
