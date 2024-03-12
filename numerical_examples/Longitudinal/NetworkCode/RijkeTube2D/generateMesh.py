import gmsh
import params
import sys

def twoDimRijke(file="MeshDir/mesh", fltk=True):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(__name__)

    a_f = params.a_f
    x_f = params.x_f[0]
    x_f = x_f[0]
    d_pipe = params.d_tube
    l_pipe = 1.0

    x_start, y_start = 0, -d_pipe/2

    dx_start = x_f-a_f
    dy_start = d_pipe

    # upstream
    gmsh.model.occ.addRectangle(x_start,y_start,0,dx_start,dy_start, tag=1)
    # flame
    gmsh.model.occ.addRectangle(dx_start,y_start,0,2*a_f,dy_start, tag=2)
    # downstream
    dx_downstream = l_pipe - (x_f+a_f)
    gmsh.model.occ.addRectangle(dx_start+2*a_f,y_start,0,dx_downstream,dy_start, tag=3)

    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    lc = 1e-2
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(2)

    # edges = gmsh.model.occ.getEntities(dim=1)
    # for edge in edges:
    gmsh.model.addPhysicalGroup(1, [4], tag = 1) # inlet
    gmsh.model.addPhysicalGroup(1, [1,5,8], tag = 2) # bottom
    gmsh.model.addPhysicalGroup(1, [3,7,10], tag = 3) # top
    gmsh.model.addPhysicalGroup(1, [9], tag = 4) # outlet

    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.addPhysicalGroup(2, [2], 0)
    gmsh.model.addPhysicalGroup(2, [3], 2)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.write("{}.msh".format(file))

    gmsh.finalize()

if __name__ == '__main__':

    twoDimRijke("MeshDir/mesh")
    from helmholtz_x.io_utils import write_xdmf_mesh
    write_xdmf_mesh("MeshDir/mesh",dimension=2)