import gmsh
import os 
import params
import sys

if not os.path.exists('MeshDir'):
    os.makedirs('MeshDir')

def threeDimRijke(file="MeshDir/mesh", fltk=True):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(__name__)

    d_pipe = params.d_tube
    l_pipe = 1.0

    x_start, y_start, z_start = 0, 0, 0

    # geometry
    gmsh.model.occ.addCylinder(x_start,y_start, z_start, 0, 0, l_pipe, d_pipe/2, tag=1)

    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    lc = 1e-2
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.generate(3)

    surfaces = gmsh.model.occ.getEntities(dim=2)
    for surface in surfaces:
       gmsh.model.addPhysicalGroup(2, [surface[1]])

    gmsh.model.addPhysicalGroup(3, [1], 1)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.write("{}.msh".format(file))

    gmsh.finalize()

if __name__ == '__main__':

    threeDimRijke("MeshDir/mesh")
    from helmholtz_x.io_utils import write_xdmf_mesh
    write_xdmf_mesh("MeshDir/mesh",dimension=3)