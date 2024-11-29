from dolfinx.fem import Function, functionspace
from dolfinx.io import XDMFFile, VTKFile
from .solver_utils import info
from mpi4py import MPI
import meshio
import os
import json
import ast

def dict_writer(filename, dictionary, extension = ".txt"):
    """Writes dictionary object into a text file.

    Args:
        filename ('str'): file path
        dictionary ('dict'): dictionary object
        extension (str, optional): file extension. Defaults to ".txt".
    """
    with open(filename+extension, 'w') as file:
        file.write(json.dumps(str(dictionary))) 
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is saved.")

def dict_loader(filename, extension = ".txt"):
    """Loads dictionary into python script

    Args:
        filename ('str'): file path
        extension (str, optional): file extension. Defaults to ".txt".

    Returns:
        dictionary ('dict'): dictionary object
    """
    with open(filename+extension) as f:
        data = json.load(f)
    data = ast.literal_eval(data)
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is loaded.")
    return data

def xdmf_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    el = function.function_space.element 
    if el.basix_element.degree>1:
        el_type = el.basix_element.family
        V = functionspace(mesh, (el_type,1))
        function_interpolation = Function(V)
        function_interpolation.interpolate(function)
    else:
        function_interpolation = function


    with XDMFFile(MPI.COMM_WORLD, name+".xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function_interpolation)

def vtk_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    with VTKFile(MPI.COMM_WORLD, name+".pvd", "w") as vtk:
        vtk.write_mesh(mesh)
        vtk.write_function(function)
        
def create_mesh(mesh, cell_type, prune_z):
    """Subroutine for mesh creation by using meshio library

    Args:
        mesh (meshio._mesh.Mesh): mesh to be converted into Dolfinx mesh
        cell_type ('str'): type of cell (it becomes tetrahedral most of the time)
        prune_z ('bool'): whether consider the 3th dimension's coordinate or not, (it should be False for 2D cases)

    Returns:
        meshio._mesh.Mesh: converted dolfinx mesh
    """

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    
    if cell_type=="tetra":
        print("Number of 3D cells:  {:,}".format(len(cells)))
    elif cell_type=="triangle":
        print("Number of 2D cells (facets):  {:,}".format(len(cells)))

    return out_mesh

def write_xdmf_mesh(name, dimension, write_edge=False):
    """Writes gmsh (.msh) mesh as an .xdmf mesh

    Args:
        name ('str'): filename
        dimension ('int'): Dimension of the mesh (2 or 3)
    """

    if MPI.COMM_WORLD.Get_rank() == 0:
        msh_name = name + ".msh"
        msh = meshio.read(msh_name)

        if dimension == 1:
            prune_z = True
            volume_mesh = create_mesh(msh, "line",prune_z)
            tag_mesh = create_mesh(msh, "vertex",prune_z)

        if dimension == 2:
            prune_z = True
            volume_mesh = create_mesh(msh, "triangle",prune_z)
            tag_mesh = create_mesh(msh, "line",prune_z)

        elif dimension == 3:
            prune_z = False
            volume_mesh = create_mesh(msh, "tetra",prune_z)
            tag_mesh = create_mesh(msh, "triangle",prune_z)
            if write_edge:
                edge_mesh = create_mesh(msh, "line",prune_z)
                xdmf_edge_name = name + "_edgetags.xdmf"
                meshio.write(xdmf_edge_name, edge_mesh)
            
        # Create and save one file for the mesh, and one file for the facets and one file for the edges
        xdmf_name = name + ".xdmf"
        xdmf_tags_name = name + "_tags.xdmf"
        
        meshio.write(xdmf_name, volume_mesh)
        meshio.write(xdmf_tags_name, tag_mesh)

        print(str(dimension)+"D XDMF mesh is generated.")

def load_xdmf_mesh(name):
    """Loads xdmf mesh into python script

    Args:
        name ('str'): Name of the .xdmf file

    Returns:
        tuple: mesh, boundary tags and volume tags of the geometry
    """
    mesh_loader_name = name + ".xdmf"
    tag_loader_name = name + "_tags.xdmf"
    with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    
    if MPI.COMM_WORLD.rank == 0:
        print("XDMF Mesh is loaded.")
    return mesh, cell_tags, facet_tags

class XDMFReader:
    """This class generates geometry objec to load its instances and information about it (number of elements etc.)
    """
    def __init__(self, name):
        self.name = name
        self._mesh = None
        self._cell_tags = None
        self._facet_tags = None
        self._gdim = None
        mesh_loader_name = name + ".xdmf"
        tag_loader_name = name + "_tags.xdmf"
        edgetag_loader_name = name + "_edgetags.xdmf"

        with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
            self._mesh = xdmf.read_mesh(name="Grid")
            self._cell_tags = xdmf.read_meshtags(self.mesh, name="Grid")
        info("\nXDMF Mesh - Cell data loaded.")

        self._mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim)        
        self._mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim)        
        self._mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
            self._facet_tags = xdmf.read_meshtags(self.mesh, name="Grid")    
        info("XDMF Mesh - Facet data loaded.")

        if os.path.exists(edgetag_loader_name):
            self.mesh.topology.create_connectivity(self.mesh.topology.dim-1, self.mesh.topology.dim-2)
            with XDMFFile(MPI.COMM_WORLD, edgetag_loader_name, "r") as xdmf:
                self._edge_tags = xdmf.read_meshtags(self.mesh, name="Grid")
            info("XDMF Mesh - Edge data loaded.")
    
    @property
    def mesh(self):
        return self._mesh
    @property
    def subdomains(self):
        return self._cell_tags
    @property
    def facet_tags(self):
        return self._facet_tags   
    @property
    def edge_tags(self):
        return self._edge_tags 
    @property
    def dimension(self):
        return self._mesh.topology.dim    

    def getAll(self):
        return self.mesh, self.subdomains, self.facet_tags
    
    def getInfo(self):
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        total_num_cells = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        if MPI.COMM_WORLD.Get_rank()==0:
            print("Number of cells:  {:,}".format(total_num_cells))
            print("Number of cores: ", MPI.COMM_WORLD.Get_size(), "\n")
        return total_num_cells