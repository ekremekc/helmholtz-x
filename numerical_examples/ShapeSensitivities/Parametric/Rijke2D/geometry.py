import bsplines
from gmsh_utils import make_geometry
from helmholtz_x.dolfinx_utils import load_xdmf_mesh
import ufl
from dolfinx.fem import FunctionSpace, Function, VectorFunctionSpace, locate_dofs_topological
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

class Geometry:

    def __init__(self, filename, points, edges, lcar):

        self.filename = filename
        self.points = points
        self.edges = edges


        self.bspl_pts = dict()
        self.bspline = dict()
        self.ctrl_pts = dict()
        
        self.elementary_entities = dict()
        self.physical_entities = dict()
        self.lcar = lcar
        self.last_index = None
        
        self.mesh = None
        self.subdomains = None
        self.facet_tags = None
    
    def parametrize_boundary(self, key):
        
        def g(p0, p1, n):
            # n is the number of intervals
            pts = []
            for i in range(n + 1):
                x = p0[0] + i*(p1[0] - p0[0])/n
                y = p0[1] + i*(p1[1] - p0[1])/n
                pts.append([x, y])
            return pts

        points_of_line = self.edges[key]["points"]
        self.ctrl_pts[key] = g(points_of_line[0],points_of_line[1] , self.edges[key]["numctrlpoints"])
        self.bspline[key] = bsplines.BSpline(self.ctrl_pts[key], num_of_pts=self.edges[key]["numctrlpoints"]*10)
        self.bspl_pts[key] = self.bspline[key].pts[1:-1] # Exclude end points 
        

    def make_elementary_entities(self, key):

        self.elementary_entities[key] = []

        points_of_line = self.edges[key]["points"]

        first_index = self.points.index(points_of_line[0])

        for newpoint in self.bspl_pts[key]:
            self.elementary_entities[key].append([first_index,len(self.points)])
            first_index = len(self.points)
            self.points.append(newpoint)

        last_index = self.points.index(points_of_line[1])
        self.elementary_entities[key].append([len(self.points)-1, last_index]) 
        self.points.append(self.edges[key]["points"][1])#Add first point again to complete loop
    
    def make_physical_entities(self, key):
        
        if key ==1:
            self.last_index = 1
            self.physical_entities[key] = 1
        else:
            if len(self.elementary_entities[key])==2:
                self.physical_entities[key] = self.last_index+1
                self.last_index +=1
            else: 
                local_last_index = self.last_index+len(self.elementary_entities[key])
                
                self.physical_entities[key] = list(range(self.last_index+1,local_last_index+1))
                self.last_index = local_last_index
    
    
    def make_mesh(self, visualization = False):

        for key in self.edges:

            if self.edges[key]["parametrization"]:
                self.parametrize_boundary(key)
                self.make_elementary_entities(key)
            else:
                
                local_ll =[]

                for point in self.edges[key]["points"]:
                    
                    local_ll.append(self.points.index(point))
        
                self.elementary_entities[key]=local_ll   
            self.make_physical_entities(key)

        if MPI.COMM_WORLD.rank == 0:
            make_geometry(self.points, self.lcar, self.elementary_entities, self.physical_entities, self.filename , visualization=visualization)
        
        # self.mesh, self.subdomains, self.facet_tags = read_from_msh(self.filename+".msh", cell_data=True, facet_data=True, gdim=2)
        self.mesh, self.subdomains, self.facet_tags = load_xdmf_mesh(self.filename)


    def get_displacement_field(self, tag, c_id):
        """
        computes displacement field's components on x and y direction'        
        
        Parameters
        ----------
        i : Tag of the physical curve (edge in this case)
        c_id : control point index in the BSpline 
        Returns
        -------
        V_x : Displacement field along x-direction
        V_y : Displacement field along y-direction
        
        """
        
        
        Q = VectorFunctionSpace(self.mesh, ("CG", 1))
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = cell_map.size_local + cell_map.num_ghosts

        vertex_map = self.mesh.topology.index_map(0)
        num_vertices = vertex_map.size_local + vertex_map.num_ghosts

        dofmap_x = self.mesh.geometry.dofmap
        dofmap = Q.dofmap
        block_size = dofmap.index_map_bs
        vertex_to_dof_map = np.empty(num_vertices * block_size, dtype=np.int32)
        for c in range(num_cells):
            dofs = dofmap.cell_dofs(c)
            dofs_x = dofmap_x[c]
            for (dof, dof_x) in zip(dofs, dofs_x):
                for j in range(block_size):
                    vertex_to_dof_map[dof_x*block_size + j] = dof * block_size + j
        # print(vertex_to_dof_map)

        facets = self.facet_tags.indices[self.facet_tags.values == tag]
        fdim = self.mesh.topology.dim-1 # facet dimension
        indices = locate_dofs_topological(Q, fdim, facets)
        x0 = self.mesh.geometry.x

        edge_coordinates = x0[indices]
        
        if self.mesh.topology.dim==2:
            edge_coordinates = np.delete(edge_coordinates, 2, 1)
        
        vertex_to_dof_map = vertex_to_dof_map.reshape((-1, self.mesh.topology.dim))

        dofs_Q = vertex_to_dof_map[indices]
        dofs_Q = dofs_Q.reshape(-1)
        
        b = np.zeros((len(edge_coordinates), 2))
        c = np.zeros((len(edge_coordinates), 2))

        for (k, p) in enumerate(edge_coordinates):
            b[k, 0] = self.bspline[tag].get_displacement_from_point(c_id, p)
            c[k, 1] = self.bspline[tag].get_displacement_from_point(c_id, p)

        d = b.reshape(-1)
        e = c.reshape(-1)

        V_x = Function(Q)
        V_x.vector[dofs_Q] = d
        V_x.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        V_y = Function(Q)
        V_y.vector[dofs_Q] = e
        V_y.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        

        return V_x, V_y

    def get_curvature_field(self, i):
        """
        Parameters
        ----------
        i : Tag of the physical curve (edge in this case)
        Returns
        -------
        curvature : curvature field of the edge(s) 
                    based on the corresponding boundary 
                    i's points    
        """
        # self.indices_of_boundary_points(i)
        # V = FunctionSpace(self.mesh, "CG", 1)
        # vertex_to_dof_V = vertex_to_dof_map(V)
        # dofs_V = vertex_to_dof_V[self.indices]

        Q = FunctionSpace(self.mesh, ("CG", 1))

        facets = np.array(self.facet_tags.indices[self.facet_tags.values == i])
        fdim = self.mesh.topology.dim-1 # facet dimension
        dofs = locate_dofs_topological(Q, fdim, facets)

        x0 = Q.tabulate_dof_coordinates()
        edge_coordinates = x0[dofs]
        if self.mesh.topology.dim==2:
            edge_coordinates = np.delete(edge_coordinates, 2, 1)
        
        array = np.zeros((len(edge_coordinates)))

        for (k, boundary_point) in enumerate(edge_coordinates):
            # print("WORKING", self.bspline[i].get_curvature_from_point(boundary_point))
            array[k] = self.bspline[i].get_curvature_from_point(boundary_point)

        
        curvature = Function(Q)
        curvature.vector[dofs] = array
        # curvature = Function(V)
        # curvature.vector()[dofs_V] = a

        return curvature

if __name__ == '__main__':

    lcar =0.2

    p0 = [0., + .0235]
    p1 = [0., - .0235]
    p2 = [1., - .0235]
    p3 = [1., + .0235]

    # p0 = [0., + 0.5]
    # p1 = [0., - .5]
    # p2 = [1., - .5]
    # p3 = [1., + .5]

    points  = [p0, p1, p2, p3]

    edges = {1:{"points":[points[0], points[1]], "parametrization": False},
             2:{"points":[points[1], points[2]], "parametrization": True, "numctrlpoints":10},
             3:{"points":[points[2], points[3]], "parametrization": False},
             4:{"points":[points[3], points[0]], "parametrization": True, "numctrlpoints":10}}


    geometry = Geometry("ekrem",points,edges,lcar)
    geometry.make_mesh(False)
    print(geometry.ctrl_pts)
    import matplotlib.pyplot as plt
    Vx, Vy = geometry.get_displacement_field(4,5)
    from dolfinx.io import XDMFFile
    with XDMFFile(MPI.COMM_WORLD, "Vy.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(geometry.mesh)
        xdmf.write_function(Vy)

    

