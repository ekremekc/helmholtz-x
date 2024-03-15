from .dolfinx_utils import cart2cyl, cyl2cart
from pyevtk.hl import pointsToVTK
from math import comb
import numpy as np
import gmsh

def derivatives_normalize(shape_derivatives):
    """Normalizes shape derivative dictionary

    Args:
        shape_derivatives (dict): complex shape derivatives as a dictionary

    Returns:
        dict: Normalized shape derivatives
    """

    shape_derivatives_real = shape_derivatives.copy()
    shape_derivatives_imag = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        
        shape_derivatives_real[key] = value.real
        shape_derivatives_imag[key] = value.imag 
        shape_derivatives[key] = value  

    max_key_real = max(shape_derivatives_real, key=lambda y: abs(shape_derivatives_real[y]))
    max_value_real = abs(shape_derivatives_real[max_key_real])
    max_key_imag = max(shape_derivatives_imag, key=lambda y: abs(shape_derivatives_imag[y]))
    max_value_imag = abs(shape_derivatives_imag[max_key_imag])

    normalized_derivatives = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        normalized_derivatives[key] =  value.real/max_value_real + 1j*value.imag/max_value_imag

    return normalized_derivatives

def nonaxisymmetric_derivatives_normalize(shape_derivatives):
    """Normalizes nonaxisymmetric 2D shape derivative dictionary

    Args:
        shape_derivatives (dict): complex nonaxisymmetric shape derivatives as a dictionary

    Returns:
        dict: Normalized nonaxisymmetric 2D shape derivatives
    """

    shape_derivatives_real = {}
    shape_derivatives_imag = {}

    for key_v, value in shape_derivatives.items():
        shape_derivatives_real[key_v] = {}
        shape_derivatives_imag[key_v] = {}
        for key_u, value_u in shape_derivatives[key_v].items():
            shape_derivatives_real[key_v][key_u] = np.real(value_u)#.real
            shape_derivatives_imag[key_v][key_u] = np.imag(value_u)
            shape_derivatives[key_v][key_u] = value_u  

    # Finding max indices for real and imag 
    u_real_indices = []
    u_imag_indices = []
    for key_v, value in shape_derivatives_real.items():
        max_key_real = max(shape_derivatives_real[key_v], key=lambda y: abs(shape_derivatives_real[key_v][y]))
        u_real_indices.append(max_key_real)
        max_key_imag = max(shape_derivatives_imag[key_v], key=lambda y: abs(shape_derivatives_imag[key_v][y]))
        u_imag_indices.append(max_key_imag)
        # print(max_key_real)
    
    # Finding max values
    max_value_real = 0
    max_value_imag = 0
    for key_v, value in shape_derivatives_real.items():
        # real part
        for key_u_real in u_real_indices:
            if abs(shape_derivatives_real[key_v][key_u_real]) > max_value_real:
                max_value_real = abs(shape_derivatives_real[key_v][key_u_real])
        # imag part
        for key_u_imag in u_imag_indices:
            if abs(shape_derivatives_imag[key_v][key_u_imag]) > max_value_imag:
                max_value_imag = abs(shape_derivatives_imag[key_v][key_u_imag])
    
    # print(max_value_real)
    # print(max_value_imag)

    normalized_derivatives = {}
    for key_v, value in shape_derivatives.items():
        normalized_derivatives[key_v] = {}
        for key_u, value_u in shape_derivatives[key_v].items():
            normalized_derivatives[key_v][key_u] =  value_u.real/max_value_real + 1j*value_u.imag/max_value_imag

    return normalized_derivatives

class FFDCylindrical:
    def __init__(self, gmsh_model, l, m , n, dim, tag=-1, includeBoundary=False, parametric=True):
        """This class generates a cylindrical FFD Lattice 

        Args:
            gmsh_model (gmsh.model): gmsh model (after meshing) 
            l (int): number of points in the x direction
            m (int): number of points in the y direction
            n (int): number of points in the z direction
            dim (int): dimension of gmsh entity - 2 or 3
            tag (int, optional): physical tag of the entity, -1 returns all tags. Defaults to -1.
        """
        self.l = l
        self.m = m
        self.n = n

        self.Px = np.zeros((l,m,n))
        self.Py = np.zeros((l,m,n))
        self.Pz = np.zeros((l,m,n))

        self.Pr = np.zeros((l,m,n))
        self.Pphi = np.zeros((l,m,n))
        self.Pz = np.zeros((l,m,n))
        
        if tag==-1:
            elementary_tag = -1
        else:
            elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(dim,tag)
        
        nodes, coords, param = gmsh_model.mesh.getNodes(dim, int(elementary_tag), includeBoundary, parametric)

        xs = coords[0::3]
        ys = coords[1::3]
        zs = coords[2::3]

        rhos, phis, zetas = cart2cyl(xs, ys, zs)

        self.dr = max(rhos)-min(rhos)
        self.dphi = 2*np.pi
        self.dz = max(zetas)-min(zetas)

        for i in range(l):
            for j in range(m):
                for k in range(n):
                    self.Pr[i, j, k] = min(rhos)  + self.dr * i / (l - 1)
                    self.Pphi[i, j, k] = min(phis)  + self.dphi * j / (m -1)
                    self.Pz[i, j, k] = min(zetas)  + self.dz * k / (n -1)

        self.P0 = np.array([self.Pr[0, 0, 0], self.Pphi[0, 0, 0], self.Pz[0, 0, 0]])

        full_nodes, full_coords, full_param = gmsh_model.mesh.getNodes(dim, -1, True, True)
        
        xs_base = full_coords[0::3]
        ys_base = full_coords[1::3]
        zs_base = full_coords[2::3]

        rhos_base, phis_base, zetas_base = cart2cyl(xs_base, ys_base, zs_base)


        self.dr_base = max(rhos_base)-min(rhos_base)
        self.dphi_base = 2*np.pi
        self.dz_base = max(zetas_base)-min(zetas_base)

    def write_ffd_points(self, name="MeshDir/FFD"):
        """writes the FFD points as a vtu file (readable using ParaView).
        """
        
        r_ffd = self.Pr.flatten()
        phi_ffd = self.Pphi.flatten()
        z_ffd = self.Pz.flatten()
        x_ffd, y_ffd, z_ffd = cyl2cart(r_ffd, phi_ffd, z_ffd)
        pointsToVTK(name, x_ffd, y_ffd, z_ffd)
        print("FFD points are saved as "+name+".vtu")
    
    def calcSTU(self, coords):
        """Calculates parametric coordinates for cylindrical lattice

        Args:
            coords (_type_): cartesian coordinates

        Returns:
            _type_: _description_
        """

        xs = coords[0::3]
        ys = coords[1::3]
        zs = coords[2::3]

        rhos, phis, zetas = cart2cyl(xs, ys, zs)

        s = (rhos - self.P0[0])/self.dr
        t = (phis - self.P0[1])/self.dphi
        u = (zetas - self.P0[2])/self.dz

        return s,t,u 

def getMeshdata(gmsh_model):
    """ Calculates the current mesh data which inputs the deformation function.

    Args:
        gmsh_model (_type_): gmsh.model

    Returns:
        dictionary: mesh data
    """
    mesh_data = {}
    for e in gmsh_model.getEntities():
        mesh_data[e] = (gmsh_model.getBoundary([e]),
                gmsh_model.mesh.getNodes(e[0], e[1]),
                gmsh_model.mesh.getElements(e[0], e[1]))
    return mesh_data

def getLocalMeshdata(gmsh_model, dim, tag):
    """ Calculates the current mesh data which inputs the deformation function.

    Args:
        gmsh_model (_type_): gmsh.model
        tag: physical tag of the entity

    Returns:
        dictionary: mesh data
    """
    elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(dim,tag)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim,int(elementary_tag))
    
    mesh_data = {}
    for e in gmsh.model.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax):
        mesh_data[e] = (gmsh_model.getBoundary([e]),
                gmsh_model.mesh.getNodes(e[0], e[1]),
                gmsh_model.mesh.getElements(e[0], e[1]))
    return mesh_data

def getNonLocalMeshdata(gmsh_model, dim, tag):
    """ Calculates the current nonlocal mesh data which inputs the deformation function.

    Args:
        gmsh_model (_type_): gmsh.model
        tag: physical tag of the entity

    Returns:
        dictionary: mesh data
    """
    elementary_tag = gmsh.model.getEntitiesForPhysicalGroup(dim,tag)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim,int(elementary_tag))
    local_entities = gmsh.model.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)
    global_entities = gmsh.model.getEntities()
    non_local_entities = list(set(global_entities) - set(local_entities))

    mesh_data = {}
    for e in non_local_entities:
        mesh_data[e] = (gmsh_model.getBoundary([e]),
                gmsh_model.mesh.getNodes(e[0], e[1]),
                gmsh_model.mesh.getElements(e[0], e[1]))
    return mesh_data

def calcSTU(coords, P0, dx, dy, dz):
    """
    Calc STU (parametric) coordinates in cartesian grid
    """
    xs = coords[0::3]
    ys = coords[1::3]
    zs = coords[2::3]

    s = (xs - P0[0])/dx
    t = (ys - P0[1])/dy
    u = (zs - P0[2])/dz

    return s,t,u

def deformCylindricalFFD(gmsh_model, mesh_data, CylindricalLattice):

    l,m,n = CylindricalLattice.l,CylindricalLattice.m, CylindricalLattice.n
    for e in mesh_data:

        if len(mesh_data[e][1][1])==3:

            old_coord = mesh_data[e][1][1]
            s,t,u = CylindricalLattice.calcSTU(old_coord)
            Xdef = np.zeros((1,3))

        else:
            old_coords = mesh_data[e][1][1]
            s,t,u = CylindricalLattice.calcSTU(old_coords)
            Xdef = np.zeros((int(len(old_coords)/3),3))
        for point, param_s in enumerate(s):
            for i in range(l):
                for j in range(m):
                    for k in range(n):
                        Xdef[point] +=  comb(l-1,i)*np.power(1-s[point], l-1-i)*np.power(s[point],i) * \
                                        comb(m-1,j)*np.power(1-t[point], m-1-j)*np.power(t[point],j) * \
                                        comb(n-1,k)*np.power(1-u[point], n-1-k)*np.power(u[point],k) * \
                                        np.asarray([CylindricalLattice.Pr[i,j,k], 
                                                    CylindricalLattice.Pphi[i,j,k],
                                                    CylindricalLattice.Pz[i,j,k]])

        Xdef_3d_cart = Xdef.copy()
        Xdef_3d_cart[:,0], Xdef_3d_cart[:,1],Xdef_3d_cart[:,2] = cyl2cart(Xdef[:,0], Xdef[:,1], Xdef[:,2]) 
        new_coord = Xdef_3d_cart.flatten()
        
        gmsh.model.addDiscreteEntity(e[0], e[1], [b[1] for b in mesh_data[e][0]])
        gmsh.model.mesh.addNodes(e[0], e[1], mesh_data[e][1][0], new_coord)
        gmsh.model.mesh.addElements(e[0], e[1], mesh_data[e][2][0], mesh_data[e][2][1], mesh_data[e][2][2])

    return gmsh_model

def deformCylindricalLocalFFD(gmsh_model, local_mesh_data, nonlocal_mesh_data, CylindricalLattice):
    # first, add local mesh data
    gmsh.model = deformCylindricalFFD(gmsh.model, local_mesh_data, CylindricalLattice)
    
    # then add the nonlocal data
    for e in sorted(nonlocal_mesh_data):
        
        coord = []
        for i in range(0, len(nonlocal_mesh_data[e][1][1]), 3):
            x = nonlocal_mesh_data[e][1][1][i]
            y = nonlocal_mesh_data[e][1][1][i + 1]
            z = nonlocal_mesh_data[e][1][1][i + 2]
            coord.append(x)
            coord.append(y)
            coord.append(z)

        gmsh.model.addDiscreteEntity(e[0], e[1], [b[1] for b in nonlocal_mesh_data[e][0]])
        gmsh.model.mesh.addNodes(e[0], e[1], nonlocal_mesh_data[e][1][0], coord)
        gmsh.model.mesh.addElements(e[0], e[1], nonlocal_mesh_data[e][2][0], nonlocal_mesh_data[e][2][1], nonlocal_mesh_data[e][2][2])

    return gmsh_model