from dolfinx.fem import Constant, VectorFunctionSpace, Function, dirichletbc, locate_dofs_topological, set_bc, form
from helmholtz_x.petsc4py_utils import conjugate_function
from helmholtz_x.shape_derivatives_utils import ffd_displacement_vector
from dolfinx.fem.assemble import assemble_scalar
from ufl import  FacetNormal, grad, dot, inner, Measure, div, variable
from ufl.operators import Dn #Dn(f) := dot(grad(f), n).
from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import scipy.linalg
import os

def _shape_gradient_Dirichlet(c, p_dir, p_adj_conj):
    # Equation 4.34 in thesis
    return - c**2 * Dn(p_adj_conj) * Dn (p_dir)

def _shape_gradient_Neumann(c, p_dir, p_adj_conj):
    # Equation 4.35 in thesis
    # p_adj_conj = conjugate_function(p_adj)
    return  div(p_adj_conj * c**2 * grad(p_dir))

def _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj):
    # Equation 4.36 in thesis
    return c**2 * dot(grad(p_adj_conj), grad(p_dir)) - omega**2 * p_adj_conj * p_dir  

def _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj_conj, index):

    # FIX ME FOR PARAMETRIC 3D
    if geometry.mesh.topology.dim == 2:
        curvature = geometry.get_curvature_field(index)
    else:
        curvature = 0

    # Equation 4.33 in thesis
    G = -p_adj_conj * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
        _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj) + \
         2 * _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)

    return G
# ________________________________________________________________________________

def degenerate_shape_derivative_neumann_nonaxisymmetric(geometry, lattice, p_dir_1, p_dir_2, p_adj_norm_1, p_adj_norm_2, c,
                                        surface_physical_tag, r_index, deg=1 ):
    """This function calculates the control pointwise shape derivatives of the neumann boundary 

    Args:
        gmsh_model : _description_
        mesh : _description_
        facet_tags : _description_
        p_dir_1 : _description_
        p_dir_2 : _description_
        p_adj_norm_1 : _description_
        p_adj_norm_2 : _description_
        c : _description_
        surface_physical_tag : _description_
        r_index : control point index in the radial direction
        
    Returns:
        shape derivative dictionary
    """

    normal = FacetNormal(geometry.mesh)
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)

    p_adj1_conj = conjugate_function(p_adj_norm_1) # it should be conjugated once
    p_adj2_conj = conjugate_function(p_adj_norm_2) # it should be conjugated once

    G_neu = []
    G_neu.append(_shape_gradient_Neumann(c, p_dir_1, p_adj1_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_dir_2, p_adj1_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_adj_norm_1, p_adj2_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_adj_norm_2, p_adj2_conj))

    derivatives = {}


    for zeta in range(0,lattice.n):

        derivatives[zeta] = {}

        for phi in range(0,lattice.m):

            V_ffd = ffd_displacement_vector(geometry, lattice, surface_physical_tag, r_index, phi, zeta, deg=deg)
            
            # the eigenvalues are 2-fold degenerate
            G = {}
            for index,uflform in enumerate(G_neu):
                G[index] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_ffd, normal) * uflform *ds(surface_physical_tag))), op=MPI.SUM)
            A = np.array(([G[0], G[1]],
                            [G[2], G[3]]))

            eig = scipy.linalg.eigvals(A)
            eig_sort_ind = np.argsort(np.abs(eig))# print(eig)
            derivatives[zeta][phi] = eig[eig_sort_ind[1]]

    return derivatives


def degenerate_shape_derivative_robin_outlet(geometry, omega, lattice, p_dir_1, p_dir_2, p_adj_norm_1, p_adj_norm_2, c,
                                        surface_physical_tag, z_index, deg=1 ):
    """This function calculates the control pointwise shape derivatives of the neumann boundary 

    Args:
        gmsh_model : _description_
        mesh : _description_
        facet_tags : _description_
        p_dir_1 : _description_
        p_dir_2 : _description_
        p_adj_norm_1 : _description_
        p_adj_norm_2 : _description_
        c : _description_
        surface_physical_tag : _description_
        r_index : control point index in the radial direction
        
    Returns:
        shape derivative dictionary
    """

    normal = FacetNormal(geometry.mesh)
    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)

    p_adj1_conj = conjugate_function(p_adj_norm_1) # it should be conjugated once
    p_adj2_conj = conjugate_function(p_adj_norm_2) # it should be conjugated once

    index = 0 # FIXME
    G_neu = []
    G_neu.append(_shape_gradient_Robin(geometry, c, omega, p_dir_1, p_adj1_conj, index))
    G_neu.append(_shape_gradient_Robin(geometry, c, omega, p_dir_2, p_adj1_conj, index))
    G_neu.append(_shape_gradient_Robin(geometry, c, omega, p_adj_norm_1, p_adj2_conj, index))
    G_neu.append(_shape_gradient_Robin(geometry, c, omega, p_adj_norm_2, p_adj2_conj, index))

    derivatives = {}


    for radius in range(0,lattice.l):

        derivatives[radius] = {}

        for phi in range(0,lattice.m):

            V_ffd = ffd_displacement_vector(geometry, lattice, surface_physical_tag, radius, phi, z_index, deg=deg)
            
            # the eigenvalues are 2-fold degenerate
            G = {}
            for index,uflform in enumerate(G_neu):
                G[index] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_ffd, normal) * uflform *ds(surface_physical_tag))), op=MPI.SUM)
            A = np.array(([G[0], G[1]],
                            [G[2], G[3]]))

            eig = scipy.linalg.eigvals(A)
            eig_sort_ind = np.argsort(np.abs(eig))# print(eig)
            derivatives[radius][phi] = eig[eig_sort_ind[1]]

    return derivatives

def PerpendicularShapeDerivativesAxial(geometry, boundary_conditions, omega, p_dir, p_adj, c):
    """Calculates the shape derivatives on the surfaces

    Args:
        geometry (Geometry): Geometry 
        boundary_conditions (dict): boundary conditions of the system
        omega (complex): direct eigenvalue
        p_dir (Function): non-normalized direct eigenvector
        p_adj (Function): conditioned(normalized) adjoint eigenvector
        c (Function): speed of sound field

    Returns:
        dict: Calculated shape derivatives dictionary
    """
    
    results = {} 

    ds = Measure('ds', domain=geometry.mesh, subdomain_data=geometry.facet_tags)
    p_adj_conj = conjugate_function(p_adj)

    #Clean Master and Slave tags
    for k in list(boundary_conditions.keys()):
        if boundary_conditions[k]=={'Master'} or boundary_conditions[k]=={'Slave'}:
            del boundary_conditions[k]
    
    for tag, value in boundary_conditions.items():

        C = Constant(geometry.mesh, PETSc.ScalarType(1))
        A = MPI.COMM_WORLD.allreduce(assemble_scalar(form(C * ds(tag))), op=MPI.SUM) # For parallel runs
        normalizer = Constant(geometry.mesh, PETSc.ScalarType(1/A)) 
        if value == 'Dirichlet':
            G = _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
        elif value == 'Neumann':
            G = _shape_gradient_Neumann(c, p_dir, p_adj_conj)
        else :
            G = _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj_conj, tag)
        
        results[tag] = MPI.COMM_WORLD.allreduce(assemble_scalar(form(normalizer * G * ds(tag))), op=MPI.SUM)  
        
    return results

def axial_shape_derivative_neumann_nonaxisymmetric(gmsh_model, mesh, facet_tags, p_dir_1, p_adj_norm_1, c,
                                        surface_elementary_tag, surface_physical_tag,
                                        N_cpt_v, knots_array_u, knots_array_v, deg=2 ):
    """This function calculates the control pointwise shape derivatives of the neumann boundary 

    Args:
        gmsh_model : _description_
        mesh : _description_
        facet_tags : _description_
        p_dir_1 : _description_
        p_adj_norm_1 : _description_
        c : _description_
        surface_elementary_tag : _description_
        surface_physical_tag : _description_
        N_cpt_v : _description_
        knots_array_u : _description_
        knots_array_v : _description_
        deg (int, optional): degree of the NURBS surface. Defaults to 2.

    Returns:
        shape derivative dictionary
    """

    n = FacetNormal(mesh)
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    p_adj1_conj = conjugate_function(p_adj_norm_1) # it should be conjugated once

    G_neu = _shape_gradient_Neumann(c, p_dir_1, p_adj1_conj)   

    derivatives = {}
        
    for span_v in range(1,N_cpt_v-1): # We do not compute the derivatives of the first and the last cpt on the surface

        derivatives[span_v] = {}

        for span_u in range(0,8):

            V = ffd_displacement_vector(gmsh_model, mesh, facet_tags, 
                                                    surface_elementary_tag, surface_physical_tag, 
                                                    span_u, span_v, 
                                                    knots_array_u, knots_array_v, deg)
            
            eig = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V, n) * G_neu *ds(surface_physical_tag))), op=MPI.SUM)
            print(eig)
            derivatives[span_v][span_u] = eig

    return derivatives