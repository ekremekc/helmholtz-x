from helmholtz_x.shape_derivatives_utils import calculate_displacement_field
from helmholtz_x.petsc4py_utils import conjugate_function
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.fem import Constant, form
from ufl import  FacetNormal, grad, dot, inner, Measure, div
from ufl.operators import Dn
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import scipy.linalg

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

def ShapeDerivativesAxial(geometry, boundary_conditions, omega, p_dir, p_adj, c):
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
            G = _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj)
        else :
            G = _shape_gradient_Robin(geometry, c, omega, p_dir, p_adj_conj, tag)
        
        results[tag] = MPI.COMM_WORLD.allreduce(assemble_scalar(form(normalizer * G * ds(tag))), op=MPI.SUM)  
        
    return results

def parametric_degenerate_shape_derivative_neumann(gmsh_model, mesh, facet_tags, p_dir_1, p_dir_2, p_adj_norm_1, p_adj_norm_2, c,
                                        surface_elementary_tag, surface_physical_tag,
                                        span_u, N_cpt_v, knots_array_u, knots_array_v, deg=2 ):

    n = FacetNormal(mesh)
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    p_adj1_conj = conjugate_function(p_adj_norm_1) # it should be conjugated once
    p_adj2_conj = conjugate_function(p_adj_norm_2) # it should be conjugated once

    G_neu = []
    G_neu.append(_shape_gradient_Neumann(c, p_dir_1, p_adj1_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_dir_2, p_adj1_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_adj_norm_1, p_adj2_conj))
    G_neu.append(_shape_gradient_Neumann(c, p_adj_norm_2, p_adj2_conj))

    derivatives = {}

    for span_v in range(1,N_cpt_v-1): # We do not compute the derivatives of the first and the last cpt on the surface
        
        V = calculate_displacement_field(gmsh_model, mesh, facet_tags, 
                                                surface_elementary_tag, surface_physical_tag, 
                                                span_u, span_v, 
                                                knots_array_u, knots_array_v, deg)
        
        # the eigenvalues are 2-fold degenerate
        G = {}
        for index,uflform in enumerate(G_neu):
            G[index] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V, n) * uflform *ds(surface_physical_tag))), op=MPI.SUM)
        A = np.array(([G[0], G[1]],
                        [G[2], G[3]]))

        eig = scipy.linalg.eigvals(A)
        eig_sort_ind = np.argsort(np.abs(eig))
        derivatives[span_v] = eig[eig_sort_ind[1]]

    return derivatives

def ShapeDerivativesDegenerate(geometry, boundary_conditions, omega, 
                               p_dir1, p_dir2, p_adj1, p_adj2, c):
    
    #Clean Master and Slave tags
    for k in list(boundary_conditions.keys()):
        if boundary_conditions[k]=='Master' or boundary_conditions[k]=='Slave':
            del boundary_conditions[k]

    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)
    p_adj1_conj, p_adj2_conj = conjugate_function(p_adj1), conjugate_function(p_adj2)
    results = {} 

    for tag, value in boundary_conditions.items():
        # print("Shape derivative of ", tag)
        C = Constant(geometry.mesh, PETSc.ScalarType(1))
        A = assemble_scalar(form(C * ds(tag)))
        # print("Area of : ", A)
        A = MPI.COMM_WORLD.allreduce(A, op=MPI.SUM) # For parallel runs
        print("tag: ", tag, "BC: ",value)
        C = C / A

        G = []
        if value == 'Dirichlet':

            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj1_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj1_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir1, p_adj2_conj))
            G.append(_shape_gradient_Dirichlet(c, p_dir2, p_adj2_conj))
        elif value == 'Neumann':
            
            G.append(_shape_gradient_Neumann2(c, omega, p_dir1, p_adj1_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir2, p_adj1_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir1, p_adj2_conj))
            G.append(_shape_gradient_Neumann2(c, omega, p_dir2, p_adj2_conj))
        else :

            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj1_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj1_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir1, p_adj2_conj, tag))
            G.append(_shape_gradient_Robin(geometry, c, omega, p_dir2, p_adj2_conj, tag))
        
        # the eigenvalues are 2-fold degenerate
        for index,uflform in enumerate(G):
            # value = assemble_scalar(C * form *ds(tag))
            G[index] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(C * uflform *ds(tag))), op=MPI.SUM)
        A = np.array(([G[0], G[1]],
                      [G[2], G[3]]))
        
        eig = scipy.linalg.eigvals(A)
        # print("eig: ",eig)
        results[tag] = eig.tolist()
    
    return results

def ShapeDerivativesParametric2D(geometry, boundary_conditions, omega, p_dir, p_adj, c):

    mesh = geometry.mesh
    facet_tags = geometry.facet_tags
    
    n = FacetNormal(mesh)
    ds = Measure('ds', domain = mesh, subdomain_data = facet_tags)

    p_adj_conj = conjugate_function(p_adj) # it should be conjugated once

    results = {}
    
    for tag, value in boundary_conditions.items():
        
        if tag in geometry.ctrl_pts:
            derivatives = np.zeros((len(geometry.ctrl_pts[tag]),2), dtype=complex)

            if value == 'Dirichlet':
                G = _shape_gradient_Dirichlet(c, p_dir, p_adj)
            elif value == 'Neumann':
                G = _shape_gradient_Neumann(c, p_dir, p_adj_conj)
            else :
                G = _shape_gradient_Robin(geometry, c, omega, p_dir,  p_adj_conj, tag)

            for j in range(len(geometry.ctrl_pts[tag])):

                V_x, V_y = geometry.get_displacement_field(tag,j)
                
                derivatives[j][0] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_x, n) * G * ds(tag))), op=MPI.SUM)
                derivatives[j][1] = MPI.COMM_WORLD.allreduce(assemble_scalar( form(inner(V_y, n) * G * ds(tag))), op=MPI.SUM)
            
            results[tag] = derivatives
            
    return results

def ShapeDerivativesLocal2D(geometry, boundary_conditions, omega, p_dir, p_adj, c):

    ds = Measure('ds', domain = geometry.mesh, subdomain_data = geometry.facet_tags)

    p_adj_conj = conjugate_function(p_adj) # it should be conjugated once

    results = {}

    const = Constant(geometry.mesh, PETSc.ScalarType(1))

    for tag, value in boundary_conditions.items():
        
        L = MPI.COMM_WORLD.allreduce(assemble_scalar(const * ds(tag)), op=MPI.SUM)
        # print("L: ", L)
        C = const / L
        
        if value == 'Dirichlet':
            G = _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
        elif value == 'Neumann':
            G = _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj)
        else :
            curvature = 0
            G = -p_adj_conj * (curvature * c ** 2 + c * Dn(c)*Dn(p_dir)) + \
            _shape_gradient_Neumann2(c, omega, p_dir, p_adj_conj) + \
            2 * _shape_gradient_Dirichlet(c, p_dir, p_adj_conj)
        print("CHECK: ", MPI.COMM_WORLD.allreduce(assemble_scalar(G * ds(tag)), op=MPI.SUM)) 
        results[tag] = MPI.COMM_WORLD.allreduce(assemble_scalar(C * G * ds(tag)), op=MPI.SUM)
            
    return results