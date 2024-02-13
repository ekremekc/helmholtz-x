from mpi4py import MPI
from petsc4py import PETSc
from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.dolfinx_utils import XDMFReader, xdmf_writer, write_xdmf_mesh
from helmholtz_x.flame_transfer_function import state_space
from helmholtz_x.eigensolvers import fixed_point_iteration_eps
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector, normalize_adjoint

import params

from datetime import datetime
startTime = datetime.now()

from micca_flame import geom_1
if MPI.COMM_WORLD.rank == 0:
        
    # Generate mesh
    msh_params = {'R_in_p': .14,
            'R_out_p': .21,
            'l_p': .07,
            'h_b': .0165,
            'l_b': .014,
            'h_pp': .00945,
            'l_pp': .006,
            'h_f': .018,
            'l_f': .006,
            'R_in_cc': .15,
            'R_out_cc': .2,
            'l_cc': .2,
            'l_ec': 0.041,
            'lc_1': 2e-2,
            'lc_2': 2e-2
            }


    foo = {'pl_rear': 1,
    'pl_outer': 2,
    'pl_inner': 3,
    'pl_front': 4,
    'b_lateral': 5,
    'b_front': 6,
    'pp_lateral': 7,
    'cc_rear': 8,
    'cc_outer': 9,
    'cc_inner': 10,
    'cc_front': 11
    }

    geom_1('MeshDir/Micca', fltk=False, **msh_params)
    write_xdmf_mesh("MeshDir/micca_flame", 3)

# Read mesh 
micca = XDMFReader("MeshDir/Micca")
mesh, subdomains, facet_tags = micca.getAll()

FTF = state_space(params.S1, params.s2, params.s3, params.s4)

# ________________________________________________________________________________
# EVERYWHERE İS NEUMANN EXCEPT OUTLET(COMBUSTION CHAMBER OUTLET)
boundary_conditions = {1: 'Neumann',
                       2: 'Neumann',
                       3: 'Neumann',
                       4: 'Neumann',
                       5: 'Neumann',
                       6: 'Neumann',
                       7: 'Neumann',
                       8: 'Neumann',
                       9: 'Neumann',
                       10: 'Neumann',
                       11: 'Dirichlet'}

degree = 2

target_dir = PETSc.ScalarType(3200)
target_adj = PETSc.ScalarType(3200)
c = params.c(mesh)

matrices = PassiveFlame(mesh, subdomains, facet_tags, boundary_conditions, c, degree=degree)
matrices.assemble_A()
matrices.assemble_C()

D = ActiveFlame(mesh, subdomains, params.x_r, params.rho_amb, params.Q_tot, params.U_bulk, FTF, degree=degree)

D.assemble_submatrices('direct')

E = fixed_point_iteration_eps(matrices, D, target_dir**2, i=0, tol=1e-4)

omega_1, p_dir1 = normalize_eigenvector(mesh, E, i=0, degree=degree)
omega_2, p_dir2 = normalize_eigenvector(mesh, E, i=1, degree=degree)
# ________________________________________________________________________________

D.assemble_submatrices('adjoint')

E_adj = fixed_point_iteration_eps(matrices, D, target_adj**2, i=0, tol=1e-4, problem_type='adjoint')

omega_adj_1, p_adj_1 = normalize_eigenvector(mesh, E_adj, i=0, degree=degree)
omega_adj_2, p_adj_2 = normalize_eigenvector(mesh, E_adj, i=1, degree=degree)

p_adj_norm_1 = normalize_adjoint(omega_1, p_dir1, p_adj_1, matrices, D)
p_adj_norm_2 = normalize_adjoint(omega_2, p_dir2, p_adj_2, matrices, D)

# Save eigenvalues, eigenvectors and shape derivatives

eigs = {'omega_1': omega_1,
        'omega_2': omega_2,
        'omega_adj_1': omega_adj_1,
        'omega_adj_2': omega_adj_2}

print(eigs)

xdmf_writer("Results/p_1", mesh, p_dir1)
xdmf_writer("Results/p_2", mesh, p_dir2)

xdmf_writer("Results/p_adj_1", mesh, p_adj_1)
xdmf_writer("Results/p_adj_2", mesh, p_adj_2)


from helmholtz_x.shape_derivatives_x import ShapeDerivativesDegenerate

omega = (omega_1 + omega_2)/2

results = ShapeDerivativesDegenerate(micca, boundary_conditions, omega, 
                               p_dir1, p_dir2, p_adj_norm_1, p_adj_norm_2, c)

print(results)

print("Total execution time is: ", datetime.now() - startTime)