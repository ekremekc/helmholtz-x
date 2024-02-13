from helmholtz_x.flame_matrices import ActiveFlame
from helmholtz_x.flame_transfer_function import n_tau
from helmholtz_x.eigensolvers import fixed_point_iteration_pep
from helmholtz_x.acoustic_matrices import PassiveFlame
from helmholtz_x.eigenvectors import normalize_eigenvector, normalize_adjoint
from helmholtz_x.petsc4py_utils import vector_matrix_vector
from math import pi

from mpi4py import MPI
from geometry import Geometry
from dolfinx.mesh import meshtags, locate_entities
import dolfinx
import numpy as np
import os
import matplotlib.pyplot as plt
import params

if not os.path.exists("MeshDir"):
    os.mkdir("MeshDir")
    
lcar =0.01

p0 = [0., + .0235]
p1 = [0., - .0235]
p2 = [1., - .0235]
p3 = [1., + .0235]

points  = [p0, p1, p2, p3]

edges = {1:{"points":[points[0], points[1]], "parametrization": False},
         2:{"points":[points[1], points[2]], "parametrization": True, "numctrlpoints":10},
         3:{"points":[points[2], points[3]], "parametrization": False},
         4:{"points":[points[3], points[0]], "parametrization": True, "numctrlpoints":10}}

geometry = Geometry("MeshDir/ekrem", points, edges, lcar)
geometry.make_mesh(False)

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)

tdim = geometry.mesh.topology.dim
marked_cells = locate_entities(geometry.mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = meshtags(geometry.mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))


degree = 2

boundary_conditions = {1: {'Robin': params.R_in},
                       2: 'Neumann',
                       3: {'Robin': params.R_out},
                       4: 'Neumann'}

c = params.c(geometry.mesh)

matrices = PassiveFlame(geometry.mesh, subdomains, geometry.facet_tags, boundary_conditions, c, degree =degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

ftf = n_tau(params.n, params.tau)

D = ActiveFlame(geometry.mesh, subdomains,
                    params.x_r, params.rho_in, 1., 1., ftf,
                    degree=degree)

D.assemble_submatrices('direct')

E = fixed_point_iteration_pep(matrices, D, pi, i=0, print_results = False)

omega_dir, p_dir = normalize_eigenvector(geometry.mesh, E, i=0, degree=degree)

# ________________________________________________________________________________

D.assemble_submatrices('adjoint')

E_adj = fixed_point_iteration_pep(matrices, D, pi, i=0, problem_type='adjoint', print_results=False)

omega_adj, p_adj = normalize_eigenvector(geometry.mesh, E_adj, i=0, degree=degree)

print("omega_dir: ", omega_dir,
          "omega_adj: ", omega_adj)
# ____________________________________

p_adj_norm = normalize_adjoint(omega_dir, p_dir, p_adj, matrices)

dL_domega = ( matrices.B + matrices.C*(2 * omega_dir))

print("Normalization check: ", np.abs(vector_matrix_vector(p_adj_norm.vector,dL_domega,p_dir.vector)))


from helmholtz_x.shape_derivatives import ShapeDerivatives

shapeder = ShapeDerivatives(geometry, boundary_conditions, p_dir, p_adj_norm, c)

shape_der_4 = shapeder(4)

print(shape_der_4)


ctrl_pts_2 = np.array([np.array(foo) for foo in geometry.ctrl_pts[2]])
ctrl_pts_4 = np.array([np.array(foo) for foo in geometry.ctrl_pts[4]])

fig1 = plt.figure(figsize = (8,8))

ax1 = fig1.add_subplot(211)

ax1.set_ylim([-0.12,0.12])
ax1.plot(ctrl_pts_2[:, 0], ctrl_pts_2[:, 1], 's-k', markersize=1.5)
ax1.plot(ctrl_pts_4[:, 0], ctrl_pts_4[:, 1], 's-k', markersize=1.5)
ax1.set_ylabel(r"$\omega_r'$")
scale = 0.005

# for i, point in enumerate(ctrl_pts_2):
#     ax1.arrow(point[0], point[1], 0, shape_der2[2][i].real*scale, head_width=0.005, head_length=0.005, fc='orange', ec='orange')

for i, point in enumerate(ctrl_pts_4):
    ax1.arrow(point[0], point[1], 0, shape_der_4[i][1].real*scale, head_width=0.005, head_length=0.005, fc='orange', ec='orange')


ax2 = fig1.add_subplot(212)
ax2.set_ylim([-0.12,0.12])
ax2.plot(ctrl_pts_2[:, 0], ctrl_pts_2[:, 1], 's-k', markersize=1.5)
ax2.plot(ctrl_pts_4[:, 0], ctrl_pts_4[:, 1], 's-k', markersize=1.5)
ax2.set_ylabel(r"$\omega_i'$")
scale = 0.15

# for i, point in enumerate(ctrl_pts_2):
#     ax2.arrow(point[0], point[1], 0, shape_der2[2][i].imag*scale, head_width=0.005, head_length=0.005, fc='r', ec='r')

for i, point in enumerate(ctrl_pts_4):
    ax2.arrow(point[0], point[1], 0, shape_der_4[i][1].imag*scale, head_width=0.005, head_length=0.005, fc='r', ec='r')

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])
fig1.tight_layout()

plt.savefig("derivatives_active.pdf")





