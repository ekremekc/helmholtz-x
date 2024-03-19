import datetime
start_time = datetime.datetime.now()
from helmholtz_x.io_utils import XDMFReader, xdmf_writer, write_xdmf_mesh, dict_writer
from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, sound_speed
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import absolute
from helmholtz_x.shape_derivatives_utils import FFDCylindrical, getMeshdata, nonaxisymmetric_derivatives_normalize
from helmholtz_x.shape_derivatives import shapeDerivativesFFD
from mpi4py import MPI
import numpy as np
import params
import gmsh
import sys
import os  

path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/MeshDir/ShapeDerivatives"
mesh_name = "/mesh"
results_dir = "/Results/ShapeDerivatives"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add(__name__)

R = 0.047/2
L_total = 1

gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L_total, R, tag=1)
gmsh.model.occ.synchronize()

# Physical tags
surfaces = gmsh.model.occ.getEntities(dim=2)

for surface in surfaces:
    gmsh.model.addPhysicalGroup(2, [surface[1]])

gmsh.model.addPhysicalGroup(3, [1], tag=1) # Geometry tag 

lc = 0.007 #0.007

gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(3)

# Retrieve mesh data before it disappears
mesh_data = getMeshdata(gmsh.model)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.write("{}.msh".format(path+mesh_dir+mesh_name))

write_xdmf_mesh(path+mesh_dir+mesh_name,dimension=3)

Rijke3D = XDMFReader(path+mesh_dir+mesh_name)
mesh, subdomains, facet_tags = Rijke3D.getAll()
Rijke3D.getInfo()

boundary_conditions = {1:  {'Neumann'},
                       2:  {'Robin': params.R_out},
                       3:  {'Robin': params.R_in}}

degree = 2

T = temperature_step(mesh, params.x_f, params.T_in, params.T_out) 
c = sound_speed(T)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T , degree = degree)

FTF = nTau(params.n, params.tau)
rho = params.rho_func(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
D = DistributedFlameMatrix(mesh, w, h, rho, T, params.q_0, params.u_b, FTF, degree=degree, gamma=params.gamma)
D.assemble_submatrices('direct')       
                    
target = 180 * 2 * np.pi
E_dir = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False)
omega_dir, p_dir = normalize_eigenvector(mesh, E_dir, 0, degree=degree, which='right')

D.assemble_submatrices('adjoint')       
E_adj = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False, problem_type='adjoint')
omega_adj, p_adj = normalize_eigenvector(mesh, E_adj, 0, degree=degree, which='right')

omega_dict = {'direct':omega_dir, 'adjoint': omega_adj}
dict_writer(path+results_dir+"/eigenvalues", omega_dict)

xdmf_writer(path+results_dir+"/p_dir", mesh, p_dir)
xdmf_writer(path+results_dir+"/p_dir_abs", mesh, absolute(p_dir))

xdmf_writer(path+results_dir+"/p_adj", mesh, p_adj)
xdmf_writer(path+results_dir+"/p_adj_abs", mesh, absolute(p_adj))

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)

### Introducing FFD 
l, m, n = 2, 4, 9
CylindricalLattice = FFDCylindrical(gmsh.model, l, m , n, 3, tag=-1, includeBoundary=True, parametric=False) 
CylindricalLattice.write_ffd_points(path+mesh_dir+"/FFDinitial")

# Computing shape derivatives
derivatives = shapeDerivativesFFD(Rijke3D, CylindricalLattice, 1, omega_dir, p_dir, p_adj, c, matrices, D)
derivatives_normalized = nonaxisymmetric_derivatives_normalize(derivatives)
dict_writer("ShapeDerivatives/normalized",derivatives_normalized)