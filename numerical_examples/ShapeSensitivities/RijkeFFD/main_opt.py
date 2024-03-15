import datetime
start_time = datetime.datetime.now()
from helmholtz_x.io_utils import XDMFReader, xdmf_writer, dict_writer
from helmholtz_x.parameters_utils import temperature_step,gaussianFunction
from helmholtz_x.acoustic_matrices import AcousticMatrices
from helmholtz_x.flame_transfer_function import nTau
from helmholtz_x.flame_matrices import DistributedFlameMatrix
from helmholtz_x.eigensolvers import fixed_point_iteration
from helmholtz_x.eigenvectors import normalize_eigenvector
from helmholtz_x.dolfinx_utils import absolute
from mpi4py import MPI
import numpy as np
import params
import os

path = os.path.dirname(os.path.abspath(__file__))
mesh_dir = "/MeshDir/Optimized"
mesh_name = "/mesh"
results_dir = "/Results/Optimized"

Rijke3D = XDMFReader(path+mesh_dir+mesh_name)
mesh, subdomains, facet_tags = Rijke3D.getAll()
Rijke3D.getInfo()

boundary_conditions = {1:  {'Neumann'},
                       2:  {'Robin': params.R_out},
                       3:  {'Robin': params.R_in}}

degree = 1

T = temperature_step(mesh, params.x_f, params.T_in, params.T_out)
matrices = AcousticMatrices(mesh, facet_tags, boundary_conditions, T , degree = degree)

FTF = nTau(params.n, params.tau)
rho = params.rho_func(mesh, params.x_f, params.a_f, params.rho_d, params.rho_u)
w = gaussianFunction(mesh, params.x_r, params.a_r)
h = gaussianFunction(mesh, params.x_f, params.a_f)
D = DistributedFlameMatrix(mesh, w, h, rho, T, params.q_0, params.u_b, FTF, degree=degree, gamma=params.gamma)
D.assemble_submatrices('direct')       

target = 180 * 2 * np.pi
E = fixed_point_iteration(matrices, D, target, nev=2, i=0, print_results= False)
omega_dir, p_dir = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

omega_dict = {'direct':omega_dir}
dict_writer(path+results_dir+"/eigenvalue", omega_dict)

xdmf_writer(path+results_dir+"/p_dir", mesh, p_dir)
xdmf_writer(path+results_dir+"/p_dir_abs", mesh, absolute(p_dir))

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)