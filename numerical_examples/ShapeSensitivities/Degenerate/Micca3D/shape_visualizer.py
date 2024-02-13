
from helmholtz_x.dolfinx_utils import load_xdmf_mesh, write_xdmf_mesh, XDMFReader
from dolfinx.fem import Function, FunctionSpace, dirichletbc, set_bc, locate_dofs_topological
from mpi4py import MPI
import dolfinx.io
import numpy as np

write_xdmf_mesh("MeshDir/Micca",dimension=3)


shape_derivatives={1: [(-106216.64819010241-28842.47439768992j), (-103946.74776043865-18915.913927299673j)],
                   2: [(-120427.8118610474-22496.42216972298j), (-115882.5094856025-7674.228319903667j)],
                   3: [(-63458.781851744425-33189.66853070488j), (-63112.45562449481-26373.75179107947j)],
                   4: [(-73717.70010223417-131939.37288347658j), (-49697.42069054075-63019.47856905913j)],
                #    5: [(684957.2778507026-579238.1325467043j), (532694.4248702549-1595269.9056810471j)],
                #    6: [(967578.1344932624+909531.8896318685j), (999481.6784963995+1340884.680977521j)],
                #    7: [(10869188.591875758+7106584.5287730675j), (10621772.732638739+8758862.148712022j)],
                   8: [(53168.429515806725-26932.53306653828j), (54254.63724106943-37712.01325530575j)],
                   9: [(1810.7615785676226-13717.787385280362j), (3043.4302117964553-17031.987110311344j)],
                   10: [(3039.992465215346-18417.460987942064j), (4626.13921984906-22611.29868677982j)],
                   11: [(-1736.0096429541263+9576.824700545447j), (-4317.264466591498+12156.929409672326j)]}

# import json
# import ast
# with open('shape_derivatives.txt', 'w') as file:
#      file.write(json.dumps(str(shape_derivatives))) # use `json.loads` to do the reverse

# with open('shape_derivatives.txt') as f:
#     data = json.load(f)
   
# # read = json.loads('shape_derivatives.json')
# data = ast.literal_eval(data)
# print(data,type(data))
from helmholtz_x.dolfinx_utils import dict_writer,dict_loader

filename = "shape_derivatives"
dict_writer(filename,shape_derivatives)
data = dict_loader(filename)

assert data == shape_derivatives
print("IO is correct")
normalize = True
if normalize:
    shape_derivatives_real = shape_derivatives.copy()
    shape_derivatives_imag = shape_derivatives.copy()


    for key, value in shape_derivatives.items():
        shape_derivatives_real[key] = value[0].real
        shape_derivatives_imag[key] = value[0].imag 
        shape_derivatives[key] = value[0]  # get the first eigenvalue of each list

    max_key_real = max(shape_derivatives_real, key=lambda y: abs(shape_derivatives_real[y]))
    max_value_real = abs(shape_derivatives_real[max_key_real])
    max_key_imag = max(shape_derivatives_imag, key=lambda y: abs(shape_derivatives_imag[y]))
    max_value_imag = abs(shape_derivatives_imag[max_key_imag])

    normalized_derivatives = shape_derivatives.copy()

    for key, value in shape_derivatives.items():
        normalized_derivatives[key] =  value.real/max_value_real + 1j*value.imag/max_value_imag

    shape_derivatives = normalized_derivatives
print(normalized_derivatives)


micca = XDMFReader("MeshDir/Micca")

mesh, subdomains, facet_tags = micca.getAll()

V = FunctionSpace(mesh, ("CG",1))

fdim = mesh.topology.dim - 1

bcs = []
U = Function(V)
for i in shape_derivatives:
    # print(i,shape_derivatives[i])           
    facets = np.array(facet_tags.indices[facet_tags.values == i])
    dofs = locate_dofs_topological(V, fdim, facets)
    U.x.array[dofs] = shape_derivatives[i] #first element of boundary

# print(U.x.array)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Results/derivatives.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(U)














