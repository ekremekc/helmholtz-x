from ufl import Measure, TestFunction, TrialFunction, inner, as_vector, grad, dx
from dolfinx.cpp.geometry import determine_point_ownership
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem  import FunctionSpace, form
from .parameters_utils import gamma_function
from .solver_utils import info
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import basix

class FlameMatrix:
    def  __init__(self, mesh, h, q_0, u_b, FTF, degree, bloch_object=None, tol=1e-5):

        self.mesh = mesh
        self.h = h
        self.q_0 = q_0 
        self.u_b = u_b
        self.FTF = FTF
        self.degree = degree
        self.bloch_object=bloch_object
        self.tol = tol
        
        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)
        self.gdim = self.mesh.geometry.dim

        # Matrix utils
        self.global_size = self.V.dofmap.index_map.size_global
        self.local_size = self.V.dofmap.index_map.size_local

        # Vector for reference direction
        if self.gdim == 1:
            self.n_ref_dist = as_vector([1])
            self.n_ref_pointwise = np.array([[1]])
        elif self.gdim == 2:
            self.n_ref_dist = as_vector([1,0])
            self.n_ref_pointwise = np.array([[1, 0]]).T
        else:
            self.n_ref_dist = as_vector([0,0,1])
            self.n_ref_pointwise = np.array([[0, 0, 1]]).T

        # Utility objects for flame matrix
        self._D_ij = None
        self._D_ij_adj = None
        self._D = None
        self._D_adj = None

    @property
    def matrix(self):
        return self._D
    @property
    def submatrices(self):
        return self._D_ij
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def adjoint_submatrices(self):
        return self._D_ij_adj
    
    @staticmethod
    def indices_and_values(dofmaps, form, tol):

        temp = assemble_vector(form)
        temp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        temp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        packed = temp.array
        packed.real[abs(packed.real) < tol] = 0.0
        packed.imag[abs(packed.imag) < tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))
        return packed

    @staticmethod
    def get_sparse_matrix_data(left, right, problem_type='direct'):
        if problem_type=='direct':
            row = [item[0] for item in left]
            col = [item[0] for item in right]

            row_vals = [item[1] for item in left]
            col_vals = [item[1] for item in right]
        
        elif problem_type=='adjoint':
            row = [item[0] for item in right]
            col = [item[0] for item in left]

            row_vals = [item[1] for item in right]
            col_vals = [item[1] for item in left]

        product = np.outer(row_vals,col_vals)
        val = product.flatten()

        return row, col, val
    
    def assemble_matrix(self, omega, problem_type='direct'):

        if problem_type == 'direct':
            self._D = self._D_ij*self.FTF(omega)
            info("- Direct matrix D is assembling...")
       
        elif problem_type == 'adjoint':
            self._D_adj = self._D_ij_adj * np.conj(self.FTF(np.conj(omega)))
            info("- Adjoint matrix D is assembling...")
        
        info("- Matrix D is assembled.")
    
    def get_derivative(self, omega):

        dD_domega = self.FTF.derivative(omega) * self._D_ij
        info("- Derivative of matrix D is assembled.")

        return dD_domega

    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':
            D_ij_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_ij = D_ij_bloch

        elif problem_type == 'adjoint':
            D_ij_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_ij_adj = D_ij_adj_bloch
    
class PointwiseFlameMatrix(FlameMatrix):

    def __init__(self, mesh, subdomains, x_r, h, rho_u, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=1.4, tol=1e-10):

        super().__init__(mesh, h, q_0, u_b, FTF, degree, bloch_object, tol)
        self.x_r = x_r
        self.rho_u = rho_u
        self.gamma = gamma
        self.dx = Measure("dx", subdomain_data=subdomains)
        
        # Reference cell data required for evaluation of derivative
        ct = self.mesh.basix_cell()
        self.coordinate_element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct.name), basix.cell.string_to_type(ct.name), 1, basix.LagrangeVariant.equispaced)

        # Data required for pull back of coordinate
        num_dofs_x = self.mesh.geometry.dofmap[0].size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        self.x_dofs = self.mesh.geometry.dofmap.reshape(num_cells, num_dofs_x)

    def _assemble_vectors(self, flame, point):
        
        left_form = form((self.gamma - 1) * self.q_0 / self.u_b * inner(self.h, self.phi_j)*self.dx(flame))
        left_vector = self.indices_and_values(self.dofmaps, left_form, tol=self.tol)

        _, _, owning_points, cell = determine_point_ownership( self.mesh._cpp_object, point, 1e-10)
        point_ref = np.zeros((len(cell), self.mesh.geometry.dim), dtype=self.mesh.geometry.x.dtype)
        right_vector = []

        if len(cell) > 0: # Only add contribution if cell is owned 

            cell_geometry = self.mesh.geometry.x[self.x_dofs[cell[0]], :self.gdim]
            point_ref = self.mesh.geometry.cmaps[0].pull_back([point[:self.gdim]], cell_geometry)
            dphi = self.coordinate_element.tabulate(1, point_ref)[1:,0,:]
            dphi = dphi.reshape((dphi.shape[0], dphi.shape[1]))
            
            J = np.dot(cell_geometry.T, dphi.T)
            Jinv = np.linalg.inv(J)  

            cell_dofs = self.dofmaps.cell_dofs(cell[0])
            global_dofs = self.dofmaps.index_map.local_to_global(cell_dofs)
            d_dx = (Jinv.T @ dphi).T
            d_dphi_j = np.dot(d_dx, np.array(self.n_ref_pointwise))[:, 0] / self.rho_u
            for i in range(len(d_dphi_j)):
                right_vector.append([global_dofs[i], d_dphi_j[i]])

        right_vector = broadcast_vector(right_vector)

        return left_vector, right_vector

    def assemble_submatrices(self, problem_type='direct'):

        info("- Generating matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('aij') 
        mat.setUp()

        for flame, point in enumerate(self.x_r):
            
            left, right = self._assemble_vectors(flame, point)
            row,col,val = self.get_sparse_matrix_data(left, right, problem_type=problem_type)

            mat.setValues(row,col,val, addv=PETSc.InsertMode.ADD_VALUES)
            info("- Matrix contribution of flame "+str(flame)+" is computed.")

        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D_ij = mat
        elif problem_type == 'adjoint':
            self._D_ij_adj = mat

class DistributedFlameMatrix(FlameMatrix):

    def __init__(self, mesh, w, h, rho, T, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=None, tol=1e-5):
        super().__init__(mesh, h, q_0, u_b, FTF, degree, bloch_object, tol)

        if gamma==None: # Variable gamma depends on temperature
            gamma = gamma_function(T) 

        self.left_form = form((gamma - 1) * q_0 / u_b * self.phi_i * h *  dx)
        self.right_form = form(inner(self.n_ref_dist,grad(self.phi_j)) / rho * w * dx)
    
    def _assemble_vectors(self, problem_type='direct'):
       
        left_vector = self.indices_and_values(self.dofmaps, self.left_form, self.tol)
        right_vector = self.indices_and_values(self.dofmaps, self.right_form, self.tol)

        if problem_type == 'direct':
            left_vector = distribute_vector_as_chunks(left_vector)
            right_vector = broadcast_vector(right_vector)
        elif problem_type == 'adjoint':
            right_vector = distribute_vector_as_chunks(right_vector)
            left_vector = broadcast_vector(left_vector)

        return left_vector, right_vector

    def assemble_submatrices(self, problem_type='direct'):

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('mpiaij')

        left, right = self._assemble_vectors(problem_type=problem_type)
        row,col,val = self.get_sparse_matrix_data(left, right, problem_type=problem_type)

        info("- Generating matrix D..")

        ONNZ = len(col)*np.ones(self.local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D_ij = mat
        elif problem_type == 'adjoint':
            self._D_ij_adj = mat

def distribute_vector_as_chunks(vector):
    vector = MPI.COMM_WORLD.gather(vector, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        vector = [j for i in vector for j in i]
        chunks = [[] for _ in range(MPI.COMM_WORLD.Get_size())]
        for i, chunk in enumerate(vector):
            chunks[i % MPI.COMM_WORLD.Get_size()].append(chunk)
    else:
        vector = None
        chunks = None
    vector = MPI.COMM_WORLD.scatter(chunks, root=0)
    return vector

def broadcast_vector(vector):
    vector = MPI.COMM_WORLD.gather(vector, root=0)
    if vector:
        vector = [j for i in vector for j in i]
    else:
        vector=[]
    vector = MPI.COMM_WORLD.bcast(vector,root=0)
    return vector