from dolfinx.fem  import FunctionSpace, Constant, form, assemble_scalar
from dolfinx.fem.petsc import assemble_vector
from ufl import Measure, TestFunction, TrialFunction, inner, as_vector, grad, dx, exp, conj
from .parameters_utils import gamma_function
from .solver_utils import info
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import basix
import dolfinx

class PointwiseFlameMatrix:

    def __init__(self, mesh, subdomains, x_r, rho_u, q_0, u_b, FTF, degree=1, bloch_object=None, gamma=1.4, tol=1e-10):

        self.mesh = mesh
        self.subdomains = subdomains
        self.x_r = x_r
        self.FTF = FTF
        self.bloch_object = bloch_object
        self.coeff = (gamma - 1) * q_0 / (u_b * rho_u)
        self.tol = tol

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)
        self.dx = Measure("dx", subdomain_data=self.subdomains)
        self.gdim = self.mesh.geometry.dim

        # PETSc matrix utils
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.global_size = self.V.dofmap.index_map.size_global
        self.local_size = self.V.dofmap.index_map.size_local

        # Reference cell data required for evaluation of derivative
        ct = self.mesh.basix_cell()
        self.coordinate_element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct.name), basix.cell.string_to_type(ct.name), 1, basix.LagrangeVariant.equispaced)

        # Data required for pull back of coordinate
        num_dofs_x = self.mesh.geometry.dofmap[0].size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        self.x_dofs = self.mesh.geometry.dofmap.reshape(num_cells, num_dofs_x)

        if self.gdim == 1:
            self.n_ref = np.array([[1]])
        elif self.gdim == 2:
            self.n_ref = np.array([[1, 0]]).T
        else:
            self.n_ref = np.array([[0, 0, 1]]).T

        # Utility objects for flame matrix
        self._a = {}
        self._b = {}
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        # Starting assembly for the flames
        for flame, x in enumerate(self.x_r):
            self._a[str(flame)] = self._assemble_left_vector(flame)
            self._b[str(flame)] = self._assemble_right_vector(x)
            info("- Matrix contribution of flame "+str(flame)+" is computed.")

    @property
    def matrix(self):
        return self._D
    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _assemble_left_vector(self, fl):

        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*self.dx(fl))
        V_fl = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        gradient_form = form(inner(1/V_fl, self.phi_j)*self.dx(fl))

        left_vector = indices_and_values(self.dofmaps, gradient_form, tol=self.tol)

        return left_vector

    def _assemble_right_vector(self, point):
        
        _, _, owning_points, cell = dolfinx.cpp.geometry.determine_point_ownership( self.mesh._cpp_object, point, 1e-10)

        point_ref = np.zeros((len(cell), self.mesh.geometry.dim), dtype=self.mesh.geometry.x.dtype)

        right_vector = []

        # Only add contribution if cell is owned 
        if len(cell) > 0:

            cell_geometry = self.mesh.geometry.x[self.x_dofs[cell[0]], :self.gdim]
            point_ref = self.mesh.geometry.cmaps[0].pull_back([point[:self.gdim]], cell_geometry)
            dphi = self.coordinate_element.tabulate(1, point_ref)[1:,0,:]
            dphi = dphi.reshape((dphi.shape[0], dphi.shape[1]))
            
            J = np.dot(cell_geometry.T, dphi.T)
            Jinv = np.linalg.inv(J)  

            cell_dofs = self.dofmaps.cell_dofs(cell[0])
            global_dofs = self.dofmaps.index_map.local_to_global(cell_dofs)
            d_dx = (Jinv.T @ dphi).T
            d_dphi_j = np.dot(d_dx, self.n_ref)[:, 0]
            for i in range(len(d_dphi_j)):
                right_vector.append([global_dofs[i], d_dphi_j[i]])

        right_vector = broadcast_vector(right_vector)

        return right_vector

    @staticmethod
    def _csr_matrix(a, b):

        nnz = len(a) * len(b)
        
        row = np.zeros(nnz)
        col = np.zeros(nnz)
        val = np.zeros(nnz, dtype=np.complex128)

        for i, c in enumerate(a):
            for j, d in enumerate(b):
                row[i * len(b) + j] = c[0]
                col[i * len(b) + j] = d[0]
                val[i * len(b) + j] = c[1] * d[1]

        row = row.astype(dtype='int32')
        col = col.astype(dtype='int32')

        return row, col, val

    def assemble_submatrices(self, problem_type='direct'):

        row, col, val = dict(), dict(), dict()

        for fl, point_contribution in enumerate(self.x_r):

            a = None
            b = None

            if problem_type == 'direct':
                a = self._a[str(fl)]
                b = self._b[str(fl)]

            elif problem_type == 'adjoint':
                a = self._b[str(fl)]
                b = self._a[str(fl)]

            row[str(fl)], col[str(fl)], val[str(fl)] = self._csr_matrix(a, b)

        row = np.concatenate([row[str(fl)] for fl in range(len(self.x_r))])
        col = np.concatenate([col[str(fl)] for fl in range(len(self.x_r))])
        val = np.concatenate([val[str(fl)] for fl in range(len(self.x_r))])
        
        i = np.argsort(row)

        row, col, val = row[i], col[i], val[i]

        info("- Generating matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('aij') 
        mat.setUp()
        for i in range(len(row)):
            mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D_kj = mat
        elif problem_type == 'adjoint':
            self._D_kj_adj = mat

    def assemble_matrix(self, omega, problem_type='direct'):

        if problem_type == 'direct':

            z = self.FTF(omega)
            self._D = self._D_kj*z*self.coeff
            info("- Direct matrix D is assembling...")
       
        elif problem_type == 'adjoint':

            z = np.conj(self.FTF(np.conj(omega)))
            self._D_adj = self.coeff * z * self._D_kj_adj
            info("- Adjoint matrix D is assembling...")
        
        info("- Matrix D is assembled.")

    def get_derivative(self, omega):

        z = self.FTF(omega, k=1)
        dD_domega = z * self._D_kj
        dD_domega = self.coeff * dD_domega
        info("- Derivative of matrix D is assembled.")

        return dD_domega

    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':

            D_kj_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_kj = D_kj_bloch

        elif problem_type == 'adjoint':

            D_kj_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_kj_adj = D_kj_adj_bloch

class DistributedFlameMatrix:

    def __init__(self, mesh, w, h, rho, T, eta, tau, degree=1, gamma=None, tol=1e-5):

        self.mesh = mesh
        self.w = w
        self.h = h
        self.rho = rho
        self.T = T
        self.eta = eta
        self.tau = tau
        self.degree = degree

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = tol
        self.omega = Constant(self.mesh, PETSc.ScalarType(0))

        if gamma==None: # Variable gamma
            gamma = gamma_function(self.T) 

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        phi_i = TrialFunction(self.V)
        phi_j = TestFunction(self.V)

        self.dx = dx

        # left vector
        self.left_form_direct = form((gamma - 1) * eta * phi_i * h * exp(1j*self.omega*tau)  *  dx)
        self.left_form_adjoint = form((gamma - 1) * eta *  phi_i * h * conj(exp(1j*self.omega*tau)) *  dx)
        self.left_form_direct_der = form((gamma - 1)  *  1j * tau * phi_i * h * eta * exp(1j*self.omega*tau) * dx)
        self.left_form_adjoint_der = form((gamma - 1)  *  tau * phi_i * h * eta * conj( 1j * exp(1j*self.omega*tau)) * dx)

        # right vector
        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        self.right_form = form(inner(n_ref,grad(phi_j)) / rho * w * dx)

    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def matrix(self):
        return self._D
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b
    
    def _assemble_left_vector(self, Derivative=False, problem_type='direct'):

        if Derivative == False:
            if problem_type == 'direct':
                form_to_assemble = self.left_form_direct
            elif problem_type == 'adjoint':
                form_to_assemble =  self.left_form_adjoint
        else:
            if problem_type == 'direct':
                form_to_assemble = self.left_form_direct_der
            elif problem_type == 'adjoint':
                form_to_assemble = self.left_form_adjoint_der

        left_vector = indices_and_values(self.dofmaps, form_to_assemble, tol=self.tol)
        
        if problem_type == 'direct':
            left_vector = distribute_vector_as_chunks(left_vector)
        elif problem_type == 'adjoint':
            left_vector = broadcast_vector(left_vector)

        return left_vector

    def _assemble_right_vector(self, problem_type='direct'):

        right_vector = indices_and_values(self.dofmaps, self.right_form, tol=self.tol)

        if problem_type == 'direct':
            right_vector = broadcast_vector(right_vector)
        elif problem_type == 'adjoint':
            right_vector = distribute_vector_as_chunks(right_vector)
        
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):

        if problem_type=='direct':
            row = [item[0] for item in self._a]
            col = [item[0] for item in self._b]

            row_vals = [item[1] for item in self._a]
            col_vals = [item[1] for item in self._b]
        
        elif problem_type=='adjoint':
            row = [item[0] for item in self._b]
            col = [item[0] for item in self._a]

            row_vals = [item[1] for item in self._b]
            col_vals = [item[1] for item in self._a]

        product = np.outer(row_vals,col_vals)
        val = product.flatten()

        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        info("- Generating matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        
        ONNZ = len(col)*np.ones(local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])

        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        info ("- Submatrix D is Assembled.")

        if problem_type == 'direct':
            self._D = mat
        elif problem_type == 'adjoint':
            self._D_adj = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        
        if problem_type == 'direct':
            self.omega.value = omega
            info("- Direct matrix D is assembling...")
        elif problem_type == 'adjoint':
            self.omega.value = omega.conjugate()
            info("- Adjoint matrix D is assembling...")
        
        self._a = self._assemble_left_vector(problem_type=problem_type)
        self._b = self._assemble_right_vector(problem_type=problem_type)
        # print(self._a)
        self.assemble_submatrices(problem_type)
        info("- Matrix D is assembled.")

    def get_derivative(self, problem_type='direct'):

        self._a = self._assemble_left_vector(Derivative=True,problem_type=problem_type)
        self._b = self._assemble_right_vector(problem_type=problem_type)
        self.assemble_submatrices(problem_type)
        info("- Derivative of matrix D is assembled.")
        return self._D
    
    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':

            D_kj_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_kj = D_kj_bloch

        elif problem_type == 'adjoint':

            D_kj_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_kj_adj = D_kj_adj_bloch

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

def indices_and_values(dofmaps, form, tol=1e-5):

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