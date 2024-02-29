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

class ActiveFlame:

    def __init__(self, mesh, subdomains, x_r, rho_u, Q, u_b, FTF, degree=1, bloch_object=None, gamma=1.4):

        self.mesh = mesh
        self.subdomains = subdomains
        self.x_r = x_r
        self.FTF = FTF
        self.bloch_object = bloch_object
        self.coeff = (gamma - 1) * Q / (u_b * rho_u)

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

        # Utility objects for flame matrix
        self._a = {}
        self._b = {}
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        # Starting assembly for the flames
        for fl, x in enumerate(self.x_r):
            self._a[str(fl)] = self._assemble_left_vector(fl)
            self._b[str(fl)] = self._assemble_right_vector(x)

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
        """
        Assembles \int v(x) \phi_k dV
        Parameters
        ----------
        fl : int
            flame tag
        Returns
        -------
        v : <class 'list'>
            includes assembled nonzero elements of a and row indices
        """

        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*self.dx(fl))
        V_fl = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        gradient_form = form(inner(1/V_fl, self.phi_j)*self.dx(fl))

        left_vector = assemble_vector(gradient_form)
        left_vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        left_vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        indices = np.array(np.flatnonzero(left_vector.array),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, left_vector.array[indices]))

        return packed

    def _assemble_right_vector(self, point):
        """
        Calculates degree of freedoms and column indices of 
        right vector of
        
        \nabla(\phi_j(x_r)) . n
        
        which includes gradient value of test function at
        reference point x_r
        
        Parameters
        ----------
        x : np.array
            flame location vector
        Returns
        -------
        np.array
            Array of degree of freedoms and indices as vector b.
        """
        tdim = self.mesh.topology.dim

        n_ref = np.array([[0, 0, 1]]).T
        if tdim == 1:
            n_ref = np.array([[1]])
        elif tdim == 2:
            n_ref = np.array([[1, 0]]).T

        _, _, owning_points, cell = dolfinx.cpp.geometry.determine_point_ownership( self.mesh._cpp_object, point, 1e-10)

        # Data required for pull back of coordinate
        num_dofs_x = self.mesh.geometry.dofmap[0].size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(tdim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        x_dofs = self.mesh.geometry.dofmap.reshape(num_cells, num_dofs_x)
        
        point_ref = np.zeros((len(cell), self.mesh.geometry.dim), dtype=self.mesh.geometry.x.dtype)

        right_vector = []

        # Only add contribution if cell is owned 
        if len(cell) > 0:

            cell = cell[0]
            cell_geometry = self.mesh.geometry.x[x_dofs[cell], :self.gdim]
            point_ref = self.mesh.geometry.cmaps[0].pull_back([point[:self.gdim]], cell_geometry)
            dphi = self.coordinate_element.tabulate(1, point_ref)[1:,0,:]
            dphi = dphi.reshape((dphi.shape[0], dphi.shape[1]))
            
            J = np.dot(cell_geometry.T, dphi.T)
            Jinv = np.linalg.inv(J)  

            cell_dofs = self.dofmaps.cell_dofs(cell)
            global_dofs = self.dofmaps.index_map.local_to_global(cell_dofs)
            d_dx = (Jinv.T @ dphi).T
            d_dphi_j = np.dot(d_dx, n_ref)[:, 0]
            for i in range(len(d_dphi_j)):
                right_vector.append([global_dofs[i], d_dphi_j[i]])

        right_vector = self.comm.gather(right_vector, root=0)
        if right_vector:
            right_vector = [j for i in right_vector for j in i]
        else:
            right_vector=[]
        right_vector = self.comm.bcast(right_vector,root=0)

        return right_vector

    @staticmethod
    def _csr_matrix(a, b):
        """ This function gets row, column and values of (row.col) pairs of vector a and b

        Args:
            a ([type]): [description]
            b ([type]): [description]

        Returns:
            [type]: rows columns and values
        """
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
        """
        This function handles efficient cross product of the 
        vectors a and b calculated above and generates highly sparse 
        matrix D_kj which represents active flame matrix without FTF and
        other constant multiplications.
        Parameters
        ----------
        problem_type : str, optional
            Specified problem type. The default is 'direct'.
            Matrix can be obtained by selecting problem type, other
            option is adjoint.
        
        """

        row = dict()
        col = dict()
        val = dict()

        for fl, point_contribution in enumerate(self.x_r):

            u = None
            v = None

            if problem_type == 'direct':
                u = self._a[str(fl)]
                v = self._b[str(fl)]

            elif problem_type == 'adjoint':
                u = self._b[str(fl)]
                v = self._a[str(fl)]

            row[str(fl)], col[str(fl)], val[str(fl)] = self._csr_matrix(u, v)

        row = np.concatenate([row[str(fl)] for fl in range(len(self.x_r))])
        col = np.concatenate([col[str(fl)] for fl in range(len(self.x_r))])
        val = np.concatenate([val[str(fl)] for fl in range(len(self.x_r))])

        i = np.argsort(row)

        row, col, val = row[i], col[i], val[i]

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(self.local_size, self.global_size), (self.local_size, self.global_size)])
        mat.setType('aij') 
        mat.setUp()
        for i in range(len(row)):
            mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        if problem_type == 'direct':
            self._D_kj = mat
        elif problem_type == 'adjoint':
            self._D_kj_adj = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        """
        This function handles the multiplication of the obtained matrix D
        with Flame Transfer Function and other constants.
        The operation is
        
        D = (gamma - 1) / rho_u * Q / U * D_kj       
        
        At the end the matrix D is petsc4py.PETSc.Mat
        Parameters
        ----------
        omega : complex
            eigenvalue that found by solver
        problem_type : str, optional
            Specified problem type. The default is 'direct'.
            Matrix can be obtained by selecting problem type, other
            option is adjoint.
        Returns
        -------
        petsc4py.PETSc.Mat
        """

        if problem_type == 'direct':

            z = self.FTF(omega)
            self._D = self._D_kj*z*self.coeff
                        
        elif problem_type == 'adjoint':

            z = np.conj(self.FTF(np.conj(omega)))
            self._D_adj = self.coeff * z * self._D_kj_adj

    def get_derivative(self, omega):

        """ Derivative of the unsteady heat release (flame) operator D
        wrt the eigenvalue (complex angular frequency) omega."""

        z = self.FTF(omega, k=1)
        dD_domega = z * self._D_kj
        dD_domega = self.coeff * dD_domega

        return dD_domega

    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':

            D_kj_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_kj = D_kj_bloch

        elif problem_type == 'adjoint':

            D_kj_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_kj_adj = D_kj_adj_bloch

class ActiveFlameNT:

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

        if gamma: # Constant gamma [PRF paper (2018, Matthew Juniper)] 
            self.gamma = gamma 
        
        else: # Variable gamma [LOTAN]
            self.gamma = gamma_function(self.T) 

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dx = dx

        # left vector

        coefficient = (self.gamma - 1) 
        self.form_direct = form(coefficient *  self.phi_i  * self.eta * self.h * exp(1j*self.omega*self.tau)  *  self.dx)
        self.form_adjoint = form(coefficient *  self.phi_i  * self.eta * self.h * conj(exp(1j*self.omega*self.tau)) *  self.dx)
        self.form_direct_der = form(coefficient *  1j * self.tau * self.phi_i * self.h * self.eta * exp(1j*self.omega*self.tau) * self.dx)
        self.form_adjoint_der = form(coefficient *  self.tau * self.phi_i * self.h * self.eta * conj( 1j * exp(1j*self.omega*self.tau)) * self.dx)

        # right vector

        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        self.gradient_form = form(inner(n_ref,grad(self.phi_j)) / self.rho * self.w * self.dx)

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

    def _indices_and_values(self, form):

        temp = assemble_vector(form)
        temp.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        temp.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        packed = temp.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))
        return packed

    def _distribute_vector_as_chunks(self, vector):
        
        vector = self.comm.gather(vector, root=0)
        if self.rank == 0:
            vector = [j for i in vector for j in i]
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(vector):
                chunks[i % self.size].append(chunk)
        else:
            vector = None
            chunks = None
        vector = self.comm.scatter(chunks, root=0)
        
        return vector
    
    def _broadcast_vector(self, vector):
        
        vector = self.comm.gather(vector, root=0)
        if vector:
            vector = [j for i in vector for j in i]
        else:
            vector=[]
        vector = self.comm.bcast(vector,root=0)

        return vector

    def _assemble_left_vector(self, Derivative=False, problem_type='direct'):

        if Derivative == False:
            if problem_type == 'direct':
                form_to_assemble = self.form_direct
            elif problem_type == 'adjoint':
                form_to_assemble =  self.form_adjoint
        else:
            if problem_type == 'direct':
                form_to_assemble = self.form_direct_der
            elif problem_type == 'adjoint':
                form_to_assemble = self.form_adjoint_der

        left_vector = self._indices_and_values(form_to_assemble)
        
        if problem_type == 'direct':
            left_vector = self._distribute_vector_as_chunks(left_vector)
        elif problem_type == 'adjoint':
            left_vector = self._broadcast_vector(left_vector)

        return left_vector

    def _assemble_right_vector(self, problem_type='direct'):

        right_vector = self._indices_and_values(self.gradient_form)

        if problem_type == 'direct':
            right_vector = self._broadcast_vector(right_vector)
        elif problem_type == 'adjoint':
            right_vector = self._distribute_vector_as_chunks(right_vector)
        
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

    def get_derivative(self, omega, problem_type='direct'):

        self._a = self._assemble_left_vector(Derivative=True,problem_type=problem_type)
        self._b = self._assemble_right_vector(problem_type=problem_type)
        self.assemble_submatrices(problem_type)
        info("- Derivative of matrix D is assembled.")
        return self._D