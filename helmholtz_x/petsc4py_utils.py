from petsc4py import PETSc
import numpy as np

def multiply(x0, z):
    """
    multiply a complex vector by a complex scalar
    """

    x1 = x0.copy()

    x1.scale(z)

    return x1

def conjugate(y0):
    """
    Takes complex conjugate of vector y0

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    y1 : petsc4py.PETSc.Vec
        Complex vector

    """
    y1 = y0.copy()
    y1.conjugate()

    return y1

def conjugate_function(p):

    p_conj = p
    p_conj.x.array[:] = np.conjugate(p_conj.x.array)
    
    return p_conj

def vector_vector(y0, x0):
    """
    it does y0^H * x0 
    y1 = y0^H(conjugate-transpose of y0)

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector
    x0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    z : Complex scalar product

    """


    y1 = y0.copy()
    y1 = y0.dot(x0)

    return y1


def vector_matrix_vector(y0, A, x0):
    """
    multiplies complex vector, matrix and complex vector 

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector
    A : petsc4py.PETSc.Mat
        Matrix.
    x0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    z : complex scalar product

    """
    x1 = x0.copy()
    A.mult(x0, x1) # x1 = A'*x0
    z = vector_vector(y0, x1)

    return z

def matrix_vector(Mat, x):
    """
    Multiplies matrix and vector for corresponding sizes
    """
    dummy, vector = Mat.createVecs()
    Mat.mult(x, vector)  # x1 = A'*x0

    return vector

def FixSign(x):
    # Force the eigenfunction to be real and positive, since
    # some eigensolvers may return the eigenvector multiplied
    # by a complex number of modulus one.
    comm = x.getComm()
    rank = comm.getRank()
    n = 1 if rank == 0 else 0
    aux = PETSc.Vec().createMPI((n, PETSc.DECIDE), comm=comm)
    if rank == 0: aux[0] = x[0]
    aux.assemble()
    x0 = aux.sum()
    sign = x0/abs(x0)
    x.scale(1.0/sign)