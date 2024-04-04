from .petsc4py_utils import vector_matrix_vector
from .eigenvectors import normalize_eigenvector
from .solver_utils import info
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np

def results(E):
    if MPI.COMM_WORLD.Get_rank()==0:
        print()
        print("******************************")
        print("*** SLEPc Solution Results ***")
        print("******************************")
        print()

        its = E.getIterationNumber()
        print("Number of iterations of the method: %d" % its)

        eps_type = E.getType()
        print("Solution method: %s" % eps_type)

        nev, ncv, mpd = E.getDimensions()
        print("Number of requested eigenvalues: %d" % nev)

        tol, maxit = E.getTolerances()
        print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

        nconv = E.getConverged()
        print("Number of converged eigenpairs %d" % nconv)

        A = E.getOperators()[0]
        vr, vi = A.createVecs()

        if nconv > 0:
            print()
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            print("%15f, %15f" % (k.real, k.imag))
        print()

def eps_solver(A, C, target, nev, two_sided=False, print_results=False):

    E = SLEPc.EPS().create(MPI.COMM_WORLD)

    C = - C
    E.setOperators(A, C)

    # spectral transformation
    st = E.getST()
    st.setType('sinvert')
    # E.setKrylovSchurPartitions(1) # MPI.COMM_WORLD.Get_size()

    eps_target = target**2 
    E.setTarget(eps_target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
    E.setTwoSided(two_sided)

    E.setDimensions(nev, SLEPc.DECIDE)
    E.setTolerances(1e-15)
    E.setFromOptions()
    info("- EPS solver started.")
    E.solve()
    info("- EPS solver converged. Eigenvalue computed.")
    if print_results and MPI.COMM_WORLD.rank == 0:
        results(E)

    return E

def pep_solver(A, B, C, target, nev, print_results=False):
    """
    This function defines solved instance for
    A + wB + w^2 C = 0

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Matrix of Grad term
    B : petsc4py.PETSc.Mat
        Empty Matrix
    C : petsc4py.PETSc.Mat
        Matrix of w^2 term.
    target : float
        targeted eigenvalue
    nev : int
        Requested number of eigenvalue
    print_results : boolean, optional
        Prints the results. The default is False.

    Returns
    -------
    Q : slepc4py.SLEPc.PEP
        Solution instance of eigenvalue problem.

    """

    Q = SLEPc.PEP().create(MPI.COMM_WORLD)

    operators = [A, B, C]
    Q.setOperators(operators)

    # spectral transformation
    st = Q.getST()
    st.setType('sinvert')

    Q.setTarget(target)
    Q.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
    Q.setDimensions(nev, SLEPc.DECIDE)
    Q.setTolerances(1e-15)
    Q.setFromOptions()

    info("- PEP solver started.")

    Q.solve()

    info("- PEP solver converged. Eigenvalue computed.")

    if print_results and MPI.COMM_WORLD.rank == 0:
        results(Q)

    return Q

def fixed_point_iteration_eps(operators, D, target, nev=2, i=0,
                              tol=1e-8, maxiter=50,
                              print_results=False,
                              problem_type='direct',
                              two_sided=False):

    A = operators.A
    B = operators.B
    C = operators.C
    if problem_type == 'adjoint':
        B = operators.B_adj

    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)

    info("--> Fixed point iteration started.\n")

    E = eps_solver(A, C, target, nev, print_results=print_results)
    eig = E.getEigenvalue(i)
    
    omega[0] = np.sqrt(eig)
    alpha[0] = 0.5

    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    if MPI.COMM_WORLD.rank == 0:
        print("+ Starting eigenvalue is found: {}  {}j. ".format(
                 s.format(omega[k + 1].real), s.format(omega[k + 1].imag)))
    info("-> Iterations are starting.\n ")
    while abs(domega) > tol:

        k += 1
        E.destroy()
        if MPI.COMM_WORLD.rank == 0:
            print("* iter = {:2d}".format(k+1))

        D.assemble_matrix(omega[k], problem_type)
        D_Mat = D.matrix
        if problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix

        if not B:
            D_Mat = A - D_Mat
        else:
            D_Mat = A + (omega[k] * B) - D_Mat

        E = eps_solver(D_Mat, C, target, nev, two_sided=two_sided, print_results=print_results)
        del D_Mat
        eig = E.getEigenvalue(i)

        f[k] = np.sqrt(eig)

        if k != 0:
            alpha[k] = 1/(1 - ((f[k] - f[k-1])/(omega[k] - omega[k-1])))
            
        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]
        if MPI.COMM_WORLD.rank == 0:
            print('+ omega = {}  {}j,  |domega| = {:.2e}\n'.format(
                 s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)
            ))

    return E

def fixed_point_iteration_pep( operators, D,  target, nev=2, i=0,
                                    tol=1e-8, maxiter=50,
                                    print_results=False,
                                    problem_type='direct'):
    A = operators.A
    C = operators.C
    B = operators.B
    if problem_type == 'adjoint':
        B = operators.B_adj

    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)
    E = pep_solver(A, B, C, target, nev, print_results=print_results)
    vr, vi = A.getVecs()
    eig = E.getEigenpair(i, vr, vi)
    omega[0] = eig
    alpha[0] = .5

    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)
    
    info("-> Fixed point iteration started.\n")

    while abs(domega) > tol:

        k += 1
        E.destroy()
        if MPI.COMM_WORLD.rank == 0:
            print("* iter = {:2d}".format(k+1))
        D.assemble_matrix(omega[k], problem_type)
        if problem_type == 'direct':
            D_Mat = D.matrix
        if problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix

        D_Mat = A - D_Mat

        E = pep_solver(D_Mat, B, C, target, nev, print_results=print_results)

        D_Mat.destroy()
        eig = E.getEigenpair(i, vr, vi)
        f[k] = eig

        if k != 0:
            alpha[k] = 1 / (1 - ((f[k] - f[k-1]) / (omega[k] - omega[k-1])))

        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]
        if MPI.COMM_WORLD.rank == 0:
            print('+ omega = {}  {}j,  |domega| = {:.2e}\n'.format(
                 s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)
            ))

    return E

def fixed_point_iteration( operators, D,  target, nev=2, i=0,
                                    tol=1e-8, maxiter=50,
                                    print_results=False,
                                    problem_type='direct'):
    if operators.B:
        E = fixed_point_iteration_pep( operators, D,  target, nev=nev, i=i,
                                    tol=tol, maxiter=maxiter,
                                    print_results=print_results,
                                    problem_type=problem_type)
    else:
        E = fixed_point_iteration_eps( operators, D,  target, nev=nev, i=i,
                                    tol=tol, maxiter=maxiter,
                                    print_results=print_results,
                                    problem_type=problem_type)
    
    return E

def newtonSolver(operators, D, init, nev=2, i=0, tol=1e-3, maxiter=100, print_results=False):
    """
    The convergence strongly depends/relies on the initial value assigned to omega.
    Targeting zero in the shift-and-invert (spectral) transformation or, more in general,
    seeking for the eigenvalues nearest to zero might also be problematic.
    The implementation uses the TwoSided option to compute the adjoint eigenvector.
    """

    A = operators.A
    C = operators.C
    B = operators.B

    omega = np.zeros(maxiter, dtype=complex)
    omega[0] = init

    domega = 2 * tol
    k = 0

    # formatting
    tol_ = "{:.0e}".format(tol)
    tol_ = int(tol_[-2:])
    s = "{{:+.{}f}}".format(tol_)

    relaxation = 1.0

    info("-> Newton solver started.\n")

    while abs(domega) > tol:

        D.assemble_matrix(omega[k])
        if not B:
            L = A + omega[k] ** 2 * C - D.matrix
        
            dL_domega = 2 * omega[k] * C - D.get_derivative(omega[k])
        else:
            L = A + omega[k] * B + omega[k]** 2 * C  - D.matrix
                
            dL_domega = B + (2 * omega[k] * C) - D.get_derivative(omega[k])

        # solve the eigenvalue problem L(\omega) * p = \lambda * C * p
        # set the target to zero (shift-and-invert)
        E = eps_solver(L, - C, 0, nev, two_sided=True, print_results=print_results)

        eig = E.getEigenvalue(i)

        omega_dir, p = normalize_eigenvector(operators.mesh, E, i, degree=1, which='right', print_eigs=False)

        omega_adj, p_adj = normalize_eigenvector(operators.mesh, E, i, degree=1, which='left', print_eigs=False)

        # convert into PETSc.Vec type
        p_vec = p.vector
        p_adj_vec = p_adj.vector

        # numerator and denominator
        num = vector_matrix_vector(p_adj_vec, dL_domega, p_vec)
        den = vector_matrix_vector(p_adj_vec, C, p_vec)

        deig = num / den
        domega = - relaxation * eig / deig
        relaxation *= 0.8

        omega[k + 1] = omega[k] + domega
        
        if MPI.COMM_WORLD.rank == 0:
            print('iter = {:2d},  omega = {}  {}j,  |domega| = {:.2e}'.format(
                    k, s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)))

        k += 1

        del E

    return omega[k], p