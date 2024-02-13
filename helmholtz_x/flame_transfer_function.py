"""
    flame transfer functions:

    n-tau model,
    state-space representation
"""

import numpy as np

def n_tau(N3, tau):
    """
    :param N3: non-dimensional interaction index
    :param tau: time delay [s]
    :return: function
    """
    def inner_func(omega, k=0):
        return N3 * (1j * tau)**k * np.exp(1j * omega * tau)
    return inner_func

def time_delay(tau):
    """
    :omega: angular eigenfrequency [rad/s]
    :param tau: time delay [s]
    :return: function
    """
    def inner_func(omega, derivative_deg=0):
        return (1j * tau)**derivative_deg * np.exp(1j * omega * tau)
    return inner_func

def state_space(A, b, c, d):
    """
    vectfit3.m is written using the expansion e^{i omega t},
    while the Helmholtz solver is written using e^{- i omega t}.
    omega in vectfit3.m is the complex conjugate of omega in the Helmholtz solver
    and the same goes for the flame transfer function.

    :param A: (N, N) array
    :param b: (N, 1) array
    :param c: (1, N) array
    :param d: (1, 1) array
    :return: function
    """
    Id = np.eye(*A.shape)

    def inner_func(omega, k=0):
        """
        :param omega: complex angular frequency [rad/s]
        :param k: k-th order derivative
        :return: transfer function/frequency response
        """
        omega = np.conj(omega)
        Mat = (- 1j) ** k * np.math.factorial(k) * \
            np.linalg.matrix_power(1j * omega * Id - A, - (k + 1))
        row = np.dot(c, Mat)
        H = np.dot(row, b)
        if k == 0:
            H += d
        return np.conj(H[0][0])
    return inner_func