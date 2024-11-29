import numpy as np
from math import factorial
from ufl import exp

class nTau:
    def __init__(self, n, tau):
        self.n = n
        self.tau = tau

    def __call__(self, omega):
        return self.n * exp(1j * omega * self.tau)
    
    def derivative(self, omega):
        return self.n * (1j * self.tau) * exp(1j * omega * self.tau)

class stateSpace:
    def __init__(self, S1, s2, s3, s4):
        self.A = S1
        self.b = s2
        self.c = s3
        self.d = s4
    
        self.Id = np.eye(*S1.shape)

    def __call__(self, omega):
        k = 0
        omega = np.conj(omega)
        Mat = (- 1j) ** k * factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat)
        H = np.dot(row, self.b)
        H += self.d
        return np.conj(H[0][0])
    
    def derivative(self, omega):
        k = 1
        omega = np.conj(omega)
        Mat = (- 1j) ** k * factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat)
        H = np.dot(row, self.b)
        return np.conj(H[0][0])