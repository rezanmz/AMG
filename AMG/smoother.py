import numpy as np
from scipy import sparse
from scipy.linalg import norm


class Smoother:
    def __init__(self, A, b, n, x=None, verbosity=True):
        self.A = A
        self.b = b
        self.n = n
        self.x = x
        self.verbosity = verbosity

    def gauss_seidel(self):
        if self.x is None:
            self.x = np.zeros(len(self.A[0]))

        L = np.tril(self.A)
        U = self.A - L
        residual = []
        for i in range(self.n):
            self.x = np.dot(np.linalg.inv(L), self.b - np.dot(U, self.x))
            residual.append(norm(self.b - self.A.dot(self.x), ord=np.inf))
            if self.verbosity:
                print(f'Iteration {i}: Norm L2 of residual is {residual[-1]}')
        return self.x, residual

    def jacobi(self):
        """
        Solves the equation Ax=b via the Jacobi iterative method.
        """
        # Create an initial guess if needed
        print('-' * 50)
        print('Jacobi numberical solver')
        if self.x is None:
            self.x = np.zeros(self.A.shape[0])

        # Create a vector of the diagonal elements of A
        # and subtract them from A
        D = self.A.diagonal()
        R = self.A - sparse.diags(D)

        residual = []
        # Iterate for n times
        for i in range(self.n):
            self.x = (self.b - R.dot(self.x)) / D
            residual.append(norm(self.b - self.A.dot(self.x), ord=np.inf))
            if self.verbosity:
                print(f'Iteration {i}: Norm L2 of residual is {residual[-1]}')
        if self.verbosity:
            print('-' * 50)
        return self.x, residual
