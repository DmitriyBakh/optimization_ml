import sys
# import time
from time import time
from datetime import datetime

from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS

import numpy as np
import scipy.sparse
from numpy.linalg import LinAlgError
from scipy.special import expit
from scipy.optimize.linesearch import scalar_search_wolfe2


#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


assert sys.version_info >= (3, 6), (
    "Please use Python3.6+ to make this assignment"
)


class LineSearchTool:
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        if previous_alpha is not None:
            self.alpha_0 = previous_alpha

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        dphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        if self._method == 'Wolfe':
            # alpha = scipy.optimize.linesearch.scalar_search_wolfe2(phi, dphi, c1=self.c1, c2=self.c2)[0]
            alpha = scalar_search_wolfe2(phi, dphi, c1=self.c1, c2=self.c2)[0]
            if alpha is None:
                return LineSearchTool(method='Armijo', c1=self.c1, alpha_0=self.alpha_0).line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Armijo':
            alpha = self.alpha_0
            while phi(alpha) > phi(0) + self.c1 * alpha * dphi(0):
                alpha = alpha / 2
        elif self._method == 'Constant':
            alpha = self.c

        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory BroydenвЂ“FletcherвЂ“GoldfarbвЂ“Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement L-BFGS method.
    # Use line_search_tool.line_search() for adaptive step size.
    def update_history():
        if not trace:
            return
        history['func'].append(oracle.func(x_k))
        history['time'].append((datetime.now() - t_0).seconds)
        history['grad_norm'].append(grad_k_norm)
        if x_len <= 2:
            history['x'].append(np.copy(x_k))

    def show_display():
        if not display:
            return
        if len(x_k) <= 4:
            print('x = {}, '.format(np.round(x_k, 4)), end='')
        print('func= {}, grad_norm = {}'.format(np.round(oracle.func(x_k), 4), np.round(grad_k_norm, 4)))

    t_0 = datetime.now()
    x_len = len(x_k)
    message = 'success'

    grad_k = oracle.grad(x_k)
    grad_0_norm = grad_k_norm = np.linalg.norm(grad_k)

    def bfgs_multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        s, y = H[-1]
        H = H[:-1]
        v_new = v - (s @ v) / (y @ s) * y
        z = bfgs_multiply(v_new, H, gamma_0)
        result = z + (s @ v - y @ z) / (y @ s) * s
        return result

    def bfgs_direction():
        if len(H) == 0:
            return -grad_k
        s, y = H[-1]
        gamma_0 = (y @ s) / (y @ y)
        return bfgs_multiply(-grad_k, H, gamma_0)

    H = []
    for k in range(max_iter):
        show_display()
        update_history()

        d = bfgs_direction()
        alpha = line_search_tool.line_search(oracle, x_k, d)
        x_new = x_k + alpha * d
        grad_new = oracle.grad(x_new)
        H.append((x_new - x_k, grad_new - grad_k))
        if len(H) > memory_size:
            H = H[1:]
        x_k, grad_k = x_new, grad_new
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
            break

    show_display()
    update_history()

    if not grad_k_norm ** 2 < tolerance * grad_0_norm ** 2:
        message = 'iterations_exceeded'

    return x_k, message, history


#######################################################
#                                                     #
#                     ORACLES                         #
#                                                     #
#######################################################


class BaseSmoothOracle:
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        m = len(self.b)
        return (np.logaddexp(0, -self.b * self.matvec_Ax(x)).dot(np.ones(m)) / m +
                self.regcoef / 2 * np.linalg.norm(x) ** 2)

    def grad(self, x):
        # TODO: Implement
        m = len(self.b)
        sigma = expit(self.b * self.matvec_Ax(x))
        return self.regcoef * x - self.matvec_ATx((1 - sigma) * self.b) / m

    def hess(self, x):
        # TODO: Implement
        m = len(self.b)
        n = len(x)
        sigma = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(sigma * (1 - sigma)) / m + self.regcoef * np.eye(n)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # def matvec_Ax(x):
    #     # TODO: implement proper matrix-vector multiplication
    #     return x

    # def matvec_ATx(x):
    #     # TODO: implement proper martix-vector multiplication
    #     return x

    # def matmat_ATsA(s):
    #     # TODO: Implement
    #     return None
    if scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        matmat_ATsA = lambda x: matvec_ATx(matvec_ATx(scipy.sparse.diags(x)).T)
    else:
        matvec_Ax = lambda x: np.dot(A, x)
        matvec_ATx = lambda x: np.dot(A.T, x)
        matmat_ATsA = lambda s: np.dot(A.T, np.dot(np.diag(s), A))

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    n = len(x)
    E = np.eye(n)
    result = np.zeros(n)
    for i in range(n):
        result[i] = (func(x + eps * E[i]) - func(x)) / eps
    return result


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    n = len(x)
    E = np.eye(n)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            result[i, j] = (func(x + eps * E[i] + eps * E[j]) - func(x + eps * E[i]) - func(x + eps * E[j])
                            + func(x)) / eps ** 2
            result[j, i] = result[i, j]
    return result
