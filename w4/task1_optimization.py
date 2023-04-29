from datetime import datetime
from collections import defaultdict

import numpy as np
import scipy.sparse
from numpy.linalg import LinAlgError
from scipy.special import expit

import time


#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


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

        if previous_alpha is not None:
            self.alpha_0 = previous_alpha

        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        dphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        if self._method == 'Wolfe':
            alpha = scipy.optimize.linesearch.scalar_search_wolfe2(phi, dphi, c1=self.c1, c2=self.c2)[0]
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


def update_history(trace, display, history, oracle, time, x_k, i):
    if display:
        print("Iteration ", i, ": x_k = ", x_k, sep='')
    if trace:
        history['time'].append(time)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k)))
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))
    return history

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> line_search_options = {'method': 'Armijo', 'c1': 1e-4}
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options=line_search_options)
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)


    it = 0
    start_time = time.time()
    history = update_history(trace, display, history, oracle, 0, x_k, it)
    grad_0 = oracle.grad(x_0)

    while True:
        grad_k = oracle.grad(x_k)
        if np.linalg.norm(grad_k) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history

        it += 1
        d_k = -grad_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        # x_k += alpha * d_k
        x_k = x_k + alpha * d_k

        history = update_history(trace, display, history, oracle, time.time() - start_time, x_k, it)

        if (None in x_k) or (x_k > 10 ** 9).any():
            return x_k, 'computational_error', history

        if it > max_iter:
            return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix
                (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient
                on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    it = 0
    start_time = time.time()
    history = update_history(trace, display, history, oracle, 0, x_k, it)
    grad_0 = oracle.grad(x_0)

    while True:
        grad_k = oracle.grad(x_k)
        if np.linalg.norm(grad_k) ** 2 <= tolerance * np.linalg.norm(grad_0) ** 2:
            return x_k, 'success', history

        try:
            c, lower = scipy.linalg.cho_factor(oracle.hess(x_k))
            d_k = scipy.linalg.cho_solve((c, lower), -grad_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha * d_k
        it += 1

        history = update_history(trace, display, history, oracle, time.time() - start_time, x_k, it)

        if (None in x_k) or (x_k > 10 ** 9).any():
            return x_k, 'computational_error', history

        if it > max_iter:
            return x_k, 'iterations_exceeded', history



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
        m = len(self.b)
        return (np.logaddexp(0, -self.b * self.matvec_Ax(x)).dot(np.ones(m)) / m +
                self.regcoef / 2 * np.linalg.norm(x) ** 2)


    def grad(self, x):
        m = len(self.b)
        sigma = expit(self.b * self.matvec_Ax(x))
        return self.regcoef * x - self.matvec_ATx((1 - sigma) * self.b) / m


    def hess(self, x):
        m = len(self.b)
        n = len(x)
        sigma = expit(self.b * self.matvec_Ax(x))
        return self.matmat_ATsA(sigma * (1 - sigma)) / m + self.regcoef * np.eye(n)


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).
    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.x = None
        self.d = None
        self.xhat = None                # x_hat = x + alpha * d
        self.A_xhat = None              # A_xhat = Ax + alpha * Ad
        self.Ad = None
        self.Ax = None
        self.ATx = None


    def update_Ax(self, x):
        if np.all(x == self.x):
            return

        self.x = x
        self.Ax = self.matvec_Ax(x)

    def update_Ad(self, d):
        if np.all(d == self.d):
            return

        self.d = d
        self.Ad = self.matvec_Ax(d)

    def update_xhat(self, x, alpha, d):
        if np.all(self.xhat == x + alpha * d):
            return

        self.xhat = x + alpha * d
        self.A_xhat = self.Ax + alpha * self.Ad

    def func(self, x):
        m = len(self.b)

        # last point in task
        if np.all(self.xhat == x):
            in_log = - self.b * self.A_xhat
            loss = np.logaddexp(0, in_log)
            return (np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(x) ** 2

        self.update_Ax(x)
        in_log = - self.b * self.Ax
        loss = np.logaddexp(0, in_log)
        return (np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(x) ** 2

    def grad(self, x):
        m = len(self.b)

        if np.all(self.xhat == x):
            sigmoid_and_label = scipy.special.expit(self.A_xhat) - (self.b + 1) / 2
            res = self.matvec_ATx(sigmoid_and_label) / m
            return res + self.regcoef * x

        self.update_Ax(x)
        sigmoid_and_label = scipy.special.expit(self.Ax) - (self.b + 1) / 2
        res = self.matvec_ATx(sigmoid_and_label) / m
        return res + self.regcoef * x

    def hess(self, x):
        m = len(self.b)
        n = len(x)

        if np.all(self.xhat == x):
            sigmoid_der = scipy.special.expit(self.A_xhat) * (1 - scipy.special.expit(self.A_xhat))
            res = self.matmat_ATsA(sigmoid_der)
            return res / m + self.regcoef * np.eye(n)

        self.update_Ax(x)
        sigmoid_der = scipy.special.expit(self.Ax) * (1 - scipy.special.expit(self.Ax))
        res = self.matmat_ATsA(sigmoid_der)
        return res / m + self.regcoef * np.eye(n)


    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        m = len(self.b)

        self.update_Ad(d)
        self.update_Ax(x)
        self.update_xhat(x, alpha, d)

        in_log = - self.b * self.A_xhat
        loss = np.logaddexp(0, in_log)
        return np.squeeze((np.ones(m) @ loss) / m + (self.regcoef / 2) * np.linalg.norm(self.xhat) ** 2)

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        m = len(self.b)

        self.update_Ad(d)
        self.update_Ax(x)
        self.update_xhat(x, alpha, d)

        sigmoid_and_label = scipy.special.expit(self.A_xhat) - (self.b + 1) / 2
        return (np.transpose(sigmoid_and_label) @ self.Ad / m) + self.regcoef * self.A_xhat @ d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
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
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle        
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

    n = len(x)
    E = np.eye(n)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            result[i, j] = (func(x + eps * E[i] + eps * E[j]) - func(x + eps * E[i]) - func(x + eps * E[j])
                            + func(x)) / eps ** 2
            result[j, i] = result[i, j]
    return result
