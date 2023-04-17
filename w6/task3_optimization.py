import sys

from collections import defaultdict
from time import time

import numpy as np
from numpy.linalg import LinAlgError
import scipy
from numpy.linalg import norm, solve
from scipy.special import expit
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.optimize import line_search


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


def update_history(trace, display, history, oracle, time, x_k, i, u_k=None, duality_gap_k=None, variant='newton'):
    if display:
        print("Iteration ", i, ": x_k = ", x_k, sep='')
    if trace:
        history['time'].append(time)

        if variant == 'newton':
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k)))
        else:
            history['func'].append(oracle.original_func(np.concatenate([x_k, u_k])))
            history['duality_gap'].append(duality_gap_k)
        
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))
    return history


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
    def get_alpha(x_concat, d_concat):
        x, u = np.array_split(x_concat, 2)
        grad_x, grad_u = np.array_split(d_concat, 2)
        alphas = [1.]
        THETA = 0.99
        for i in range(len(grad_x)):
            if grad_x[i] > grad_u[i]:
                alphas.append(THETA * (u[i] - x[i]) / (grad_x[i] - grad_u[i]))
            if grad_x[i] < -grad_u[i]:
                alphas.append(THETA * (x[i] + u[i]) / (-grad_x[i] - grad_u[i]))
        return min(alphas)
    
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    it = 0
    start_time = time()
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

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=get_alpha(x_k, d_k))
        
        x_k += alpha * d_k
        it += 1

        history = update_history(trace, display, history, oracle, time() - start_time, x_k, it)

        if (None in x_k) or (x_k > 10 ** 9).any():
            return x_k, 'computational_error', history

        if it > max_iter:
            return x_k, 'iterations_exceeded', history


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=None,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    # raise NotImplementedError
    oracle = BarrierLassoOracle(A, b, reg_coef, t_0)
    lasso_duality_gap = lasso_duality_gap if lasso_duality_gap else lasso_duality_gap
    lasso_duality_gap_ = lambda x_: lasso_duality_gap(x_, A @ x_ - b, A.T @ (A @ x_ - b), b, reg_coef)

    history = defaultdict(list) if trace else None
    x_k, u_k = np.copy(x_0), np.copy(u_0)
    start_time = time()
    t_k = t_0
    duality_gap_k = lasso_duality_gap_(x_k)
    message = None

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        update_history(trace, display, history, oracle, time() - start_time, x_k, k, u_k, duality_gap_k, variant='barrier_method_lasso')
        oracle.t = t_k
        x = np.concatenate([x_k, u_k])
        x_new, message_newton, _ = newton(oracle, x, max_iter=max_iter_inner, tolerance=tolerance_inner,
                                        line_search_options={'c1' : c1, 'method': 'Armijo'})

        x_k, u_k = np.array_split(x_new, 2)
        if message_newton == 'computational_error':
            message = message_newton
            break

        t_k *= gamma
        duality_gap_k = lasso_duality_gap_(x_k)

    update_history(trace, display, history, oracle, time() - start_time, x_k, k, u_k, duality_gap_k, variant='barrier_method_lasso')
    if not message:
        message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return (x_k, u_k), message, history


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    if np.linalg.norm(ATAx_b, ord=np.inf) < 1e-5:
        mu = Ax_b
    else:
        mu = np.min([1., regcoef / np.linalg.norm(ATAx_b, ord=np.inf)]) * Ax_b
    nu = 0.5 * np.linalg.norm(Ax_b) ** 2 + regcoef * np.linalg.norm(x, 1) + 0.5 * np.linalg.norm(mu) ** 2 + b @ mu
    
    return nu


#######################################################
#                                                     #
#                     ORACLES                         #
#                                                     #
#######################################################


class BaseSmoothOracle(object):
    """
    Base class for smooth function.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func is not implemented.')

    def grad(self, x):
        """
        Computes the gradient vector at point x.
        """
        raise NotImplementedError('Grad is not implemented.')

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
    

class BarrierLassoOracle(BaseSmoothOracle):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.
    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    """
    def __init__(self, A, b, regcoef, t):
        self.A = A
        self.matvec_Ax = lambda x: A @ x
        self.matvec_ATx = lambda x: A.T @ x
        self.b = b
        self.regcoef = regcoef
        self.t = t * 1.0

    def original_func(self, point):
        x, u = np.array_split(point, 2)
        return 0.5 * np.linalg.norm(self.matvec_Ax(x) - self.b) ** 2 + self.regcoef * np.sum(u)

    def func(self, point):
        x, u = np.array_split(point, 2)
        return self.t * self.original_func(point) - np.sum(np.log(u + x) + np.log(u - x))

    def grad(self, point):
        x, u = np.array_split(point, 2)
        grad_f_x = self.t * self.matvec_ATx(self.matvec_Ax(x) - self.b)
        grad_f_u = self.t * self.regcoef * np.ones(len(u))
        grad_bar_x = -1. / (u + x) + 1. / (u - x)
        grad_bar_u = -1. / (u + x) - 1. / (u - x)
        return np.hstack([grad_f_x + grad_bar_x, grad_f_u + grad_bar_u])

    def hess(self, point):
        x, u = np.array_split(point, 2)
        hess_xx = self.A.T @ self.A * self.t + np.diag(1. / (u - x) ** 2 + 1. / (u + x) ** 2)
        hess_xu = np.diag(1. / (u + x) ** 2 - 1. / (u - x) ** 2)
        hess_uu = np.diag(1. / (u + x) ** 2 + 1. / (u - x) ** 2)
        return np.vstack((np.hstack((hess_xx, hess_xu)), np.hstack((hess_xu, hess_uu))))
