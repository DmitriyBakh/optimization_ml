import datetime
import sys

from collections import defaultdict
from time import time

import numpy as np
import scipy
from numpy.linalg import norm, solve
from scipy.special import expit


assert sys.version_info >= (3, 6), (
    "Please use Python3.6+ to make this assignment"
)


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
    def phi(t, x, u, A, b):
        Ax_b = A.dot(x) - b
        return t * (0.5 * np.linalg.norm(Ax_b) ** 2 + reg_coef * np.sum(u)) - np.sum(np.log(u + x) + np.log(u - x))

    def grad_phi(t, x, u, A, b):
        Ax_b = A.dot(x) - b
        ATAx_b = A.T.dot(Ax_b)
        return t * (ATAx_b + reg_coef * np.sign(u)) + (1 / (u + x) - 1 / (u - x))

    def hessian_phi(t, x, u, A, b):
        AT_A = A.T.dot(A)
        return t * AT_A + np.diag(1 / ((u + x) * (u - x)))

    def backtracking_line_search(t, x, u, A, b, descent_dir, c1=0.01, max_iter_inner=100):
        alpha = 1
        x_size = len(x)

        for _ in range(max_iter_inner):
            x_new = x - alpha * descent_dir[:x_size]
            u_new = u - alpha * descent_dir[x_size:]

            if np.min(u_new) > 0 and phi(t, x_new, u_new, A, b) <= phi(t, x, u, A, b) + c1 * alpha * np.dot(grad_phi(t, x, u, A, b), -descent_dir):
                break
            else:
                alpha *= 0.5
        return alpha

    if trace:
        history = defaultdict(list)

    x = x_0.copy()
    u = u_0.copy()
    t = t_0

    for k in range(max_iter):
        start_time = time()
        for _ in range(max_iter_inner):
            grad = grad_phi(t, x, u, A, b)
            hessian = hessian_phi(t, x, u, A, b)
            try:
                descent_dir = -solve(hessian, grad)
            except np.linalg.LinAlgError:
                return x, u, 'computational_error', history if trace else None
            if norm(descent_dir) ** 2 <= tolerance_inner * norm(grad) ** 2:
                break
            step_size = backtracking_line_search(t, x, u, A, b, descent_dir, c1, max_iter_inner)
            x -= step_size * descent_dir[:x.size]
            u -= step_size * descent_dir[x.size:]
        
        Ax_b = A.dot(x) - b
        ATAx_b = A.T.dot(Ax_b)
        if lasso_duality_gap:
            duality_gap = lasso_duality_gap(x, Ax_b, ATAx_b, b, reg_coef)
            if trace:
                history['duality_gap'].append(duality_gap)
            if duality_gap <= tolerance:
                return x, u, 'success', history if trace else None
        
        if trace:
            history['time'].append(time() - start_time)
            history['func'].append(0.5 * np.linalg.norm(Ax_b) ** 2 + reg_coef * np.sum(u))
            if x.size <= 2:
                history

            history['x'].append(x.copy())

        if display:
            print(f'Iteration {k + 1}: Duality gap = {duality_gap}')

        t *= gamma
    return x, u, 'iterations_exceeded', history if trace else None


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    # TODO: implement.
    # raise NotImplementedError
    # n = len(x)
    # dual_scale = np.dot(Ax_b, b) / (regcoef * n)
    
    # if dual_scale < 0:
    #     dual_scale = 0
    
    # y = dual_scale * Ax_b
    # primal = 0.5 * np.dot(Ax_b, Ax_b) + regcoef * np.sum(np.abs(x))
    # dual = 0.5 * np.dot(y - b, y - b)
    
    # return primal - dual

    f = 0.5 * Ax_b @ Ax_b + regcoef * np.sum(x)
    mu = np.min([1, regcoef / np.max(np.abs(ATAx_b))]) * Ax_b
    return f + 0.5 * mu @ mu + b @ mu    
