from collections import defaultdict
from time import time

import numpy as np
import scipy
from scipy.special import expit

from datetime import datetime

#######################################################
#                                                     #
#                   OPTIMIZATION                      #
#                                                     #
#######################################################


def update_history(trace, display, history, oracle, time, x_k, i, duality_gap_k):
    if display:
        print("Iteration ", i, ": x_k = ", x_k, sep='')
    
    if trace:
        history['time'].append(time)
        history['func'].append(oracle.func(x_k))
        history['duality_gap'].append(duality_gap_k)
        
        if len(x_k) <= 2:
            history['x'].append(np.copy(x_k))

    return history


def proximal_gradient_method(oracle, x_0, L_0=1, tolerance=1e-5,
                             max_iter=1000, trace=False, display=False):
    """
    Gradient method for composite optimization.

    Parameters
    ----------
    oracle : BaseCompositeOracle-descendant object
        Oracle with .func() and .grad() and .prox() methods implemented
        for computing function value, its gradient and proximal mapping
        respectively.
        If available, .duality_gap() method is used for estimating f_k - f*.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    L_0 : float
        Initial value for adaptive line-search.
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
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
            - history['func'] : list of objective function values phi(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    # TODO: implement.
    history = defaultdict(list) if trace else None
    x_k, L_k = np.copy(x_0), L_0
    start_time = datetime.now()
    grad_k = oracle.grad(x_k)
    duality_gap_k = oracle.duality_gap(x_k)

    for k in range(max_iter):
        if duality_gap_k < tolerance:
            break

        update_history(trace, display, history, oracle, datetime.now() - start_time,
                       x_k, k, duality_gap_k)

        while True:
            x_new = oracle.prox(x_k - grad_k / L_k, 1 / L_k)
            if oracle._f.func(x_new) > oracle._f.func(x_k) + grad_k @ (x_new - x_k) +\
                    L_k / 2 * np.linalg.norm(x_new - x_k) ** 2:
                L_k *= 2
            else:
                x_k = x_new
                break

        L_k /= 2
        grad_k = oracle.grad(x_k)
        duality_gap_k = oracle.duality_gap(x_k)

    update_history(trace, display, history, oracle, datetime.now() - start_time,
                   x_k, k, duality_gap_k)
    
    message = 'success' if duality_gap_k < tolerance else 'iterations_exceeded'
    return x_k, message, history



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


class BaseProxOracle(object):
    """
    Base class for proximal h(x)-part in a composite function f(x) + h(x).
    """

    def func(self, x):
        """
        Computes the value of h(x).
        """
        raise NotImplementedError('Func is not implemented.')

    def prox(self, x, alpha):
        """
        Implementation of proximal mapping.
        prox_{alpha}(x) := argmin_y { 1/(2*alpha) * ||y - x||_2^2 + h(y) }.
        """
        raise NotImplementedError('Prox is not implemented.')


class BaseCompositeOracle(object):
    """
    Base class for the composite function.
    phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part.
    """

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def func(self, x):
        """
        Computes the f(x) + h(x).
        """
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        """
        Computes the gradient of f(x).
        """
        return self._f.grad(x)

    def prox(self, x, alpha):
        """
        Computes the proximal mapping.
        """
        return self._h.prox(x, alpha)

    def duality_gap(self, x):
        """
        Estimates the residual phi(x) - phi* via the dual problem, if any.
        """
        raise NotImplementedError('Duality gap is not implemented.')


class LeastSquaresOracle(BaseSmoothOracle):
    """
    Oracle for least-squares regression.
        f(x) = 0.5 ||Ax - b||_2^2
    """
    # TODO: implement.
    def __init__(self, matvec_Ax, matvec_ATx, b):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = b

    def func(self, x):
        return 0.5 * np.linalg.norm(self.matvec_Ax(x) - self.b)**2

    def grad(self, x):
        return self.matvec_ATx(self.matvec_Ax(x) - self.b)


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer.
        h(x) = regcoef * ||x||_1.
    """
    # TODO: implement.
    def __init__(self, regcoef):
        self.regcoef = regcoef

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        return np.sign(x) * np.maximum(np.abs(x) - alpha * self.regcoef, 0)


class LassoProxOracle(BaseCompositeOracle):
    """
    Oracle for 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        f(x) = 0.5 * ||Ax - b||_2^2 is a smooth part,
        h(x) = regcoef * ||x||_1 is a simple part.
    """
    # TODO: implement.
    def __init__(self, f, h):
        super().__init__(f, h)

    def lasso_duality_gap(self, x, Ax_b, ATAx_b, b, regcoef):
        """
        Estimates f(x) - f* via duality gap for 
            f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
        """
        if np.linalg.norm(ATAx_b, ord=np.inf) < 1e-5:
            mu = Ax_b
        else:
            mu = np.min([1., regcoef / np.linalg.norm(ATAx_b, ord=np.inf)]) * Ax_b

        return np.linalg.norm(Ax_b) ** 2 / 2 + regcoef * np.linalg.norm(x, 1) + \
            np.linalg.norm(mu) ** 2 / 2 + b @ mu

    def duality_gap(self, x):
        Ax_b = self._f.matvec_Ax(x) - self._f.b
        ATAx_b = self._f.matvec_ATx(Ax_b)
        return self.lasso_duality_gap(x, Ax_b, ATAx_b, self._f.b, self._h.regcoef)


def create_lasso_prox_oracle(A, b, regcoef):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    return LassoProxOracle(LeastSquaresOracle(matvec_Ax, matvec_ATx, b),
                           L1RegOracle(regcoef))