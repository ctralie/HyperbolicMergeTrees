import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def getvals(xs, ws, idxs):
    """
    Helper function for f and gradf
    """
    xs_ws = np.concatenate((xs, ws))
    x = xs_ws[idxs[0]]
    others = np.zeros(5)
    for i, idx in enumerate(idxs[1::]):
        if type(idx) == list:
            others[i] = xs_ws[idx[1]] - xs_ws[idx[0]]
        else:
            others[i] = xs_ws[idx]
    [a, b, c, d, e] = others.tolist()
    return [x, a, b, c, d, e]

def f(xs, ws, idxs):
    """
    Set up an objective function for satisfying a particular equation

    Parameters
    ----------
    xs: ndarray(n)
        Current estimate of the partial weights
    ws: ndarray(n*2 + 3)
        Weights of the tree edges
    idxs: list(6) indexing into (n*3+3)
        Indexes into xs and ws specifying which quantities are involved
        in this equation.  xs are indexed first, followed by ws.
        The order is x, a, b, c, d, e
        If it is a 2-element list, then it is ws_xs[1] - ws_xs[0] = k - x
        for some merge tree edge length k and a variable x
    
    Returns
    -------
    f(xs): int
        The evaluation of the function
    """
    [x, a, b, c, d, e] = getvals(xs, ws, idxs)
    return (a+x)*(b+x) - (c-x+d)*(c-x+e)

def gradf(xs, ws, idxs):
    """
    The gradient of an objective function for satisfiability of a particular equation
    at a point, with respect to all of the xs
    Parameters
    ----------
    xs: ndarray(n)
        Current estimate of the partial weights
    ws: ndarray(n*2 + 3)
        Weights of the tree edges
    idxs: list(6) indexing into (n*3+3)
        Indexes into xs and ws specifying which quantities are involved
        in this equation.  xs are indexed first, followed by ws.
        The order is x, a, b, c, d, e
        If it is a 2-element list, then it is ws_xs[1] - ws_xs[0] = k - x
        for some merge tree edge length k and a variable x
    
    Returns
    -------
    grad(xs): ndarray(n)
        Gradient at the point xs
    """
    grad = np.zeros(len(xs))
    vals = getvals(xs, ws, idxs)
    [x, a, b, c, d, e] = vals
    N = len(xs)
    grad[idxs[0]] = a + b + 2*c + d + e
    coeffweight = [1, 1, -1, -1]
    for coeffweight, elem, idx in zip([1, 1, -1, -1], [b, a, e, c+d], [idxs[i] for i in [1, 2, 4, 5]]):
        if type(idx) == list:
            grad[idx[0]] = -x - coeffweight*elem
        else:
            if idx < N:
                grad[idx] = x + coeffweight*elem
    return grad

def g(xs, ws, allidxs):
    """
    Return the objective function 0.5*[ sum_i f_i(x)^2 ]
    Parameters
    ----------
    xs: ndarray(n)
        Current estimate of the partial weights
    ws: ndarray(n*2 + 3)
        Weights of the tree edges
    idxs: list (num equations)
        A list of all of the equation index lists, as specified for f and gradf    
    Returns
    -------
    g(xs): float
        Value of the objective function at the point xs
    """
    res = 0.0
    for idxs in allidxs:
        res += f(xs, ws, idxs)**2
    return 0.5*res

def gradg(xs, ws, allidxs):
    """
    Return the gradient of the objective function 0.5*[ sum_i f_i(x)^2 ]
    Parameters
    ----------
    xs: ndarray(n)
        Current estimate of the partial weights
    ws: ndarray(n*2 + 3)
        Weights of the tree edges
    idxs: list (num equations)
        A list of all of the equation index lists, as specified for f and gradf    
    Returns
    -------
    gradg(xs): ndarray(n)
        Value of the gradient of g at the point xs
    """
    res = np.zeros(xs.size)
    for idxs in allidxs:
        res += f(xs, ws, idxs)*gradf(xs, ws, idxs)
    return res