import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from HypMergeTree import *

def f1(x, ws):
    x1, x2 = x
    w1, w2, w3, w4, w5, w6, w7 = ws
    return w1*w5-w6*w2-w2**2 + x1*(w1+w5+2*w2+w6) - (w2+w6)*x2 + x1*x2

def f2(x, ws):
    x1, x2 = x
    w1, w2, w3, w4, w5, w6, w7 = ws
    return w6*w2-w3*w7-w4*w7-w3**2-w3*w4 - w6*x1 + (w6+w7+2*w3+w4+w2)*x2 - x1*x2

def g(x, ws):
    return 0.5*(f1(x, ws)**2 + f2(x, ws)**2)

def gradg(x, ws):
    x1, x2 = x
    w1, w2, w3, w4, w5, w6, w7 = ws
    g1 = f1(x, ws)*(w1+w5+2*w2+w6+x2) - f2(x, ws)*(w6+x2)
    g2 = f1(x, ws)*(x1-w2-w6) + f2(x, ws)*(w6+w7+2*w3+w2+w4-x1)
    return np.array([g1, g2])

def get_pentagon_xs(ws, verbose = True):
    x0 = np.ones(2)
    res = opt.minimize(g, x0, args = (ws), method='BFGS', jac=gradg)
    xsol = res['x']
    if verbose:
        print("g(x0) = ", g(x0, ws))
        print("xsol = ", xsol)
        print("g(xsol) = ", g(xsol, ws))
    return xsol

def get_pentagon_zs_rs(ws):
    xs = get_pentagon_xs(ws)
    z1 = 1.0
    getz2 = lambda z1, xs, ws: z1*(xs[0]+ws[4])/(ws[1]-xs[0]+ws[5])
    getz3 = lambda z2, xs, ws: z2*(xs[1]+ws[5])/(ws[2]-xs[1]+ws[6])

    z2 = getz2(z1, xs, ws)
    z3 = getz3(z2, xs, ws)
    x1, x2 = xs
    w1, w2, w3, w4, w5, w6, w7 = ws
    r0 = z1*(w1+w5)
    r1 = z1*(x1+w5)
    r2 = z2*(x2+w6)
    r3 = z3*(w4+w7)
    rinf = (z1+z2+z3)/(w1+w2+w3+w4)

    return xs, np.array([0, z1, z1+z2, z1+z2+z3]), np.array([r0, r1, r2, r3, rinf])


def neg_weight_case():
    np.random.seed(3)
    ws = np.random.rand(7)
    ws = np.round(ws*10)/10.0

    """    
    print("ws = ", ws)
    print("zs = ", zs)
    print("rs = ", rs)
    """

    HMT = HypMergeTree()
    rscale = 0.5

    w6s = np.linspace(ws[5], 0, 300)
    plt.figure(figsize=(6, 6))
    for i, w6 in enumerate(w6s):
        ws[5] = w6
        xs, zs, rs = get_pentagon_zs_rs(ws)
        rs[0:-1] *= rscale
        rs[-1] /= rscale
        HMT.z = zs
        HMT.radii = rs
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.plot([zs[1]]*2, [0, 3.5], linestyle=':', color='k')
        plt.plot([zs[2]]*2, [0, 3.5], linestyle=':', color='k')
        plt.xlim([-0.5, 4])
        plt.ylim([-0.5, 4])
        plt.title("x1 = %.3g, x2 = %.3g"%tuple(xs.tolist()))
        plt.savefig("%i.png"%i)


if __name__ == '__main__':
    neg_weight_case()