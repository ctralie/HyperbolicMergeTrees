import numpy as np
import matplotlib.pyplot as plt
from HypMergeTree import *
from PolynomialSystem import g, gradg, solvesystem

def get_pentagon_xs_zs_rs(ws, x0 = np.ones(2)):
    # a = w5, b = w1, c = w2, d = w6, e = x2
    idxs1 = [0, 6, 2, 3, 7, 1]
    # a = w6, b = (w2-x1), c = w3, d = w7, e = w4
    idxs2 = [1, 7, [0, 3], 4, 8, 5]
    xs = solvesystem(x0, ws, [idxs1, idxs2])

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

def get_hexagon_xs(ws, x0 = np.ones(3), verbose = False):
    # a = w6, b = w1, c = w2, d = w7, e = x2
    idxs1 = [0, 8, 3, 4, 9, 1]
    # a = w7, b = (w2-x1), c = w3, d = w8, e = x3
    idxs2 = [1, 9, [0, 4], 5, 10, 2]
    # a = w8, b = w3-x2, c = w4, d = w9, e = w5
    idxs3 = [2, 10, [1, 5], 6, 11, 7]
    xs = solvesystem(x0, ws, [idxs1, idxs2, idxs3])

    z1 = 1.0
    getz2 = lambda z1, xs, ws: z1*(xs[0]+ws[5])/(ws[1]-xs[0]+ws[6])
    getz3 = lambda z2, xs, ws: z2*(xs[1]+ws[6])/(ws[2]-xs[1]+ws[7])

def pentagon_edgecollapse():
    np.random.seed(3)
    ws = np.random.rand(7)
    ws = np.round(ws*10)/10.0
    print("ws = ", ws)
    HMT = HypMergeTree()
    rscale = 0.5

    w6s = np.linspace(ws[5], 0, 300)
    plt.figure(figsize=(6, 6))
    for i, w6 in enumerate(w6s):
        ws[5] = w6
        xs, zs, rs = get_pentagon_xs_zs_rs(ws)
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
    pentagon_edgecollapse()