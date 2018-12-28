import numpy as np
import matplotlib.pyplot as plt
from HypMergeTree import *
from PolynomialSystem import g, gradg, solvesystem

def testQuadEdgeFlip(NSamples = 200):
    w1, w2, w4, w5 = 1.0, 2.0, 1.0, 1.5
    z1 = 1.0
    getz2 = lambda w1, w2, w3, w4, w5: z1*(w2+w3+w5)/(w1+w3+w4)
    #First do branch on right
    HMT = HypMergeTree()
    framenum = 0
    w3max = 1.0
    w3s = w3max - w3max*np.linspace(0, 1, NSamples+2)
    rscale = 0.3
    w3s = w3s[1:-1]
    for w3 in w3s:
        z2 = getz2(w1, w2, w3, w4, w5)
        alpha_inf = w1 + w3 + w5
        alpha0 = w1 + w2
        alpha1 = w2 + w3 + w4
        alpha2 = w4 + w5
        A = z2*alpha1/(z1+z2)
        rinf = (z1+z2)/alpha_inf
        r0 = z1*alpha0
        r1 = z1*A
        r2 = z2*alpha2
        r0, r1, r2 = r0*rscale, r1*rscale, r2*rscale
        rinf = rinf/rscale
        HMT.z = np.array([0, z1, z1+z2], dtype = np.float64)
        HMT.radii = np.array([r0, r1, r2, rinf])
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.title("z2 = %.3g, r0 = %.3g, r1 = %.3g, r2 = %.3g\n$r_{\infty}$ = %.3g, w3 = %.3g\n$\\alpha$0=%.3g, $\\alpha$1=%.3g, $\\alpha$2=%.3g, $\\alpha_{\infty}$=%.3g"%(z2, r0, r1, r2, rinf, w3, alpha0, alpha1, alpha2, alpha_inf))
        ax = plt.gca()
        ax.set(xlim=[-1, 5], ylim=[-1, 5], aspect=1)
        plt.savefig("%i.png"%framenum, bbox_inches='tight', dpi=300)
        framenum += 1
    
    w3s = w3max - w3s
    for w3 in w3s:
        print("framenum = %i"%framenum)
        z2 = getz2(w1, w2, w3, w4, w5)
        alpha_inf = w1 + w5
        alpha0 = w1 + w2 + w3
        alpha1 = w2 + w4
        alpha2 = w3 + w4 + w5
        A = z1*z2*alpha1/(z1+z2) #This is the only one that's different!
        rinf = (z1+z2)/alpha_inf
        r0 = z1*alpha0
        r1 = z1*A
        r2 = z2*alpha2
        r0, r1, r2 = r0*rscale, r1*rscale, r2*rscale
        rinf = rinf/rscale
        HMT.z = np.array([0, z1, z1+z2], dtype = np.float64)
        HMT.radii = np.array([r0, r1, r2, rinf])
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.title("z2 = %.3g, r0 = %.3g, r1 = %.3g, r2 = %.3g\n$r_{\infty}$ = %.3g, w3 = %.3g\n$\\alpha$0=%.3g, $\\alpha$1=%.3g, $\\alpha$2=%.3g, $\\alpha_{\infty}$=%.3g"%(z2, r0, r1, r2, rinf, w3, alpha0, alpha1, alpha2, alpha_inf))
        ax = plt.gca()
        ax.set(xlim=[-1, 5], ylim=[-1, 5], aspect=1)
        plt.savefig("%i.png"%framenum, bbox_inches='tight', dpi=300)
        framenum += 1

def get_pentagon_xs_zs_rs(ws, x0 = np.ones(2), z1 = 1.0):
    # a = w5, b = w1, c = w2, d = w6, e = x2
    idxs1 = [0, 6, 2, 3, 7, 1]
    # a = w6, b = (w2-x1), c = w3, d = w7, e = w4
    idxs2 = [1, 7, [0, 3], 4, 8, 5]
    xs = solvesystem(x0, ws, [idxs1, idxs2])

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

    return xs, np.array([0, z1, z2, z3]), np.array([r0, r1, r2, r3, rinf])

def get_hexagon_xs(ws, x0 = np.ones(3), verbose = False):
    # a = w6, b = w1, c = w2, d = w7, e = x2
    idxs1 = [0, 8, 3, 4, 9, 1]
    # a = w7, b = (w2-x1), c = w3, d = w8, e = x3
    idxs2 = [1, 9, [0, 4], 5, 10, 2]
    # a = w8, b = w3-x2, c = w4, d = w9, e = w5
    idxs3 = [2, 10, [1, 5], 6, 11, 7]
    return solvesystem(x0, ws, [idxs1, idxs2, idxs3], verbose)

def get_hexagon_xs_zs_rs(ws, x0 = np.ones(3), xs = np.array([]), z1 = 1.0, verbose = False):
    if xs.size == 0:
        xs = get_hexagon_xs(ws, x0)
    getz2 = lambda z1, xs, ws: z1*(xs[0]+ws[5])/(ws[1]-xs[0]+ws[6])
    getz3 = lambda z2, xs, ws: z2*(xs[1]+ws[6])/(ws[2]-xs[1]+ws[7])
    getz4 = lambda z3, xs, ws: z3*(xs[2]+ws[7])/(ws[3]-xs[2]+ws[8])
    z2 = getz2(z1, xs, ws)
    z3 = getz3(z2, xs, ws)
    z4 = getz4(z3, xs, ws)
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = ws
    x1, x2, x3 = xs
    r0 = z1*(w1+w6)
    r1 = z1*(w6+x1)
    r2 = z2*(w7+x2)
    r3 = z3*(w8+x3)
    r4 = z4*(w5+w9)
    rinf = (z1+z2+z3+z4)/(w1+w2+w3+w4+w5)

    return xs, np.array([0, z1, z2, z3, z4]), np.array([r0, r1, r2, r3, r4, rinf])

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
        zs = np.cumsum(zs)
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

def hexagon_edgecollapse():
    """
    Showing a collapse of the first left branch edge of an "easy hexagon"
    This example shows that the optimal transport metric I came up with
    intially doesn't work
    """
    np.random.seed(3)
    ws = np.random.rand(9)
    ws = np.round(ws*10)/10.0
    ws[6] = 0.4
    print("ws = ", ws)
    HMT = HypMergeTree()
    rscale = 0.5

    w7s = np.linspace(ws[6], 0, 300)
    plt.figure(figsize=(6, 6))
    idx = 0
    for w7 in w7s:
        ws[6] = w7
        xs, zs, rs = get_hexagon_xs_zs_rs(ws)
        rs[0:-1] *= rscale
        rs[-1] /= rscale
        zs = np.cumsum(zs)
        HMT.z = zs
        HMT.radii = rs
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.plot([zs[1]]*2, [0, 3.5], linestyle=':', color='k')
        plt.plot([zs[2]]*2, [0, 3.5], linestyle=':', color='k')
        plt.plot([zs[3]]*2, [0, 3.5], linestyle=':', color='k')
        plt.xlim([-0.5, 5])
        plt.ylim([-0.5, 5])
        plt.title("x1 = %.3g, x2 = %.3g, x3 = %.3g"%tuple(xs.tolist()))
        plt.savefig("%i.png"%idx)
        idx += 1
    
    # Now plot the pentagon that results after this edge is gone
    wsp = [ws[0], ws[1]+ws[2], ws[3], ws[4], ws[5], ws[7], ws[8]]
    xs, zs, rs = get_pentagon_xs_zs_rs(wsp)
    rs[0:-1] *= rscale
    rs[-1] /= rscale
    zs = np.cumsum(zs)
    HMT.z = zs
    HMT.radii = rs
    plt.clf()
    HMT.refreshNeeded = True
    HMT.renderVoronoiDiagram()
    plt.plot([zs[1]]*2, [0, 3.5], linestyle=':', color='k')
    plt.plot([zs[2]]*2, [0, 3.5], linestyle=':', color='k')
    plt.xlim([-0.5, 5])
    plt.ylim([-0.5, 5])
    plt.title("x1 = %.3g, x2 = %.3g"%tuple(xs.tolist()))
    plt.savefig("%i.png"%idx)


def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getGreedyPerm(X, tol = 1e-3, Verbose = False):
    """
    Purpose: Naive O(NM) algorithm to do the greedy permutation
    :param X: Nxd array of Euclidean points
    :param tol: Cutoff for when sampling is done
    :returns: (permutation (N-length array of indices), \
            lambdas (N-length array of insertion radii))
    """
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = []
    lambdas = []
    ds = getCSM(X[0, :][None, :], X).flatten()
    r = np.inf
    while r > tol:
        idx = np.argmax(ds)
        perm.append(idx)
        lambdas.append(ds[idx])
        r = lambdas[-1]
        ds = np.minimum(ds, getCSM(X[idx, :][None, :], X).flatten())
    perm = np.array(perm)
    lambdas = np.array(lambdas)
    Y = X[perm, :]
    return {'Y':Y, 'perm':perm, 'lambdas':lambdas}

def hexagon_multiinit():
    """
    A test for solutions of the "easy hexagon" with random initializations.
    So far, it seems that there is exactly one solution in which the
    zs are nonnegative
    """
    nonnegative_only = True
    for seed in range(1000):
        np.random.seed(seed)
        ws = np.random.rand(9)
        ws = np.round(ws*10)/10.0
        print("ws = ", ws)
        HMT = HypMergeTree()
        rscale = 0.5

        NTrials = 1000
        x0s = np.random.randn(NTrials, 3)
        xsols = []
        zsols = []
        for i in range(NTrials):
            xs, zs, rs = get_hexagon_xs_zs_rs(ws, x0 = x0s[i, :])
            if (not nonnegative_only) or np.sum(zs < 0) == 0:
                xsols.append(xs)
        xsols = np.array(xsols)
        
        # Do greedy furthest point sampling in solution space
        xsols = getGreedyPerm(xsols)['Y']

        plt.figure(figsize=(10, 10))
        for i in range(xsols.shape[0]):
            xs, zs, rs = get_hexagon_xs_zs_rs(ws, xs=xsols[i, :])
            rs[0:-1] *= rscale
            rs[-1] /= rscale
            HMT.z = np.cumsum(zs)
            HMT.radii = rs
            plt.clf()
            HMT.refreshNeeded = True
            HMT.renderVoronoiDiagram()
            plt.axis('equal')
            s = "x1 = %.3g, x2 = %.3g, x3 = %.3g\n"%tuple(xs.tolist())
            for k, z in enumerate(zs):
                zstr = "z%i = %.3g"%(k, z)
                if z < 0:
                    zstr = "$\\mathbf{" + zstr + "}$"
                s += zstr
                if k < len(zs)-1:
                    s += ", "
            s += "\n"
            for k, r in enumerate(rs):
                s += "r%i = %.3g"%(k, r)
                if k < len(rs)-1:
                    s += ", "

            plt.title(s)
            plt.savefig("HexagonExample%i_%i.png"%(seed+1, i+1))


def testEvenEdges():
    HMT = HypMergeTree()
    ws = np.ones(9)
    rscale = 1.0
    xs, zs, rs = get_hexagon_xs_zs_rs(ws)
    print("xs = ", xs)
    print("zs = ", zs)
    print("rs = ", rs)
    rs[0:-1] *= rscale
    rs[-1] /= rscale
    zs = np.cumsum(zs)
    HMT.z = zs
    HMT.radii = rs
    HMT.refreshNeeded = True
    HMT.renderVoronoiDiagram()
    plt.show()


def testQuadSimplified(NSamples = 100):
    w1, w2, w4, w5 = 1.0, 2.0, 1.0, 1.5
    z1 = 1.0
    z2 = 2.0
    #First do branch on right
    HMT = HypMergeTree()
    framenum = 0
    w3max = 1.0
    w3s = w3max - w3max*np.linspace(0, 1, NSamples+2)
    w3s = w3s[1:-1]
    for w3 in w3s:
        r0 = z1*(w1+w2)
        r1 = (w2+w3+w4)*(z2-z1)*z1
        r2 = z2*(w4+w5)
        rinf = (z1+z2)/(w1+w3+w5)
        HMT.z = np.array([0, z1, z2], dtype = np.float64)
        HMT.radii = np.array([r0, r1, r2, rinf])
        plt.clf()
        HMT.refreshNeeded = True
        HMT.renderVoronoiDiagram()
        plt.title("w3 = %.3g"%w3)
        plt.axis('equal')
        plt.savefig("%i.png"%framenum, bbox_inches='tight', dpi=300)
        framenum += 1

if __name__ == '__main__':
    #pentagon_edgecollapse()
    #hexagon_edgecollapse()
    #hexagon_multiinit()
    #testEvenEdges()
    #testQuadEdgeFlip()
    testQuadSimplified()