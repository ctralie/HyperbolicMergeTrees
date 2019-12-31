import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sn
from HypDelaunay import *
from HypMergeTree import *
from MergeTree import *

EPS = 1e-4

def plotSolutions(T):
    """
    Parameters
    ----------
    T: MergeTree
        A merge tree object from which to construct
        a hyperbolic structure
    """
    hd = HyperbolicDelaunay()
    hd.init_from_mergetree(T)

    N = len(hd.vertices)

    NTrials = 20
    solutions = []
    times = np.zeros(NTrials)
    np.random.seed(0)
    NInvalid = 0

    HMT = HypMergeTree()
    HMT.z = np.zeros(N-1)
    HMT.radii = np.zeros(N)

    for i, trial in enumerate(range(NTrials)):
        tic = time.time()
        zs0 = np.sort(np.random.randn(N-1))
        zs0 -= zs0[0]
        zs0 *= np.random.rand(zs0.size)
        rs0 = np.abs(np.random.randn(N))
        vs0 = np.random.randn(N-3)
        res = hd.solve_equations(zs0, rs0, vs0)
        times[i] = time.time()-tic
        zs, rs, vs = res['zs'], res['rs'], res['vs']
        x_sol = np.concatenate((zs, rs, vs))
        if np.sum(zs[1::] - zs[0:-1] < 0) == 0 and np.sum(zs < 0) == 0 and np.sum(zs[1::]-zs[0:-1] < EPS) == 0 and zs[0] > EPS:
            HMT.z = res['zs']
            HMT.radii = res['rs']
        else:
            NInvalid += 1

    symbolic=False
    plt.subplot(231)
    T.render(offset=np.array([0, 0]))
    plt.title("Merge Tree")
    plt.subplot(232)
    hd.render(symbolic=symbolic)
    plt.title("Topological Triangulation")
    plt.subplot(235)
    lengths = hd.get_horocycle_arclens()
    z = np.zeros_like(lengths)
    z[0:-1] = HMT.z
    z[-1] = -1
    plt.stem(z, lengths)
    plt.title("Masses (Total Mass %.3g)"%np.sum(lengths))
    plt.xlabel("z")
    plt.ylabel("Mass")
    
    plt.subplot(234)
    HMT.refreshNeeded = True
    HMT.renderVoronoiDiagram()
    plt.title("Hyperbolic Voronoi Diagram")

    plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    plt.text(0, 0, hd.get_equations_tex(symbolic=symbolic))
    plt.axis('off')
    plt.title("Hyperbolic Equations")


def testEdgeCollapse():
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([1, 4]))
    T.root.addChildren([A, B])
    C = MergeNode(np.array([0.5, 3]))
    D = MergeNode(np.array([2, 3]))
    B.addChildren([C, D])
    E = MergeNode(np.array([1.5, 2]))
    F = MergeNode(np.array([3, 2]))
    D.addChildren([E, F])

    plt.figure(figsize=(12, 12))
    N = 20
    for i, Ey in enumerate(np.linspace(2, 3, N)):
        E.X[1] = Ey
        plt.clf()
        plotSolutions(T)
        plt.subplot(221)
        plt.title("Merge Tree (Height = %.3g)"%Ey)
        plt.subplot(223)
        plt.xlim([-0.5, 4.5])
        plt.ylim([0, 3])
        plt.subplot(224)
        plt.xlim([-1.5, 4.5])
        plt.ylim([0, 6.5])
        plt.savefig("%i.png"%i, bbox_inches='tight')
    
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([1, 4]))
    T.root.addChildren([A, B])
    C = MergeNode(np.array([0.5, 3]))
    D = MergeNode(np.array([2, 3]))
    #B.addChildren([C, D])
    #E = MergeNode(np.array([1.5, 2]))
    F = MergeNode(np.array([3, 2]))
    #D.addChildren([E, F])
    B.addChildren([C, F])

    plt.clf()
    plotSolutions(T)
    plt.subplot(223)
    plt.xlim([-0.5, 4.5])
    plt.ylim([0, 3])
    plt.subplot(224)
    plt.xlim([-1.5, 4.5])
    plt.ylim([0, 6.5])
    plt.savefig("%i.png"%N, bbox_inches='tight')
        

if __name__ == '__main__':
    testEdgeCollapse()