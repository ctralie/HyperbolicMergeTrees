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
    zs0 = np.arange(N-2)+1
    rs0 = np.ones(N-1)*1
    vs0 = np.zeros(N-3)

    NTrials = 20
    solutions = []
    times = np.zeros(NTrials)
    np.random.seed(0)
    NInvalid = 0
    fxs_initial = np.zeros(NTrials)
    fxs_sol = np.zeros(NTrials)
    allzs = []
    x0 = 0
    rinf = 1

    HMT = HypMergeTree()
    HMT.z = np.zeros(N-1)
    HMT.z[0] = x0
    HMT.radii = np.zeros(N)
    HMT.radii[-1] = rinf

    for i, trial in enumerate(range(NTrials)):
        tic = time.time()
        zs0 = np.sort(np.random.randn(zs0.size))
        zs0 -= zs0[0]
        zs0 *= np.random.rand(zs0.size)
        rs0 = np.abs(np.random.randn(rs0.size))
        vs0 = np.random.randn(vs0.size)
        res = hd.solve_equations(zs0, rs0, vs0, x0=x0, rinf=rinf)
        times[i] = time.time()-tic
        fxs_initial[i] = res['fx_initial']
        fxs_sol[i] = res['fx_sol']
        zs, rs, vs = res['zs'], res['rs'], res['vs']
        allzs.append(zs)
        x_sol = np.concatenate((zs, rs, vs))
        #print(hd.f(x_sol, hd.get_equations(), x0, rinf))
        if np.sum(zs[1::] - zs[0:-1] < 0) == 0 and np.sum(zs < 0) == 0 and np.sum(zs[1::]-zs[0:-1] < EPS) == 0 and zs[0] > EPS:
            HMT.z[1::] = res['zs']
            HMT.radii[0:-1] = res['rs']
        else:
            NInvalid += 1

    symbolic=False
    plt.subplot(221)
    T.render(offset=np.array([0, 0]))
    plt.title("Merge Tree")
    plt.subplot(222)
    hd.render(symbolic=symbolic)
    plt.title("Topological Triangulation")
    plt.subplot(224)
    #plt.text(0, 0, hd.get_equations_tex(symbolic=symbolic))
    #plt.axis('off')
    #plt.title("Hyperbolic Equations")
    lengths = hd.get_horocycle_arclens()
    z = np.zeros_like(lengths)
    z[0:-1] = HMT.z
    z[-1] = -1
    plt.stem(z, lengths)
    plt.title("Masses (Total Mass %.3g)"%np.sum(lengths))
    #plt.stem(HMT.z, HMT.radii[0:-1])    
    #plt.title("Radii (Radii Sum = %.3g)"%np.sum(HMT.radii[0:-1]))
    plt.xlabel("z")
    plt.ylabel("Mass")
    
    plt.subplot(223)
    HMT.refreshNeeded = True
    HMT.renderVoronoiDiagram()
    plt.title("Hyperbolic Voronoi Diagram")


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