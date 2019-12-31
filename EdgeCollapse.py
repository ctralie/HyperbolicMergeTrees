import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sn
from HypDelaunay import *
from HypMergeTree import *
from MergeTree import *

EPS = 1e-4

def mergetree_to_hypmergetree(T, symbolic=False, verbose=False):
    """
    Given a chiral merge tree, setup the Delaunay triangulation
    and associated equations, solve for the zs/rs, and then
    solve for the hyperbolic Voronoi diagram from the zs/rs
    Parameters
    ----------
    T: MergeTree
        A merge tree object from which to construct
        a hyperbolic structure
    symbolic: boolean
        Whether to use variables for the edge weights (True), or
        whether to display the actual numerical edge weights (False)
    verbose: boolean
        Whether to print the solutions
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

    constraints = [('z', 0, 0), ('r', -1, 1)]

    for i, trial in enumerate(range(NTrials)):
        tic = time.time()
        # Setup some random initial conditions
        zs0 = np.random.randn(N-1)
        rs0 = np.abs(np.random.randn(N))
        for (constraint_type, index, value) in constraints:
            if constraint_type == 'r':
                rs0[index] = value
            elif constraint_type == 'z':
                zs0[index] = value
        vs0 = np.random.randn(N-3)
        res = hd.solve_equations(zs0, rs0, vs0, constraints)
        times[i] = time.time()-tic
        zs, rs, vs = res['zs'], res['rs'], res['vs']
        if np.sum(zs[1::] - zs[0:-1] < 0) == 0 and np.sum(zs[1::]-zs[0:-1] < EPS) == 0:
            HMT.z = zs
            HMT.radii = rs
            if verbose:
                print("zs:", ["%.3g, "*zs.size%(tuple(list(zs)))])
                print("rs", ["%.3g, "*rs.size%(tuple(list(rs)))])
                print("fx_initial = %.3g"%res['fx_initial'])
                print("fx_sol = %.3g"%res['fx_sol'])
        else:
            NInvalid += 1
    
    plt.subplot(231)
    T.render(offset=np.array([0, 0]))
    plt.title("Merge Tree")
    plt.subplot(232)
    hd.render(symbolic=symbolic)
    plt.title("Topological Triangulation")

    plt.subplot(234)
    HMT.refreshNeeded = True
    HMT.renderVoronoiDiagram()
    plt.title("Hyperbolic Voronoi Diagram")

    plt.subplot(235)
    lengths = hd.get_horocycle_arclens()
    z = np.zeros_like(lengths)
    z[0:-1] = HMT.z
    z[-1] = -1
    plt.stem(z, lengths)
    plt.title("Masses (Total Mass %.3g)"%np.sum(lengths))
    plt.xlabel("z")
    plt.ylabel("Mass")

    plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
    plt.text(0, 0, hd.get_equations_tex(symbolic=symbolic))
    plt.title("Hyperbolic Equations")
    plt.axis('off')

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

    plt.figure(figsize=(18, 12))
    N = 20
    for i, Ey in enumerate(np.linspace(2, 3, N)):
        E.X[1] = Ey
        plt.clf()
        mergetree_to_hypmergetree(T)
        plt.subplot(231)
        plt.title("Merge Tree (Height = %.3g)"%Ey)
        plt.subplot(234)
        plt.xlim([-0.5, 4.5])
        plt.ylim([0, 3])
        plt.subplot(235)
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
    mergetree_to_hypmergetree(T)
    plt.subplot(234)
    plt.xlim([-0.5, 4.5])
    plt.ylim([0, 3])
    plt.subplot(235)
    plt.xlim([-1.5, 4.5])
    plt.ylim([0, 6.5])
    plt.savefig("%i.png"%N, bbox_inches='tight')
        

if __name__ == '__main__':
    testEdgeCollapse()