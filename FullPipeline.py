"""
Code to glue together chiral merge trees, hyperbolic
delaunay triangluations / equations, and hyperbolic
Voronoi diagrams
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sn
from HypDelaunay import *
from HypMergeTree import *
from MergeTree import *

def mergetree_to_hypmergetree(cmt, constraints, max_trials = 500, z_eps = 1e-4, sol_eps = 1e-7, verbose=True):
    """
    Given a chiral merge tree, setup the Delaunay triangulation
    and associated equations, solve for the zs/rs, and then
    solve for the hyperbolic Voronoi diagram from the zs/rs
    Parameters
    ----------
    cmt: MergeTree
        A chiral merge tree object from which to construct
        a hyperbolic structure
    constraints: list of [(variable type ('z' or 'r'), 
                            index (int), 
                            value (float)]
        A dictionary of constraints to enforce on the zs and rs.
        -1 for r is r_infinity
    max_trials: int
        Number of random initializations to try
    z_eps: float
        All zs must be at least this far apart
    sol_eps: float
        Objective function must have converged to this level
    verbose: boolean
        Whether to print the solutions
    
    Returns
    -------
    {
        'hd': HyperbolicDelaunay
            An object holding the Delaunay triangulation and
            methods for constructing, solving, and plotting the equations,
        'hmt': HypMergeTree
            An object holding the hyperbolic Voronoi diagram corresponding
            to the zs/rs solution,
        'times': ndarray(max_trials)
            The time taken to solve each initial condition,
        'n_invalid': int
            The number of solutions deemed not to be valid
    }
    """
    hd = HyperbolicDelaunay()
    hd.init_from_mergetree(cmt)

    N = len(hd.vertices)
    times = np.zeros(max_trials)
    np.random.seed(0)
    n_invalid = 0

    hmt = HypMergeTree()
    hmt.z = np.zeros(N-1)
    hmt.radii = np.zeros(N)

    i = 0
    solution_found = False
    while i < max_trials and not solution_found:
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
        fx_sol = res['fx_sol']
        # Check the following conditions
        # 1) The zs are in the right order
        # 2) Adjacent zs are more than epsilon apart
        # 3) The rs are nonzero
        # 4) The objective function is less than epsilon
        if np.sum(zs[1::] - zs[0:-1] < 0) == 0 and np.sum(zs[1::]-zs[0:-1] < z_eps) == 0 and np.sum(rs < 0) == 0 and fx_sol < sol_eps:
            # Copy over the solution to the hyperbolic voronoi diagram
            # if it is valid
            hmt.z = zs
            hmt.radii = rs
            solution_found = True
            if verbose:
                print("zs:", ["%.3g, "*zs.size%(tuple(list(zs)))])
                print("rs", ["%.3g, "*rs.size%(tuple(list(rs)))])
                print("fx_initial = %.3g"%res['fx_initial'])
                print("fx_sol = %.3g\n"%fx_sol)
        else:
            n_invalid += 1
        i += 1
    return {'hd':hd, 'hmt':hmt, 'times':times, 'n_invalid':n_invalid}

def plot_solution_grid(cmt, hd, hmt, constraints, symbolic=False,
                       xlims = None, ylims_voronoi = None, 
                       ylims_masses = None, perturb=0):
    """
    Show the original chiral merge tree next to the topological
    triangulation, the associated equations, the Voronoi diagram
    solution, and the solution expressed as point masses
    Parameters
    ----------
    cmt: MergeTree
        The original chiral merge tree
    hd: HyperbolicDelaunay
        An object holding the Delaunay triangulation and
        methods for constructing, solving, and plotting the equations
    hmt: HypMergeTree
        An object holding the hyperbolic Voronoi diagram corresponding
        to the zs/rs solution
    constraints: list of [(variable type ('z' or 'r'), 
                            index (int), 
                            value (float)]
        A dictionary of constraints to enforce on the zs and rs.
        -1 for r is r_infinity
    symbolic: boolean
        Whether to use variables for the edge weights (True), or
        whether to display the actual numerical edge weights (False)
    xlims: [int, int]
        Optional x limits for Voronoi diagram and point masses
    ylims_voronoi: [int, int]
        Optional y limits for Voronoi diagram
    ylims_masses: [int, int]
        Opitional y limits for point masses
    perturb: boolean
        Whether to perturb z positions slightly
    """   
    plt.subplot(231)
    cmt.render(offset=np.array([0, 0]))
    plt.title("Chiral Merge Tree")

    plt.subplot(232)
    hd.render(symbolic=symbolic)
    plt.title("Topological Triangulation")

    plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
    plt.text(0, 0, hd.get_equations_tex(constraints=constraints, symbolic=symbolic))
    plt.title("Hyperbolic Equations")
    plt.axis('off')

    plt.subplot(234)
    if perturb > 0:
        z_orig = np.array(hmt.z)
        hmt.z += np.random.rand(hmt.z.size)*perturb
    hmt.refreshNeeded = True
    hmt.renderVoronoiDiagram()
    if perturb > 0:
        hmt.z = z_orig
    plt.title("Hyperbolic Voronoi Diagram")
    if xlims:
        plt.xlim(xlims)
    if ylims_voronoi:
        plt.ylim(ylims_voronoi)

    plt.subplot(235)
    lengths = hd.get_horocycle_arclens()
    z = np.zeros_like(lengths)
    z[0:-1] = hmt.z
    z[-1] = -1 # Plot the infinity weight at -1
    plt.stem(z, lengths)
    if xlims:
        sxlims = [xlims[0], xlims[1]]
        # Make sure infinity shows up
        if sxlims[0] > -1.5:
            sxlims[0] = -1.5
        plt.xlim(sxlims)
    if ylims_masses:
        plt.ylim(ylims_masses)
    plt.xticks(z, ["%.3g"%zi for zi in hmt.z] + ["$\infty$"])
    plt.title("Masses (Total Mass %.3g)"%np.sum(lengths))
    plt.xlabel("z")
    plt.ylabel("Mass")



def test_pentagon_infedges_edgecollapse(constraints = [('z', 0, 0), ('r', -1, 1)]):
    """
    Test an edge collapse of a pentagon whose edges
    all go to infinity
    """
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([1, 4]))
    cmt.root.addChildren([A, B])
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
        res = mergetree_to_hypmergetree(cmt, constraints)
        plt.clf()
        plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5])
        plt.savefig("%i.png"%i, bbox_inches='tight')
    
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([1, 4]))
    cmt.root.addChildren([A, B])
    C = MergeNode(np.array([0.5, 3]))
    D = MergeNode(np.array([2, 3]))
    F = MergeNode(np.array([3, 2]))
    B.addChildren([C, F])

    plt.clf()
    res = mergetree_to_hypmergetree(cmt, constraints)
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5])
    plt.savefig("%i.png"%N, bbox_inches='tight')

def test_pentagon_general_edgecollapse(constraints=[('z', 0, 0), ('r', -1, 1)]):
    """
    Test an edge collapse of a pentagon whose edges
    don't all go to infinity
    """
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 4]))
    cmt.root.addChildren([A, B])
    C = MergeNode(np.array([-3, 3]))
    D = MergeNode(np.array([-0.5, 2.8]))
    E = MergeNode(np.array([-4, 2]))
    A.addChildren([E, D])
    I = MergeNode(np.array([1, 2.6]))
    J = MergeNode(np.array([4, 2.3]))
    B.addChildren([J, I])

    plt.figure(figsize=(18, 12))
    N = 20
    for i, Iy in enumerate(np.linspace(2.6, 4, N)):
        I.X[1] = Iy
        res = mergetree_to_hypmergetree(cmt, constraints)
        plt.clf()
        plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5])
        plt.savefig("%i.png"%i, bbox_inches='tight')
    
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 4]))
    C = MergeNode(np.array([-3, 3]))
    D = MergeNode(np.array([-0.5, 2.8]))
    E = MergeNode(np.array([-4, 2]))
    A.addChildren([E, D])
    I = MergeNode(np.array([1, 2.6]))
    J = MergeNode(np.array([4, 2.3]))
    cmt.root.addChildren([A, J])

    plt.clf()
    res = mergetree_to_hypmergetree(cmt, constraints)
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5])
    plt.savefig("%i.png"%N, bbox_inches='tight')



def test_septagon_general_edgecollapse(constraints=[('z', 0, 0), ('r', -1, 1)]):
    """
    Test an edge collapse of a pentagon whose edges
    don't all go to infinity
    """
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 4]))
    cmt.root.addChildren([A, B])
    C = MergeNode(np.array([-3, 3]))
    D = MergeNode(np.array([-0.5, 2.8]))
    A.addChildren([C, D])
    E = MergeNode(np.array([-4, 2]))
    F = MergeNode(np.array([-2, 1.6]))
    C.addChildren([E, F])
    G = MergeNode(np.array([-3, 0]))
    H = MergeNode(np.array([-1, 0.5]))
    F.addChildren([G, H])
    I = MergeNode(np.array([1, 2.6]))
    J = MergeNode(np.array([4, 2.3]))
    B.addChildren([J, I])

    plt.figure(figsize=(18, 12))
    N = 20
    xlims=[-0.5, 6]
    ylims_voronoi=[0, 6.5]
    ylims_masses=[0, 5]
    for i, Iy in enumerate(np.linspace(2.6, 4, N)):
        I.X[1] = Iy
        res = mergetree_to_hypmergetree(cmt, constraints)
        plt.clf()
        plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=xlims, ylims_voronoi=ylims_voronoi, ylims_masses=ylims_masses)
        plt.savefig("%i.png"%i, bbox_inches='tight')
    
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 4]))
    C = MergeNode(np.array([-3, 3]))
    D = MergeNode(np.array([-0.5, 2.8]))
    A.addChildren([C, D])
    E = MergeNode(np.array([-4, 2]))
    F = MergeNode(np.array([-2, 1.6]))
    C.addChildren([E, F])
    G = MergeNode(np.array([-3, 0]))
    H = MergeNode(np.array([-1, 0.5]))
    F.addChildren([G, H])
    I = MergeNode(np.array([1, 2.6]))
    J = MergeNode(np.array([4, 2.3]))
    cmt.root.addChildren([A, J])

    plt.clf()
    res = mergetree_to_hypmergetree(cmt, constraints)
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=xlims, ylims_voronoi=ylims_voronoi, ylims_masses=ylims_masses)
    plt.savefig("%i.png"%N, bbox_inches='tight')

def test_pentagon_two_small_edges(constraints=[('z', 0, 0), ('r', -1, 1)]):
    """
    Test an edge collapse of a pentagon whose edges
    don't all go to infinity
    """
    cmt = MergeTree(TotalOrder2DX)
    cmt.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 4]))
    cmt.root.addChildren([A, B])
    C = MergeNode(np.array([-3, 3]))
    D = MergeNode(np.array([-0.5, 3.9]))
    E = MergeNode(np.array([-4, 2]))
    A.addChildren([E, D])
    I = MergeNode(np.array([1, 2.6]))
    J = MergeNode(np.array([4, 2.3]))
    B.addChildren([J, I])

    plt.figure(figsize=(18, 12))
    res = mergetree_to_hypmergetree(cmt, constraints)
    plt.clf()
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5.5])
    plt.savefig("Pentagon1.png", bbox_inches='tight')

    A.X = np.array([-3, 2.5])
    D.X = np.array([-2.5, 2.4])
    res = mergetree_to_hypmergetree(cmt, constraints)
    plt.clf()
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5.5])
    plt.savefig("Pentagon2.png", bbox_inches='tight')

if __name__ == '__main__':
    #test_pentagon_infedges_edgecollapse()
    #test_pentagon_infedges_edgecollapse([('z', 0, 0), ('z', 1, 1)])
    #test_pentagon_general_edgecollapse()
    #test_septagon_general_edgecollapse()
    test_pentagon_two_small_edges()