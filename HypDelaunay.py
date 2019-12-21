"""
Contains subroutines to represent a hyperbolic Delaunay triangulation
topologically, as well as a solver subroutines to obtain the geometry (zs/rs)
from a description of the weights of the associated merge tree
"""

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sys
from heapq import heappush, heappop
from MergeTree import *

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

def solvesystem(x0, ws, allidxs, verbose=False):
    res = opt.minimize(g, x0, args = (ws, allidxs), method='BFGS', jac=gradg)
    xsol = res['x']
    if verbose:
        print("g(x0) = ", g(x0, ws, allidxs))
        print("xsol = ", xsol)
        print("g(xsol) = ", g(xsol, ws, allidxs))
    return xsol




"""
Half-edge data structure for handling Delaunay triangulations of an 
arrangement of ideal hyperbolic vertices
"""


    
class HDTHedge(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation half-edge

    Attributes
    ----------
    face: HDTTri
        The face to the left of this half-edge
    pair: HDTHedge
        The half-edge paired with this edge
    prev: HDTHedge
        The previous half-edge in CCW order
    next: HDTHedge
        The next half-edge in CCW order

    weight: float
        Weight of the vedge that crosses this edge if it's
        an internal edge, or weight of the leaf vedge that terminates
        on this edge if it's a boundary edge
    internal_idx: int
        Index of the internal edge (or -1 if it is an external edge)
    is_aux: boolean
        If this is an internal edge, then this is
        either the auxiliary length, or the weight
        minus the auxiliary length
    """
    def __init__(self):
        self.face = None
        self.pair = None
        self.prev = None
        self.next = None
        self.head = None
        self.weight = -1
        self.internal_idx = -1
        self.is_aux = False
        self.index = -1
    
    def link_next_edge(self, n):
        """
        Make this edge's next pointer n, and make
        next's prev pointer this edge
        Parameters
        ----------
        n: HDTHedge
            The half-edge that should be next in order
        """
        self.next = n
        n.prev = self
    
    def add_pair(self, other):
        """
        Pair this edge with another edge
        other: HDTHedge
            The edge with which to pair this edge
        """
        self.pair = other
        other.pair = self
    
    def get_vertices(self):
        """
        Return the two vertices that are on this edge
        Returns
        -------
        vertices: [HDTVertex, HDTVertex]
            List of vertices in CCW order along edge
        """
        return [self.prev.head, self.head]




class HDTVertex(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation vertex
    
    Attributes
    ----------
    index: int
        The CCW order of this point on the boundary of the delaunay
        triangulation (-1 means infinity)
    h: HDTHedge
        Any half-edge on this vertex
    """
    def __init__(self, index=None):
        self.index = index
        self.h = None



class HDTTri(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation triangle
    Attributes
    ----------
    h: HDTEdge
        Any half-edge with this face as its face
    """
    def __init__(self):
        self.h = None
    
    def get_vertices(self):
        """
        Return a list of the HDTVertex objects that 
        make up this triangle
        Returns
        -------
        vertices: list of [HDTVertex]
            List of vertices on this triangle
        """
        vertices = [self.h.head]
        edge = self.h.next
        while not (edge == self.h):
            vertices.append(edge.head)
            edge = edge.next
        return vertices
    
    def get_edges(self):
        """
        Return a list of the HDTVertex objects that 
        make up this triangle
        Returns
        -------
        edges: list of [HDTHedge]
            List of half-edges with this triangle to the left
        """
        edges = [self.h]
        edge = self.h.next
        while not (edge == self.h):
            edges.append(edge)
            edge = edge.next
        return edges
        

class HyperbolicDelaunay(object):
    """
    The object that stores all half-edges, faces,
    and vertices of the triangulation, as well as
    functions to help initialize it
    """
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.triangles = []
    
    def init_from_mergetree_rec(self, etop, node):
        """
        Recursive helper function for initializing from merge tree
        Parameters
        ----------
        etop: [HDTEdge, HDTEdge]
            The pair of root half-edges in the Delaunay triangulation 
            associated to this merge tree vedge.  The first element
            should represent the half-edge on the boundary to which
            the new triangle will be connected, while the second element
            should be its pair on the interior of the polygon built so far
        node: MergeNode
            The merge tree node associated to this Delaunay triangle
        """
        # Step 1: Add weights based on height differences between 
        # this node and its parent
        for k in range(2):
            etop[k].weight = node.parent.X[-1] - node.X[-1]
        # Step 2: Add new vertex, triangle, and two pairs of half 
        # edges, and update all pointers
        if len(node.children) == 2:
            cleft, cright = node.children[0], node.children[1]
            if cleft.X[0] > cright.X[0]:
                cleft, cright = cright, cleft
            vnew = HDTVertex()
            self.vertices.append(vnew)
            trinew = HDTTri()
            trinew.h = etop[0]
            self.triangles.append(trinew)
            eleft = [HDTHedge(), HDTHedge()]
            eright = [HDTHedge(), HDTHedge()]
            self.edges += eleft + eright
            triedges = [etop, eleft, eright]
            prevbefore = etop[0].prev
            nextbefore = etop[0].next
            # Link up edges to each other and the new triangle
            for i, edges in enumerate(triedges):
                edges[0].face = trinew
                edges[0].add_pair(edges[1])
                edges[0].link_next_edge(triedges[(i+1)%3][0])
            prevbefore.link_next_edge(eright[1])
            eright[1].link_next_edge(eleft[1])
            eleft[1].link_next_edge(nextbefore)
            # Link up vertices and edges
            vs = [etop[0].head, vnew, etop[1].head]
            for i in range(3):
                triedges[i][0].head = vs[i]
                triedges[i][1].head = vs[(i-1)%3]
                vs[(i-1)%3].h = triedges[i][0]
            # Recurse on left and right subtrees
            eleft.reverse()
            self.init_from_mergetree_rec(eleft, cleft)
            eright.reverse()
            self.init_from_mergetree_rec(eright, cright)
        elif len(node.children) > 0:
            sys.stderr.write("ERROR: There are %i children a node in the merge tree"%N)



    def init_from_mergetree(self, MT, rootweight = 1.0):
        """
        Initialize this structure based on geometrically realized chiral 
        merge tree, using weights on edges as the height difference of the 
        corresponding vertices
        Parameters
        ----------
        MT: MergeTree
            The merge tree object from which to initialize this structure
        rootweight: float
            The weight of the root edge above the root vertex in the merge tree
            (Default 1.0)
        """
        ## Step 1: Create first triangle
        self.vertices = [HDTVertex(-1), HDTVertex(0), HDTVertex()]
        tri1 = HDTTri()
        self.triangles = [tri1]
        self.edges = []
        tri1_edges = [[HDTHedge(), HDTHedge()] for i in range(3)]
        # Initialize all pointers properly
        for i, [e1, e2] in enumerate(tri1_edges):
            e1.face = tri1
            e1.add_pair(e2)
            self.vertices[i].h = e1
            e1.head = self.vertices[(i+1)%3]
            self.vertices[(i+1)%3].h = e2
            e2.head = self.vertices[i]
            if i == 0:
                tri1.h = e1
                e1.weight = rootweight
                e2.weight = rootweight
            tri1_edges[i][0].link_next_edge(tri1_edges[(i+1)%3][0])
            tri1_edges[i][1].link_next_edge(tri1_edges[(i-1)%3][1])
            self.edges += [e1, e2]

        ## Step 2: Recursively construct the rest of the triangles
        cleft, cright = MT.root.children[0], MT.root.children[1]
        if cleft.X[0] > cright.X[0]:
            cleft, cright = cright, cleft
        tri1_edges[1].reverse()
        self.init_from_mergetree_rec(tri1_edges[1], cleft)
        tri1_edges[2].reverse()
        self.init_from_mergetree_rec(tri1_edges[2], cright)

        ## Step 3: Figure out order of ideal vertices
        ## by walking around the boundary edges and labeling the vertices
        index = 0
        ecurr = self.edges[1].prev
        while not (ecurr == self.edges[1]):
            ecurr.head.index = index
            index += 1
            ecurr = ecurr.prev

        ## Step 4: Index the internal edges
        index = 0
        for e in self.edges:
            if e.face and e.pair.face and (e.internal_idx == -1):
                e.internal_idx = index
                e.is_aux = True
                e.pair.internal_idx = index
                e.pair.is_aux = False
                index += 1

    def getAlpha(self, v, el, er):
        """
        Return the affine function of the internal edge lengths
        that gives rise to the horo arc length between two edges
        centered at a vertex
        Parameters
        ----------
        v: HDTVertex
            Center vertex
        el: HDTHedge
            Left edge
        er: HDTHedge
            Right edge
        """
        N = len(self.vertices)
        a = 0.0
        deltas = np.zeros(N-3)
        # TODO: Finish this

    def getEquations(self):
        """
        Return all of the information that's needed to setup equations
        """
        for i, v in enumerate(self.vertices):
            vidx = v.index
            edges = list(v.edges)
            v2idxs = np.array([e.vertexAcross(v).index for e in edges])
            
            for v2 in v2idxs:
                print("%i, %i"%(vidx, v2))
            print("\n\n")



    def render(self):
        """
        Render a regular convex polygon depicting the triangulation
        and its dual.  Number the vertices and indicate the weights
        associated to each edge
        """
        N = len(self.vertices)
        ## Step 1: Draw triangulation
        # Draw 0 and infinity at the top
        dTheta = 2*np.pi/N
        theta0 = np.pi/2 - dTheta/2
        Xs = np.zeros((N, 2))
        Xs[:, 0] = np.cos(theta0 + dTheta*np.arange(N))
        Xs[:, 1] = np.sin(theta0 + dTheta*np.arange(N))
        plt.scatter(Xs[:, 0], Xs[:, 1])
        for i in range(N):
            if i == 0:
                plt.text(Xs[i, 0], Xs[i, 1], "$\\infty$")
            else:
                plt.text(Xs[i, 0], Xs[i, 1], "%i"%(i-1))
        for edge in self.edges:
            v1, v2 = edge.head, edge.prev.head
            x = Xs[v1.index+1, :]
            y = Xs[v2.index+1, :]
            plt.plot([x[0], y[0]], [x[1], y[1]])
            x = 0.5*(x + y)
            s = "%.3g"%edge.weight
            if edge.internal_idx > -1:
                s = "%s (%i)"%(s, edge.internal_idx)
            plt.text(x[0], x[1], s)
        
        ## Step 2: Draw tree inside of triangulation
        for T in self.triangles:
            vs = T.get_vertices()
            vidxs = np.array([v.index+1 for v in vs])
            c1 = np.mean(Xs[vidxs, :], 0)
            plt.scatter(c1[0], c1[1], 60, 'k')
            for e in T.get_edges():
                T2 = e.pair.face
                if T2:
                    idxs = np.array([v.index+1 for v in T2.get_vertices()])
                    c2 = np.mean(Xs[idxs, :], 0)
                else:
                    idxs = np.array([v.index+1 for v in e.get_vertices()])
                    c2 = np.mean(Xs[idxs, :], 0)
                    plt.scatter(c2[0], c2[1], 60, 'k')
                plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k')


if __name__ == '__main__':
    T = MergeTree(TotalOrder2DX)
    T.root = MergeNode(np.array([0, 5]))
    A = MergeNode(np.array([-1, 4]))
    B = MergeNode(np.array([2, 3]))
    T.root.addChildren([A, B])
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
    K = MergeNode(np.array([-4.1, 1]))
    L = MergeNode(np.array([-3.1, 1]))
    #E.addChildren([K, L])
    
    hd = HyperbolicDelaunay()
    hd.init_from_mergetree(T)

    plt.subplot(121)
    T.render(offset=np.array([0, 0]))
    plt.subplot(122)
    hd.render()
    plt.show()