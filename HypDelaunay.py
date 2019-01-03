"""
Contains subroutines to represent a hyperbolic Delaunay triangulation
topologically, as well as a solver subroutines to obtain the geometry (zs/rs)
from a description of the weights of the associated merge tree
"""

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sys
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
Classes for handling Delaunay triangulations of an arrangement of
ideal hyperbolic vertices
"""

class HDTVertex(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation vertex
    Parameters
    ----------
    index: int
        The CCW order of this point on the boundary of the delaunay
        triangulation (-1 means infinity)
    """
    def __init__(self, index):
        self.index = index
        self.edges = set([]) # Delaunay edges emanating from this vertex
    
class HDTEdge(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation edge
    """
    def __init__(self, v1, v2, internal, weight):
        """
        Initialize an edge object, and update the two vertices to 
        have this edge in their list of incident edges
        Parameters
        ----------
        v1: HDTVertex
            First vertex on this edge
        v2: HDTVertex
            Second vertex on this edge
        internal: boolean
            True if an internal edge, false if a boundary edge
        weight: float
            Weight of the vedge that crosses this edge if it's
            an internal edge, or weight of the leaf vedge that terminates
            on this edge if it's a boundary edge
        """
        self.v1, self.v2 = v1, v2
        self.f1, self.f2 = None, None
        self.internal = internal
        self.weight = weight
        v1.edges.add(self)
        v2.edges.add(self)

    def vertexAcross(self, startV):
        if startV == self.v1:
            return self.v2
        if startV == self.v2:
            return self.v1
        sys.stderr.write("Warning (vertexAcross): Vertex not member of edge\n")
        return None

    def addFace(self, face):
        if self.f1 == None:
            self.f1 = face
        elif self.f2 == None:
            self.f2 = face
        else:
            sys.stderr.write("Cannot add face to edge; already 2 there\n")
            return False
        return True
    
    def faceAcross(self, startF):
        if startF == self.f1:
            return self.f2
        if startF == self.f2:
            return self.f1
        sys.stderr.write("Warning (faceAcross): Face not member of edge\n")
        return None

class HDTTri(object):
    """
    A class for storing an ideal hyperbolic Delaunay Triangulation triangle
    """
    def __init__(self, edges):
        """
        Initialize a triangle from three edges, and update the edges
        to point to this triangle
        Parameters
        ----------
        edges: list (HDTEdge)
            A list of 3 edges making up the triangle.  The first
            edge is the root edge, the second edge is the left edge,
            and the third edge is the right edge
        """
        self.edges = edges
        for e in edges:
            e.addFace(self)

class HyperbolicDelaunay(object):
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.triangles = []
    
    def getEdge(self, v1, v2):
        edge = v1.edges & v2.edges
        if len(edge) > 1:
            sys.stderr.write("Warning: More than one edge found on vertex list intersection\n")
        for e in edge:
            return e
        return None

    def addVertex(self, index = -1):
        v = HDTVertex(index)
        self.vertices.append(v)
        return v

    def addEdge(self, v1, v2, internal, weight):
        edge = HDTEdge(v1, v2, internal, weight)
        self.edges.append(edge)
        return edge

    def addTriangle(self, edges):
        tri = HDTTri(edges)
        self.triangles.append(tri)
        return tri
    
    def init_from_mergetree_rec(self, edge, node):
        """
        Recursive helper function for initializing from merge tree
        Parameters
        ----------
        edge: HDTEdge
            The root edge in the Delaunay triangulation associated
            to this merge tree vedge
        node: MergeNode
            The merge tree node associated to this Delaunay triangle
        Returns
        -------
        internal: boolean
            Whether this edge is an external edge
        """
        N = len(node.children)
        if N == 0:
            return False
        elif not (N == 2):
            sys.stderr.write("ERROR: There are %i children a node in the merge tree"%N)
            return False
        cleft, cright = node.children[0], node.children[1]
        if cleft.X[0] > cright.X[0]:
            cleft, cright = cright, cleft
        vnew = self.addVertex()
        eleft = self.addEdge(edge.v1, vnew, internal=True, weight=node.X[-1]-cleft.X[-1])
        eright = self.addEdge(edge.v2, vnew, internal=True, weight=node.X[-1]-cright.X[-1])
        # Recurse on left and right subtrees
        if not self.init_from_mergetree_rec(eleft, cleft):
            eleft.internal = False
        if not self.init_from_mergetree_rec(eright, cright):
            eright.internal = False
        return True


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
        ## Step 1: Build triangulation
        vinf = self.addVertex(-1)
        v0 = self.addVertex(0)
        eroot = self.addEdge(vinf, v0, internal=False, weight=rootweight)
        self.init_from_mergetree_rec(eroot, MT.root)

        ## Step 2: Figure out order of ideal vertices
        for v in self.vertices:
            v.visited = False
        # First, check that every ideal vertex has exactly two internal
        # vertices connected to it by edges
        for v in self.vertices:
            count = 0
            for e in v.edges:
                if not e.internal:
                    count += 1
            assert(count == 2)
        # Now, walk around the boundary edges and label the vertices
        vcurr = v0
        eright = eroot
        index = 0
        while not (vcurr == vinf):
            vcurr.index = index
            index += 1
            # Move to the next edge in CCW order
            eright = [e for e in vcurr.edges.difference([eright]) if (not e.internal)][0]
            vcurr = eright.vertexAcross(vcurr)
    
    def render(self):
        """
        Render a regular convex polygon depicting the triangulation
        and its dual.  Number the vertices and indicate the weights
        associated to each edge
        """
        N = len(self.vertices)
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
            v1, v2 = edge.v1, edge.v2
            x = Xs[v1.index+1, :]
            y = Xs[v2.index+1, :]
            plt.plot([x[0], y[0]], [x[1], y[1]])
            x = 0.5*(x + y)
            plt.text(x[0], x[1], "%.3g"%edge.weight)
            

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
    
    hd = HyperbolicDelaunay()
    hd.init_from_mergetree(T)

    plt.subplot(121)
    T.render(offset=np.array([0, 0]))
    plt.subplot(122)
    hd.render()
    plt.show()