"""
Contains subroutines to represent a hyperbolic Delaunay triangulation
topologically, as well as a solver subroutines to obtain the geometry (zs/rs)
from a description of the weights of the associated merge tree
"""

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from heapq import heappush, heappop
from MergeTree import *
from GeomTools import *

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

    index: int
        Index of the full edge that this half-edge is part of
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
        self.index = -1
        self.internal_idx = -1
        self.is_aux = False
    
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
    
    def __str__(self):
        """
        Return a string with the indices of the edge
        and the vertices this half-edge goes between
        """
        v1, v2 = self.get_vertices()
        s = "HDTEdge(%i) from %i to %i"%(self.index, v1.index, v2.index)
        return s




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

    Attributes
    ----------
    vertices: list of HDTVertex
        An unordered list of vertices
    edges: list of HDTHedge
        An unordered list of half-edges
    triangles: list of HDTTriangle
        An unordered list of triangles
    weight_dict: {int: float}
        A dictionary to convert from an edge
        index to the weight of the edge
    """
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.triangles = []
        self.weight_dict = {}
    
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
        index = len(self.weight_dict)
        self.weight_dict[index] = node.parent.X[-1] - node.X[-1]
        for k in range(2):
            etop[k].index = index
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
                self.weight_dict[0] = rootweight
                e1.index = 0
                e2.index = 0
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
    
    def get_equations(self):
        """
        Return all of the information that's needed to setup equations
        Returns
        -------
        equations: list of dict{
            'w': z index of the w point (or -1 if infinity),
            'x': z index of the x point (or -1 if infinity),
            'y': z index of the y point (or -1 if infinity),
            'uidxs': list of indices into to the weight dictionary
                    for constant weights involved in this edge,
            'vs': {Index of v variable: coefficient of v variable}
        }
        """
        e1 = self.edges[0]
        N = len(self.vertices)
        equations = []
        for index in range(N):
            e2 = e1.next
            # Figure out the vertex index at the head
            # of the last edge
            next_index = index + 1
            if next_index == N-1:
                next_index = -1
            elif next_index == N:
                next_index = 0
            # Figure out the edges that are involved
            e2s = [e2]
            uidxs = [e1.index]
            while not (e2.head.index == next_index):
                e2 = e2.pair.next
                e2s.append(e2)
            for e2 in e2s:
                vs = {}
                if e2.internal_idx == -1:
                    uidxs.append(e2.index)
                else:
                    if e2.is_aux:
                        vs[e2.internal_idx] = 1
                    else:
                        uidxs.append(e2.index)
                        vs[e2.internal_idx] = -1
                if index == N-1:
                    index = -1
                eq = {'w':e1.prev.head.index, 'x':index, 'y':e2.head.index, 'uidxs':uidxs[::], 'vs':vs}
                if eq['y'] < eq['w']:
                    eq['y'], eq['w'] = eq['w'], eq['y']
                equations.append(eq)
                if (e2.internal_idx > -1) and e2.is_aux:
                    # The full edge will be used in the next equation
                    uidxs.append(e2.index)
            e1 = e2
        return equations
    
    def get_zvals(self, eq, zs, x0=0):
        """
        Given the default z coordinate of x0 and
        the rest of the N-1 z coordinates, return
        the actual coordinates of w, x, and y, or
        None if they happen to be infinity

        Parameters
        ----------
        eq: Dictionary
            A dictionary element returned by self.get_equations()
        zs: ndarray(N-2)
            X coordinates of the zs, excluding z_{\infty} and z0
        x0: float
            Default value of x0
        Returns
        -------
        w: float
            x coordinate of w
        x: float
            x coordinate of x
        y: float
            x coordinate of y
        """
        vals = [None, None, None]
        for i, zidx in enumerate([eq['w'], eq['x'], eq['y']]):
            if zidx == 0:
                vals[i] = x0
            elif zidx > 0:
                vals[i] = zs[zidx-1]
        return vals[0], vals[1], vals[2]

    def fi(self, eq, zs, rs, vs, x0=0, rinf = 1):
        """
        Set up an objective function for satisfying a particular equation
        Parameters
        ----------
        eq: Dictionary
            A dictionary element returned by self.get_equations()
        zs: ndarray(N-2)
            X coordinates of the zs, excluding z_{\infty} and z0
        rs: ndarray(N-1)
            R values of everything excuting r_{\infty}
        vs: ndarray(N-3)
            Values of internal edge auxiliary variables
        x0: float
            Default value of x0
        rinf: float
            Default value of r_{\infty}
        Returns
        -------
        {'res': fi(zs, rs, vs): float
            The evaluation of the function,
         'alpha': float
            alpha as a biproduct, which can be reused for the gradient}
        """
        alpha = 0
        for u in eq['uidxs']:
            alpha += self.weight_dict[u]
        for v in eq['vs']:
            coeff = eq['vs'][v]
            alpha += coeff*vs[v]
        res = 0
        uidxs, vs = eq['uidxs'], eq['vs']
        r = rinf
        if eq['x'] > -1:
            r = rs[eq['x']]
        w, x, y = self.get_zvals(eq, zs, x0)
        if w is None:
            # Case 2
            res = (r - alpha*(y-x))**2
        elif y is None:
            # Case 3
            res = (r - alpha*(x-w))**2
        elif x is None:
            # Case 4
            res = (rinf*alpha - y + w)**2
        else:
            # Case 1
            res = (r*(y-w) - alpha*(x-w)*(y-x))**2
        return {'res':res, 'alpha':alpha}

    def unpack_variables(self, x):
        """
        Unpack the 3N-6 variables into zs, rs, and vs
        Parameters
        ----------
        x: ndarray(3N-6)
            A concatenation of [zs(N-2), rs(N-1), vs(N-3)]
        Returns
        -------
        zs: ndarray(N-2)
            z positions
        rs: ndarray(N-1)
            Radii
        vs: ndarray(N-3)
            Auxiliary variables
        """
        N = len(self.vertices)
        zs = x[0:N-2]
        rs = x[N-2:N-2+N-1]
        vs = x[-(N-3)::]
        return zs, rs, vs

    def f(self, x, equations, x0=0, rinf=1):
        """
        Return the objective function sum_i f_i(x)
        Parameters
        ----------
        x: ndarray(3*N-6)
            Current estimate of all variables; a concatenation of
            [zs(N-2), rs(N-1), vs(N-3)]
        equations: dict
            Equations returned by self.get_equations()
        x0: float
            Default value of x0
        rinf: float
            Default value of r_{\infty}
        Returns
        -------
        res: float
            An evaluation of f with the given variables
        """
        res = 0.0
        zs, rs, vs = self.unpack_variables(x)
        for eq in equations:
            res += self.fi(eq, zs, rs, vs, x0, rinf)['res']
        return 0.5*res

    def solve_equations(self, zs0, rs0, vs0, x0=0, rinf=1, verbose=False):
        """
        Solve for the zs, rs, and auxiliary variables vs which
        give rise to the merge tree with this topology and weights
        Parameters
        ----------
        zs0: ndarray(N-2)
            X coordinates of the initial zs, excluding z_{\infty} and z0
        rs0: ndarray(N-1)
            R values of the initial rs, excluding r_{\infty}
        vs0: ndarray(N-3)
            Initial values of internal edge auxiliary variables
        x0: float
            Default value of x0
        rinf: float
            Default value of r_{\infty}
        """
        x_initial = np.concatenate([zs0, rs0, vs0])
        equations = self.get_equations()
        res = opt.minimize(self.f, x_initial, args = (equations, x0, rinf), method='BFGS')#, jac=gradf)
        x_sol = res['x']
        zs, rs, vs = self.unpack_variables(x_sol)
        fx_initial = self.f(x_initial, equations, x0, rinf)
        fx_sol = self.f(x_sol, equations, x0, rinf)
        return {'zs':zs, 'rs':rs, 'vs':vs, 'fx_initial':fx_initial, 'fx_sol':fx_sol}

    def render(self, symbolic=True):
        """
        Render a regular convex polygon depicting the triangulation
        and its dual.  Number the vertices and indicate the weights
        associated to each edge
        Parameters
        ----------
        symbolic: boolean
            If true, write variables for the edge weights
            If false, use the actual floating point edge lengths
        """
        N = len(self.vertices)
        ## Step 1: Draw the polygon
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
        
        ## Step 2: Draw the triangulation and the 
        ##         tree inside of triangulation
        edge_drawn = [False]*int(len(self.edges)/2)
        for T in self.triangles:
            vs = T.get_vertices()
            vidxs = np.array([v.index+1 for v in vs])
            c1 = np.mean(Xs[vidxs, :], 0)
            plt.scatter(c1[0], c1[1], 60, 'k')
            for e in T.get_edges():
                if not edge_drawn[e.index]:
                    edge_drawn[e.index] = True
                    # Draw edge on triangle
                    v1, v2 = e.head, e.prev.head
                    xedge = Xs[[v1.index+1, v2.index+1], :]
                    plt.plot(xedge[:, 0], xedge[:, 1], c='C1')
                    # Draw edge on tree
                    T2 = e.pair.face
                    if T2:
                        idxs = np.array([v.index+1 for v in T2.get_vertices()])
                        c2 = np.mean(Xs[idxs, :], 0)
                    else:
                        idxs = np.array([v.index+1 for v in e.get_vertices()])
                        c2 = np.mean(Xs[idxs, :], 0)
                    plt.scatter(c2[0], c2[1], 60, 'k')
                    xtree = np.array([c1, c2])
                    plt.plot(xtree[:, 0], xtree[:, 1], 'k')
                    # Draw text for weights and auxiliary variables
                    weight = self.weight_dict[e.index]
                    s1 = "$u_{%i}$"%e.index
                    if not symbolic:
                        s1 = "%.3g"%weight
                    if e.internal_idx == -1:
                        xtree = np.mean(xtree, 0)
                        plt.text(xtree[0], xtree[1], s1)
                    else:
                        saux = "$v_{%i}$"%e.internal_idx
                        sother = "%s - %s"%(s1, saux)
                        xint = intersectSegments2D(xedge, xtree)
                        plt.scatter(xint[0], xint[1], 10, c='k')
                        for c, aux in zip([c1, c2], [e.is_aux, not e.is_aux]):
                            xcenter = np.array([c, xint])
                            xcenter = np.mean(xcenter, 0)
                            if aux:
                                s = saux
                            else:
                                s = sother
                            plt.text(xcenter[0], xcenter[1], s)
        plt.axis('off')
    
    def get_equations_tex(self, symbolic=True):
        """
        Return tex code for the equations that come from this triangulation
        Parameters
        ----------
        symbolic: boolean
            If true, write variables for the edge weights
            If false, use the actual floating point edge lengths
        """
        equations_str = "$r_{\infty} = 1$\n$z_{\infty}=\infty$\n$z_0=0$\n"
        for eq in self.get_equations():
            w, x, y, uidxs, vs = eq['w'], eq['x'], eq['y'], eq['uidxs'], eq['vs']
            s = "$"
            alpha_str = ""
            for i, u in enumerate(uidxs):
                if symbolic:
                    alpha_str += "u_{%i}"%u
                else:
                    alpha_str += "%.3g"%self.weight_dict[u]
                if i < len(uidxs) - 1:
                    alpha_str += " + "
            for v in vs:
                if vs[v] > 0:
                    alpha_str += " + "
                else:
                    alpha_str += " - "
                alpha_str += "v_{%i}"%v
            if w == -1:
                # Case 2
                s += "r_{%i} = (%s)(z_{%i}-z_{%i})"%(x, alpha_str, y, x)
            elif y == -1:
                # Case 3
                s += "r_{%i} = (%s)(z_{%i}-z_{%i})"%(x, alpha_str, x, w)
            elif x == -1:
                # Case 4
                s += "r_{\infty}(%s) = (z_{%i}-z_{%i})"%(alpha_str, y, w)
            else:
                s += "r_{%i}(z_{%i}-z_{%i}) = ("%(x, y, w)
                s += alpha_str
                s += ")(z_{%i}-z_{%i})(z_{%i}-z_{%i})"%(x, w, y, x)
            equations_str += s + "$\n"
        return equations_str
        



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
    symbolic=True

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    T.render(offset=np.array([0, 0]))
    plt.title("Merge Tree")
    plt.subplot(132)
    hd.render(symbolic=symbolic)
    plt.title("Topological Triangulation")
    plt.subplot(133)
    plt.text(0, 0, hd.get_equations_tex(symbolic=symbolic))
    plt.axis('off')
    plt.title("Hyperbolic Equations")
    filename = "%iGon.svg"%len(hd.vertices)
    if symbolic:
        filename = "%iGon_symbolic.svg"%len(hd.vertices)
    plt.savefig(filename, bbox_inches='tight')