from os import posix_spawnp
import numpy as np 
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush, heappop
from MergeTree import *
from HypDelaunay import *

CW_ERR_MSG = "Trying to do CC rotation where CC rotation is impossible"
CCW_ERR_MSG = "Trying to do CCW rotation where CCW rotation is impossible"

class BinaryNode(object):
    def __init__(self, parent = None):
        self.parent = parent
        self.left = None
        self.right = None
        self.order = 0
        self.size = 1
        self.weight = 0
        self.al = -np.pi/10 # Angle left
        self.ar = np.pi/10 # Angle right
        self.pos = np.array([0, 0])
    
    def update_pos(self, angle=0, pos=np.array([0, 0])):
        """
        Recursively update the positions based on angle

        Parameters
        ----------
        angle: float
            Angle in radians pointing into this node
        pos: ndarray(2)
            x/y coordinates of this node
        """
        self.pos = pos
        if self.left:
            al = angle + self.al
            posl = pos + np.array([np.cos(al), np.sin(al)])
            self.left.update_pos(al, posl)
        if self.right:
            ar = angle + self.ar
            posr = pos + np.array([np.cos(ar), np.sin(ar)])
            self.right.update_pos(ar, posr)
    
    def draw(self, leaf_idx, internal_idx, angle=0):
        pos = self.pos
        plt.scatter([pos[0]], [pos[1]], 20, c='k')
        if not self.left:
            plt.text(pos[0], pos[1], "({}, {:.1f})".format(leaf_idx[0], angle*180/np.pi))
            leaf_idx[0] += 1
        else:
            #plt.text(pos[0], pos[1], "{}, ({:.1f}, {:.1f})".format(internal_idx[0], self.al*180/np.pi, self.ar*180/np.pi))
            pos2 = self.left.pos
            plt.plot([pos[0], pos2[0]], [pos[1], pos2[1]], c='k')
            if self.left.left:
                internal_idx[0] += 1
            self.left.draw(leaf_idx, internal_idx, angle+self.al)

            pos2 = self.right.pos
            plt.plot([pos[0], pos2[0]], [pos[1], pos2[1]], c='k')
            if self.right.right:
                internal_idx[0] += 1
            self.right.draw(leaf_idx, internal_idx, angle+self.ar)
        
    def get_angle_equations(self, eq, all_eq, N, internal_idx, leaf_idx):
        """
        Get angle equations

        Parameters
        ----------
        eq: list of [[idx, coeff]]
            Current equation being constructed
        all_eq: Dictionary of [list of [idx, coeff], value]
            All equations
        N: int
            Total number of leaf nodes
        internal_idx: list of [int]
            Preorder index of current internal node
        leaf_idx: list of [int]
            Inorder index of the current leaf node
        """
        ## TODO: Fix inorder of leaf nodes
        if not self.left and not self.right:
            eq.append(2*np.pi*leaf_idx[0]/N)
            all_eq[leaf_idx[0]] = eq.copy()
            leaf_idx[0] += 1
        else:
            # Recursively create equations towards the left
            idx = internal_idx[0]
            eql = eq.copy()
            eql.append([idx*2, 1])
            if self.left.left:
                internal_idx[0] += 1
            self.left.get_angle_equations(eql, all_eq, N, internal_idx, leaf_idx)
            # Recursively create equations towards the right
            eqr = eq.copy()
            eqr.append([idx*2+1, 1])
            if self.right.right:
                internal_idx[0] += 1
            self.right.get_angle_equations(eqr, all_eq, N, internal_idx, leaf_idx)

    def set_angles(self, internal_idx, angles):
        """
        Set the angles of all internal nodes

        Parameters
        ----------
        internal_idx: list of [int]
            Preorder index of current internal node
        angles: ndarray(2*n_internal)
            All of the internal angles
        """
        ## TODO: Fix inorder of leaf nodes
        if self.left and self.right: # Internal node
            # Recursively create equations towards the left
            idx = internal_idx[0]
            self.al = angles[idx*2]
            self.ar = angles[idx*2+1]
            if self.left.left:
                internal_idx[0] += 1
            self.left.set_angles(internal_idx, angles)
            # Recursively create equations towards the right
            if self.right.right:
                internal_idx[0] += 1
            self.right.set_angles(internal_idx, angles)

    def update_inorder(self, node_deque):
        """
        Compute the order of this node according to an 
        inorder traversal
        Parameters
        ----------
        node_deque: deque
            A growing list of nodes in order
        """
        if self.left:
            self.left.update_inorder(node_deque)
        node_deque.append(self)
        if self.right:
            self.right.update_inorder(node_deque)
    
    def update_subtree_size(self):
        """
        Recursively compute the size of the subtree
        """
        size = 1
        if self.left:
            size += self.left.update_subtree_size()
        if self.right:
            size += self.right.update_subtree_size()
        self.size = size
        return size
    
    def update_subtree_weight(self):
        """
        Recursively compute the weight of each subtree
        (i.e. the number of leaf nodes in that subtree)
        Returns
        -------
        weight: int
            The weight of this subtree (also updates member
            variable as a side effect)
        """
        weight = 0
        if not self.left and not self.right:
            # Leaf node
            weight = 1
        else:
            if self.left:
                weight += self.left.update_subtree_weight()
            if self.right:
                weight += self.right.update_subtree_weight()
        self.weight = weight
        return weight
        
    def get_weight_sequence(self):
        """
        Recursively compute the weight sequence, assuming
        that every non-leaf node has exactly two children.
        This function also assumes that the weights have been
        properly computed for each subtree ahead of time
        """
        result = [] # Leaf node by default
        if self.left and self.right:
            result = self.left.get_weight_sequence()
            result.append(self.left.weight)
            result += self.right.get_weight_sequence()
        return result
    
    def can_rotate_ccw(self):
        """
        See whether it is possible to perform
        a counter-clockwise tree rotation about this node
        """
        result = False
        A = self.left
        b = self.right
        if A and b:
            B = b.left
            C = b.right
            if B and C:
                result = True
        return result
    
    def rotate_ccw(self):
        """
        Perform a counter-clockwise rotation about this node
        Returns
        -------
        b: BinaryNode
            The new root node of this subtree
        """
        root_parent = self.parent
        root_left_child = True
        if root_parent:
            if root_parent.left == self:
                root_left_child = True
            else:
                root_left_child = False
        a = self
        A = a.left
        b = a.right
        if A and b:
            B = b.left
            C = b.right
            if B and C:
                a.right = B
                B.parent = a
                b.left = a
                a.parent = b
                b.parent = root_parent
                if root_parent:
                    if root_left_child:
                        root_parent.left = b
                    else:
                        root_parent.right = b
            else:
                raise Exception(CCW_ERR_MSG)
        else:
            raise Exception(CCW_ERR_MSG)
        return b
    
    def can_rotate_cw(self):
        """
        See whether it's possible to perform
        a clockwise tree rotation about this node
        """
        result = False
        a = self.left
        C = self.right
        if a and C:
            A = a.left
            B = a.right
            if A and B:
                result = True
        return result

    def rotate_cw(self):
        """
        Perform a clockwise rotation about this node
        Returns
        -------
        b: BinaryNode
            The new root node of this subtree
        """
        root_parent = self.parent
        root_left_child = True
        if root_parent:
            if root_parent.left == self:
                root_left_child = True
            else:
                root_left_child = False
        b = self
        a = self.left
        C = self.right
        if a and C:
            A = a.left
            B = a.right
            if A and B:
                a.right = b
                b.parent = a
                b.left = B
                B.parent = b
                a.parent = root_parent
                if root_parent:
                    if root_left_child:
                        root_parent.left = a
                    else:
                        root_parent.right = a
            else:
                raise Exception(CW_ERR_MSG)
        else:
            raise Exception(CW_ERR_MSG)
        return a

class BinaryTree(object):
    def __init__(self):
        self.root = BinaryNode()

    def draw(self):
        if self.root:
            leaf_idx = [0]
            internal_idx = [0]
            self.root.update_pos()
            self.root.draw(leaf_idx, internal_idx)
            plt.axis("equal")

    def update_inorder(self):
        """
        Update the order of all nodes using inorder
        Returns
        -------
        nodes_deque: collections.deque
            A deque of BinaryNode objects in inorder
        """
        nodes_deque = deque()
        self.root.update_inorder(nodes_deque)
        for i, n in enumerate(nodes_deque):
            n.order = i
        return nodes_deque

    def get_angle_equations(self):
        all_eq = {}
        nodes = self.update_inorder()
        N = (len(nodes)+1)//2
        print("n_leaves = ", N)
        if self.root:
            eq = []
            internal_idx = [0]
            leaf_idx = [0]
            self.root.get_angle_equations(eq, all_eq, N, internal_idx, leaf_idx)
        return all_eq
    
    def set_angles(self, angles):
        if self.root:
            internal_idx = [0]
            self.root.set_angles(internal_idx, angles)

    def to_merge_tree_rec(self, y, node, m_node):
        """
        Wrap into a merge tree object
        Parameters
        ---------
        height: float
            The y-position of this node
        node: BinaryNode
            The current binary node
        m_node: MergeNode
            The corresponding merge tree node
        """
        if node.left:
            XL = np.array([node.left.order, y-1])
            left = MergeNode(XL)
            m_node.addChild(left)
            self.to_merge_tree_rec(y-1, node.left, left)
        if node.right:
            XR = np.array([node.right.order, y-1])
            right = MergeNode(XR)
            m_node.addChild(right)
            self.to_merge_tree_rec(y-1, node.right, right)
    
    def to_merge_tree(self):
        """
        Convert tree to a merge tree object
        Returns
        -------
        T: MergeTree
        """
        self.update_inorder()
        y = 0
        root = MergeNode(np.array([self.root.order, y]))
        self.to_merge_tree_rec(y, self.root, root)
        MT = MergeTree()
        MT.root = root
        return MT
    
    def to_triangulation(self):
        """
        Return a triangulation corresponding to this tree,
        using unit edge weights
        Returns
        -------
        T: HyperbolicDelaunay
            The corresponding triangulation
        """
        MT = self.to_merge_tree()
        T = HyperbolicDelaunay()
        T.init_from_mergetree(MT)
        return T
    
    def get_weight_sequence(self):
        """
        Return the weight sequence for this tree
        """
        self.root.update_subtree_weight()
        return self.root.get_weight_sequence()
    
    def get_rotation_neighbors_rec(self, node, neighbors):
        """
        Paramters
        ---------
        node: BinaryNode
            The current node being examined
        neighbors: list
            A growing list of the weight sequences. Each
            element is a dictionary {'node':BinaryNode, 'dir':'CCW' or 'CW', 
                                     'w':weight sequence}
        """
        updated = False
        is_root = node == self.root
        if node.can_rotate_ccw():
            updated = True
            subtree_root = node.rotate_ccw()
            if is_root:
                self.root = subtree_root
            w = self.get_weight_sequence()
            neighbors.append({'node':node, 'dir':'CCW', 'w':w})
            # Now switch back
            subtree_root = subtree_root.rotate_cw()
            if is_root:
                self.root = subtree_root
        if node.can_rotate_cw():
            updated = True
            subtree_root = node.rotate_cw()
            if is_root:
                self.root = subtree_root
            w = self.get_weight_sequence()
            neighbors.append({'node':node, 'dir':'CW', 'w':w})
            # Now switch back
            subtree_root = subtree_root.rotate_ccw()
            if is_root:
                self.root = subtree_root
        if updated:
            # Need to change the subtree weights back
            self.root.update_subtree_weight()
        
        if node.left:
            self.get_rotation_neighbors_rec(node.left, neighbors)
        if node.right:
            self.get_rotation_neighbors_rec(node.right, neighbors)


    def get_rotation_neighbors(self):
        """
        Return a list of weight sequences corresponding
        to trees that are reachable by a CW or CCW rotation
        from this tree
        """
        self.update_inorder()
        neighbors = []
        self.get_rotation_neighbors_rec(self.root, neighbors)
        return neighbors


def weightsequence_to_binarytree(pws):
    """
    Convert a weight sequence into a binary tree object by 
    pairing the appropriate nodes
    """
    from collections import deque
    ws = [w for w in pws]
    ws.append(len(ws)+1) # The last element is implied
    N = len(ws)
    nodes = [BinaryNode() for i in range(N)]
    i = 0
    while i < len(ws):
        k = 0
        while k < ws[i]-1:
            # Pair n(i), n(i-1)
            parent = BinaryNode()
            parent.left = nodes[i-1]
            nodes[i-1].parent = parent
            parent.right = nodes[i]
            nodes[i].parent = parent
            k += ws[i-1]
            # Coalesce two nodes
            # TODO: A more efficient way to do this would be
            # using a linked list
            ws = ws[0:i-1] + ws[i::]
            nodes = nodes[0:i-1] + [parent] + nodes[i+1::]
            i -= 1
        i += 1
    T = BinaryTree()
    T.root = nodes[0]
    return T

def enumerate_weightsequences(N):
    """
    Enumerate all of the weight sequences
    for a tree with N internal nodes
    Parameters
    ----------
    N: int
        Number of internal nodes
    """
    ws = [np.ones(N, dtype=int)]
    w = np.ones(N, dtype=int)
    finished = False
    while not finished:
        i = N-1
        while w[i] >= i+1 and i >= 0:
            i -= 1
        if i == -1:
            finished = True
        else:
            j = i - w[i]
            w[i] += w[j]
            for m in range(i+1, N):
                w[m] = 1
            ws.append(np.array(w))
    return ws

def get_rotation_sequence(BT1, BT2, verbose = False):
    """
    Compute a sequence that realizes the optimal rotation dist
    between two trees, using brute force breadth-first search
    Parameters
    ----------
    BT1: BinaryTree
        First tree
    BT2: BinaryTree
        Second tree
    verbose: boolean
        Whether to print out stats
    Returns
    -------
    sequence: list of lists
        List of weight sequences along the path from start to finish.  
        Length of the list is one more than the rotation distance
    """
    import json
    prev = {} # Stores the tree that came directly
    # before this tree in a shortest path
    dist = {} # Stores the distance from BT1 to this tree
    w1 = BT1.get_weight_sequence()
    w1_str = "{}".format(w1)
    w2 = BT2.get_weight_sequence()
    w2_str = "{}".format(w2)
    h = []  # Heap will contain (distance, weight string, prev weight string)
    # Setup initial neighbors from first tree
    for neighb in BT1.get_rotation_neighbors():
        s = "{}".format(neighb['w'])
        heappush(h, (1, s, w1_str))
    num_expanded = 1
    # Perform breadth-first search
    while len(h) > 0:
        (d, s, p) = heappop(h)
        if not s in dist:
            num_expanded += 1
            dist[s] = d
            prev[s] = p
            if s == w2_str:
                break
            # Add on neighbors
            T = weightsequence_to_binarytree(json.loads(s))
            for neighb in T.get_rotation_neighbors():
                ns = "{}".format(neighb['w'])
                heappush(h, (d+1, ns, s))
    # Now backtrace to find sequence from BT1 to BT2
    sequence = [w2_str]
    while prev[sequence[-1]] != w1_str:
        sequence.append(prev[sequence[-1]])
    sequence.append(w1_str)
    sequence = sequence[::-1]
    sequence = [json.loads(s) for s in sequence]
    if verbose:
        print("dist(BT2) = ", dist[w2_str])
        print("len(sequence) = ", len(sequence))
        print("{} Expanded".format(num_expanded))
    return sequence

def alpha_sequence_heuristic(T1, T2):
    """
    Return the L1 alpha sequence distance between two trees
    with the same number of nodes
    Parameters
    ----------
    T1: HyperbolicDelaunay
        First tree
    T2: HyperbolicDelaunay
        Second tree
    Returns
    -------
    float: the distance
    """
    alpha1 = T1.get_horocycle_arclens()
    alpha2 = T2.get_horocycle_arclens()
    return 0.25*np.sum(np.abs(alpha1-alpha2))

def robinson_folds_heuristic(T1, T2):
    """
    Return the Robinson-Foulds distance between two trees
    with the same number of nodes
    Parameters
    ----------
    T1: HyperbolicDelaunay
        First tree
    T2: HyperbolicDelaunay
        Second tree
    Returns
    -------
    float: the distance
    """
    return T1.get_robinson_foulds_unweighted(T2)


def get_rotation_sequence_polyheuristic(BT1, BT2, verbose=False, heuristic = alpha_sequence_heuristic):
    """
    Compute a sequence that realizes the optimal rotation dist
    between two trees, using some heuristic based on 
    Parameters
    ----------
    BT1: BinaryTree
        First tree
    BT2: BinaryTree
        Second tree
    verbose: boolean
        Whether to print out stats
    heuristic: function: (HyperbolicDelaunay, HyperbolicDelaunay) -> float
        An edge flip heuristic between two triangulations
    Returns
    -------
    sequence: list of lists
        List of alpha sequences along the path from start to finish.  
        Length of the list is one more than the rotation distance
    """
    import json
    prev = {} # Stores the tree that came directly
    # before this tree in a shortest path
    dist = {} # Stores the distance from BT1 to this tree

    T1 = BT1.to_triangulation()
    T2 = BT2.to_triangulation()
    alpha1 = T1.get_horocycle_arclens()
    alpha1_str = "{}".format(alpha1.tolist())
    alpha2 = T2.get_horocycle_arclens()
    alpha2_str = "{}".format(alpha2.tolist())

    # Heap will contain 
    # (heuristic, true distance, alpha string, prev alpha string)
    h = []  
    # Setup initial neighbors from first tree
    for neighb in T1.get_alpha_sequence_neighbors():
        s = "{}".format(neighb.tolist())
        T = HyperbolicDelaunay()
        T.init_from_alphasequence_unweighted(neighb.tolist())
        heappush(h, (1+heuristic(T, T2), 1, s, alpha1_str, T))
    num_expanded = 1
    # Perform breadth-first search
    while len(h) > 0:
        (_, d, s, p, T) = heappop(h)
        if not s in dist:
            num_expanded += 1
            dist[s] = d
            prev[s] = p
            if s == alpha2_str:
                break
            # Add on neighbors
            for neighb in T.get_alpha_sequence_neighbors():
                ns = "{}".format(neighb.tolist())
                TNext = HyperbolicDelaunay()
                TNext.init_from_alphasequence_unweighted(neighb.tolist())
                heappush(h, (d+1+heuristic(TNext, T2), d+1, ns, s, TNext))
    # Now backtrace to find sequence from BT1 to BT2
    sequence = [alpha2_str]
    while prev[sequence[-1]] != alpha1_str:
        sequence.append(prev[sequence[-1]])
    sequence.append(alpha1_str)
    sequence = sequence[::-1]
    sequence = [json.loads(s) for s in sequence]
    if verbose:
        print("dist(BT2) = ", dist[alpha2_str])
        print("len(sequence) = ", len(sequence))
        print("{} Expanded".format(num_expanded))
        print("heuristic from start to finish: ", heuristic(T1, T2))
    return sequence


def get_meet_join(w1, w2):
    """
    Get the join and meet of two weight sequences
    corresponding to two trees
    Parameters
    ----------
    w1: ndarray(N)
        Weight sequence for the first tree with N internal
        nodes
    w2: ndarray(N)
        Weight sequence for the second tree with N internal
        nodes
    Returns
    -------
    meet: ndarray(N)
        Weight sequence for the meet,
    join: ndarray(N)
        Weight sequence for the join
    """
    meet = np.minimum(w1, w2)
    join = np.maximum(w1, w2)
    for i in range(len(join)):
        if join[i] != 1 and join[i] != i+1:
            j = np.min(np.array([k-join[k-1]+1 for k in range(i+1-join[i]+1, i+2)]))
            join[i] = (i + 1) - j
    return meet, join

def render_tree(w, N):
    T = weightsequence_to_binarytree(w)
    MT = T.to_merge_tree()
    MT.render(np.array([0, 0]))
    plt.ylim([-N-1, 1])
    plt.xlim([-1, 2*N+1])
    plt.axis('off')
    return T, MT

def make_all_tree_figures(N):
    """
    Create figures of all possible binary trees of 
    a certain size
    """
    ws = enumerate_weightsequences(N)
    plt.figure(figsize=(10, 5))
    for i, w in enumerate(ws):
        plt.clf()
        plt.subplot(1, 2, 1)
        T, MT = render_tree(w, N)
        plt.title("{}\n{} of {}".format(w, i+1, len(ws)))
        plt.subplot(1, 2, 2)
        HD = HyperbolicDelaunay()
        HD.init_from_mergetree(MT)
        HD.render(draw_vars=False)
        plt.savefig("{}.png".format(i))

def test_meet_join(N):
    np.random.seed(2)
    ws = enumerate_weightsequences(N)
    idx = np.random.permutation(len(ws))
    w1 = ws[idx[0]]
    w2 = ws[idx[1]]
    meet, join = get_meet_join(w1, w2)
    print(meet)
    print(join)
    plt.subplot(221)
    render_tree(w1, N)
    plt.title("T1: {}".format(w1))
    plt.subplot(222)
    render_tree(w2, N)
    plt.title("T2: {}".format(w2))
    plt.subplot(223)
    render_tree(meet, N)
    plt.title("Meet: {}".format(meet))
    """
    plt.subplot(224)
    render_tree(join, N)
    plt.title("Join: {}".format(join))
    """
    plt.show()

def test_rotation_distance_hyperbolic(N):
    ## Step 1: Select two random binary trees and compute
    ## their rotation distance
    np.random.seed(3)
    ws = enumerate_weightsequences(N)
    idx = np.random.permutation(len(ws))
    w1 = ws[idx[0]]
    T1 = weightsequence_to_binarytree(w1)
    w2 = ws[idx[1]]
    T2 = weightsequence_to_binarytree(w2)
    sequence = get_rotation_sequence(T1, T2)
    print("Dist = ", len(sequence)-1)
    plt.figure(figsize=(5, 5))
    for i, w in enumerate(sequence):
        plt.clf()
        render_tree(w, N)
        plt.title("Dist {}: {}".format(i, w))
        plt.savefig("Rot{}.png".format(i))

    ## Step 2: Solve for the hyperbolic structures
    ## that realize these two binary trees
    from FullPipeline import mergetree_to_hypmergetree, plot_solution_grid
    plt.figure(figsize=(18, 12))
    constraints=[('z', 0, 0), ('r', -1, 1)]
    cmt = T1.to_merge_tree()
    res = mergetree_to_hypmergetree(cmt, constraints)
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5.5], perturb=0)
    plt.savefig("T1.png", bbox_inches='tight')
    constraints=[('z', 0, 0), ('r', -1, 1)]

    plt.clf()
    cmt = T2.to_merge_tree()
    res = mergetree_to_hypmergetree(cmt, constraints)
    plot_solution_grid(cmt, res['hd'], res['hmt'], constraints, xlims=[-0.5, 4.5], ylims_voronoi=[0, 6.5], ylims_masses=[0, 5.5], perturb=0)
    plt.savefig("T2.png", bbox_inches='tight')


def test_alpha_sequences():
    """
    Verify that the alpha sequences are 1 to 1
    with possible triangulations
    """
    ws = enumerate_weightsequences(10)
    sequences = {}
    for w in ws:
        BT = weightsequence_to_binarytree(w)
        T = BT.to_triangulation()
        alphas = T.get_horocycle_arclens()
        alphas = np.array(alphas, dtype=int)
        s = "{}".format(alphas)
        if not s in sequences:
            sequences[s] = 1
        else:
            sequences[s] += 1
    print(len(ws))
    print(len(sequences))

def test_alpha_sequence_neighbors():
    """
    Plot the neighbors of a particular alpha sequence
    by flipping edges
    """
    #w = [1, 1, 1, 2, 3, 4, 1, 1]
    w = [1, 2, 3, 4, 1, 1, 1, 4]
    BT = weightsequence_to_binarytree(w)
    T = BT.to_triangulation()
    alphas = T.get_horocycle_arclens()

    N = len(w)
    dim = int(np.ceil(np.sqrt(N)))
    plt.figure(figsize=(dim*4, dim*4))
    plt.subplot(dim, dim, 1)
    T.render(draw_vars=False)
    plt.title("Original: {}".format(np.array(alphas, dtype=int)))

    dTheta = 2*np.pi/(N+2)
    theta0 = np.pi/2 - dTheta/2
    Xs = np.zeros((N+2, 2))
    Xs[:, 0] = np.cos(theta0 + dTheta*np.arange(N+2))
    Xs[:, 1] = np.sin(theta0 + dTheta*np.arange(N+2))
    neighbors = T.get_alpha_sequence_neighbors(alphas)
    for i, alphasi in enumerate(neighbors):
        T2 = HyperbolicDelaunay()
        T2.init_from_alphasequence_unweighted(alphasi)
        alphasi = T2.get_horocycle_arclens()
        plt.subplot(dim, dim, i+2)
        T2.render(draw_vars=False)
        i1, i2, _, _ = T.get_alpha_sequence_diff(i)
        xedge = Xs[[i1+1, i2+1], :]
        plt.plot(xedge[:, 0], xedge[:, 1], c='C2', linestyle='--')
        alphas_plot = np.array([alphasi, alphasi-alphas], dtype=int)
        alphas_plot = "{}".format(alphas_plot)
        alphas_plot = alphas_plot.replace("[", "")
        alphas_plot = alphas_plot.replace("]", "")
        alphas_plot = alphas_plot.lstrip()
        plt.title(alphas_plot)
    plt.savefig("Alpha_Sequence_Neighbors.svg", bbox_inches='tight')


def test_rotation_distance_heuristic(N, seed, do_bfs = True):
    ## Step 1: Select two random binary trees and compute
    ## their rotation distance
    np.random.seed(seed)
    ws = enumerate_weightsequences(N)
    idx = np.random.permutation(len(ws))
    w1 = ws[idx[0]]
    T1 = weightsequence_to_binarytree(w1)
    w2 = ws[idx[1]]
    T2 = weightsequence_to_binarytree(w2)
    if do_bfs:
        print("Doing bfs...")
        sequence = get_rotation_sequence(T1, T2, verbose=True)
    print("Doing A* alpha heuristic")
    sequence2 = get_rotation_sequence_polyheuristic(T1, T2, verbose=True)
    print("Doing A* Robinson-Foulds heuristic")
    sequence3 = get_rotation_sequence_polyheuristic(T1, T2, verbose=True, heuristic=robinson_folds_heuristic)
    if do_bfs:
        while len(sequence) < len(sequence2):
            sequence.append(sequence[-1])

    if do_bfs:
        plt.figure(figsize=(15, 5))
    else:
        plt.figure(figsize=(10, 5))
        sequence = sequence2
    for i, (w, alphas1, alphas2) in enumerate(zip(sequence, sequence2, sequence3)):
        plt.clf()
        if do_bfs:
            plt.subplot(133)
            render_tree(w, N)
            plt.title("Dist {}: {}".format(i, w))
            plt.subplot(131)
        else:
            plt.subplot(121)
        T = HyperbolicDelaunay()
        T.init_from_alphasequence_unweighted(alphas1)
        T.render(draw_vars=False)
        plt.title("Alpha {}".format(np.array(alphas1, dtype=int)))
        if do_bfs:
            plt.subplot(132)
        else:
            plt.subplot(122)
        T = HyperbolicDelaunay()
        T.init_from_alphasequence_unweighted(alphas2)
        T.render(draw_vars=False)
        plt.title("RB-Flds {}".format(np.array(alphas2, dtype=int)))
        plt.savefig("Rot{}.png".format(i))

def make_counts_dict(I):
    """
    Deal with multiset properties of persistence diagram
    by creating a dictionary of counts
    Parameters
    ----------
    I: ndarray(N, 2)
        Persistence diagram
    """
    counts = {}
    for i in range(I.shape[0]):
        s = "{}, {}".format(I[i, 0], I[i, 1])
        if s in counts:
            counts[s] += 1
        else:
            counts[s] = 1
    return counts

def test_flexion(N):
    from persim import plot_diagrams
    #BT1 = weightsequence_to_binarytree(np.arange(N)+1)
    #BT2 = weightsequence_to_binarytree(np.ones(N))

    np.random.seed(5)
    ws = enumerate_weightsequences(N)
    idx = np.random.permutation(len(ws))
    w1 = ws[idx[0]]
    BT1 = weightsequence_to_binarytree(w1)
    neighbs = BT1.get_rotation_neighbors()
    w2 = neighbs[np.random.randint(len(neighbs))]['w']
    BT2 = weightsequence_to_binarytree(w2)
    neighbs = BT2.get_rotation_neighbors()
    w2 = neighbs[np.random.randint(len(neighbs))]['w']
    BT2 = weightsequence_to_binarytree(w2)

    T1 = BT1.to_triangulation()
    T2 = BT2.to_triangulation()

    sequence = get_rotation_sequence_polyheuristic(BT1, BT2, verbose=True)
    print("True rotation distance: ", len(sequence)-1)

    dTheta = 2*np.pi/(N+2)
    theta0 = np.pi/2 - dTheta/2
    Xs = np.zeros((N+3, 2))
    Xs[:, 0] = np.cos(theta0 + dTheta*np.arange(N+3))
    Xs[:, 1] = np.sin(theta0 + dTheta*np.arange(N+3))

    plt.figure(figsize=(15, 10))
    for i in range(N+2):
        plt.clf()
        MT = T1.get_merge_tree(i)
        plt.subplot(231)
        T1.render(draw_vars=False)
        plt.plot(Xs[[i, i+1], 0], Xs[[i, i+1], 1], c='C0', linewidth=5)
        plt.subplot(232)
        MT.render(np.array([0, 0]))
        I1 = MT.get_persistence_diagram()

        MT = T2.get_merge_tree(i)
        I2 = MT.get_persistence_diagram()
        plt.subplot(234)
        T2.render(draw_vars=False)
        plt.plot(Xs[[i, i+1], 0], Xs[[i, i+1], 1], c='C0', linewidth=5)
        plt.subplot(235)
        MT.render(np.array([0, 0]))

        # Figure out which dots are not in common, possibly with
        # multiplicity greater than 1
        I1Dots = make_counts_dict(np.array(I1, dtype=int))
        I2Dots = make_counts_dict(np.array(I2, dtype=int))
        diff_text = "Unique Top:\n"
        for s in I1Dots:
            if not s in I2Dots:
                diff_text += s + "\n"
            elif I1Dots[s] > I2Dots[s]:
                for k in range(I1Dots[s] - I2Dots[s]):
                    diff_text += s + "\n"
        diff_text += "\nUnique Bottom:\n"
        for s in I2Dots:
            if not s in I1Dots:
                diff_text += s + "\n"
            elif I2Dots[s] > I1Dots[s]:
                for k in range(I2Dots[s] - I1Dots[s]):
                    diff_text += s + "\n"
        plt.subplot(236)
        plt.text(0.1, 0.1, diff_text)
        plt.axis('off')

        plt.subplot(233)
        plot_diagrams(I1)
        plt.scatter(I2[:, 0], I2[:, 1], 100, marker='x')
        plt.savefig("{}.png".format(i))

if __name__ == '__main__':
    make_all_tree_figures(3)
    #test_meet_join(7)
    #test_rotation_distance_hyperbolic(5)
    #test_alpha_sequences()
    #test_alpha_sequence_neighbors()
    #test_rotation_distance_heuristic(12, 0, True)
    #test_flexion(10)
