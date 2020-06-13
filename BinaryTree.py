import numpy as np 
import matplotlib.pyplot as plt
from collections import deque
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
        
    def update_inorder_rec(self, node_deque):
        """
        Compute the order of this node according to an 
        inorder traversal
        Parameters
        ----------
        node_deque: deque
            A growing list of nodes in order
        """
        if self.left:
            self.left.update_inorder_rec(node_deque)
        node_deque.append(self)
        if self.right:
            self.right.update_inorder_rec(node_deque)
        
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

    def update_inorder(self):
        """
        Update the order of all nodes using inorder
        Returns
        -------
        nodes_deque: collections.deque
            A deque of BinaryNode objects in inorder
        """
        nodes_deque = deque()
        self.root.update_inorder_rec(nodes_deque)
        for i, n in enumerate(nodes_deque):
            n.order = i
        return nodes_deque

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
        T = MergeTree()
        T.root = root
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
    rows = int(N+1)/2
    plt.figure(figsize=(10, 5*rows))
    for i, w in enumerate(ws):
        plt.clf()
        plt.subplot(rows, 2, 1)
        T, MT = render_tree(w, N)
        plt.title("{}\n{} of {}".format(w, i+1, len(ws)))
        plt.subplot(rows, 2, 2)
        HD = HyperbolicDelaunay()
        HD.init_from_mergetree(MT)
        HD.render()

        ws_rot = T.get_rotation_neighbors()
        for k, wn in enumerate(ws_rot):
            plt.subplot(rows, 2, 3+k)
            render_tree(wn['w'], N)
            plt.title("{} Neighbor {}\n{}".format(wn['dir'], k+1, wn['w']))

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



if __name__ == '__main__':
    make_all_tree_figures(7)
    #test_meet_join(7)