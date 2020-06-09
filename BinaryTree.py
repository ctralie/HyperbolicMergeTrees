import numpy as np 
import matplotlib.pyplot as plt
from collections import deque
from MergeTree import *

class BinaryNode(object):
    def __init__(self, parent = None):
        self.parent = parent
        self.left = None
        self.right = None
        self.order = 0
        self.size = 1
    
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
        

class BinaryTree(object):
    def __init__(self):
        self.root = BinaryNode()

    def update_inorder(self):
        """
        Update the order of all nodes using inorder
        """
        nodes_deque = deque()
        self.root.update_inorder_rec(nodes_deque)
        for i, n in enumerate(nodes_deque):
            n.order = i

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
    ws = [np.ones(N)]
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
    """
    M = len(w1)
    meet = np.minimum(w1, w2)
    join = np.zeros_like(w1)
    for i in range(M):
        join[i] = max(w1[i], w2[i])
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

def make_all_tree_figures(N):
    """
    Create figures of all possible binary trees of 
    a certain size
    """
    ws = enumerate_weightsequences(N)
    for i, w in enumerate(ws):
        plt.clf()
        render_tree(w, N)
        plt.title("{} of {}".format(i+1, len(ws)))
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
    #make_all_tree_figures(7)
    test_meet_join(7)