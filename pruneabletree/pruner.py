"""Module containing `Pruner` class.
"""
import numpy as np

INVALID_NODEID = -1
INVALID_VALUE = -2

class Pruner(object):
    """Base class for decision tree pruners.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def num_actual_nodes(tree):
        """Returns the actual number of nodes in the given tree after pruning.
        
        This differs from the `tree.node_count` property, which contains the 
        number of nodes before pruning. Updating this value would break other 
        features since the underlying data structures still have their original 
        sizes.

        Parameters
        ----------
        tree : Tree object
            The underlying tree object of a DecisionTreeClassifier (e.g. `clf.tree_`).
        """
        return (tree.n_node_samples > 0).sum()

    @staticmethod
    def num_leaves(tree):
        """Returns the number of leaves.

        Parameters
        ----------
        tree : Tree object
            The underlying tree object of a DecisionTreeClassifier (e.g. `clf.tree_`).
        """
        return ((tree.n_node_samples > 0) & (tree.feature == INVALID_VALUE)).sum()

    def num_instances(self, node_id, y_idx=None):
        """Returns the number of instances in a given node.

        If a class index is given, only returns the number of instances belonging to that class.

        Parameters
        ----------
        node_id : int
            The ID of the node in the tree
        
        y_idx : int, optional
            The index of the class the instances should belong to. 
            If provided, other classes are ignored.
        """
        if y_idx is None:
            return sum(self.tree.value[node_id][0])
        return self.tree.value[node_id][0][y_idx]

    def is_leaf(self, node_id):
        """Returns True if the given node is a leaf.

        Parameters
        ----------
        node_id : int
            The ID of the node in the tree
        """
        return self.tree.children_left[node_id] == INVALID_NODEID

    def leaf_prediction(self, node_id): #returns y_idx
        """Returns the class index of that the node with the given ID would predict.
        
        Parameters
        ----------
        node_id : int
            The ID of the node in the tree
        """
        values = self.tree.value[node_id]
        return np.argmax(values)

    def to_leaf(self, node_id, depths):
        """Convert the node with the given ID to a leaf, pruning away its children.

        Parameters
        ----------
        node_id : int
            The ID of the node in the tree

        depths : array of shape = [tree.node_count]
            Data structure that keeps track of the depth of each node.
        """
        # print("Converting node {} to leaf.".format(node_id))
        self._kill_node(self.tree.children_left[node_id], depths)
        self._kill_node(self.tree.children_right[node_id], depths)
        self._unlink_children(node_id)

        self.tree.feature[node_id] = INVALID_VALUE
        self.tree.impurity[node_id] = 0
        self.tree.threshold[node_id] = INVALID_VALUE

        #alternative: add to_leaf function in Cython
        #alternative: build new tree from scratch (add_node() exists)
        #alternative: get/edit/set state (every time or at end?)
        #alternative: compress state at end

    def _kill_node(self, node_id, depths):
        depths[node_id] = -1

        if not self.is_leaf(node_id):
            self.to_leaf(node_id, depths)

        self.tree.n_node_samples[node_id] = 0
        self.tree.value[node_id].fill(0)
        self.tree.weighted_n_node_samples[node_id] = 0 #TODO test with weights

    def _unlink_children(self, node_id):
        self.tree.children_left[node_id] = INVALID_NODEID
        self.tree.children_right[node_id] = INVALID_NODEID
