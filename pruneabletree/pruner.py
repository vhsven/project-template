"""
TODO module doc
"""
import numpy as np

INVALID_NODEID = -1
INVALID_VALUE = -2

class Pruner:
    """
    TODO class doc
    """
    
    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def num_actual_nodes(tree):
        return (tree.n_node_samples > 0).sum()

    @staticmethod
    def num_leaves(tree):
        return ((tree.n_node_samples > 0) & (tree.feature == INVALID_VALUE)).sum()

    def num_instances(self, node_id, y_idx=None):
        if y_idx is None:
            return sum(self.tree.value[node_id][0])
        return self.tree.value[node_id][0][y_idx]

    def is_leaf(self, node_id):
        return self.tree.children_left[node_id] == INVALID_NODEID

    def leaf_prediction(self, node_id): #returns y_idx
        values = self.tree.value[node_id]
        return np.argmax(values)

    def to_leaf(self, node_id, depths):
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
