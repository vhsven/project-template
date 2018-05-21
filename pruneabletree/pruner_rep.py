"""
TODO module doc
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .pruner import Pruner

class ReducedErrorPruner(Pruner):
    """
    TODO class doc
    """
    
    def __init__(self, tree):
        super(ReducedErrorPruner, self).__init__(tree)
        self.classify_result = np.zeros((self.tree.n_classes[0], self.tree.node_count))

    def prune(self, X_val, y_val):
        encoder = LabelEncoder()
        y_idxs = encoder.fit_transform(y_val) #TODO doublecheck
        self._classify_validation_set(X_val, y_idxs)
        self._reduced_error_pruning()

    def _classify_validation_set(self, X_val, y_idxs):
        k = self.tree.n_classes[0] #TODO possible for multi output?
        for y_idx in np.arange(k):
            X_ofclass = X_val[y_idxs == y_idx].astype('float32')
            self.classify_result[y_idx, :] = self.tree.decision_path(X_ofclass).sum(axis=0)

    def _reduced_error_pruning(self):
        depths = np.zeros(self.tree.node_count)
        self._prune_recursive(0, depths, 0)
        self.tree.max_depth = depths.max()
        #self.tree.node_count has to stay constant because array does not shrink

    # returns output classification error after pruning T
    def _prune_recursive(self, node_id, depths, depth):
        depths[node_id] = depth
        if self.is_leaf(node_id):
            return self._calculate_leaf_error_rep(node_id)

        error_asleaf = self._calculate_leaf_error_rep(node_id)
        error_assubtree = \
            self._prune_recursive(self.tree.children_left[node_id], depths, depth+1) + \
            self._prune_recursive(self.tree.children_right[node_id], depths, depth+1)

        if error_assubtree < error_asleaf:
            return error_assubtree

        self.to_leaf(node_id, depths)
        return error_asleaf

    def _calculate_leaf_error_rep(self, node_id):
        y_idx_pred = self.leaf_prediction(node_id)
        total_val_visits = self.classify_result[:, node_id].sum()
        correct_val_visits = self.classify_result[y_idx_pred, node_id]
        return total_val_visits - correct_val_visits
