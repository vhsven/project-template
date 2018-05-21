"""
TODO module doc
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from .pruner import Pruner
from .pruner_rep import ReducedErrorPruner
from .pruner_ebp import ErrorBasedPruner

class PruneableDecisionTreeClassifier(DecisionTreeClassifier):
    """
    TODO class doc
    """

    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 max_leaf_nodes=None,
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False,
                 prune=None,
                 rep_val_percentage=0.1,
                 ebp_confidence=0.25):
        super(PruneableDecisionTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight=class_weight,
            presort=presort)
        if prune not in [None, 'rep', 'ebp']:
            raise ValueError("`prune` parameter must be either None, 'rep' or 'ebp'")
        self.prune = prune
        if not 0.0 <= rep_val_percentage <= 1.0:
            raise ValueError("`rep_val_percentage` must be a float in [0, 1]")
        self.rep_val_percentage = rep_val_percentage

        if not 0.0 < ebp_confidence <= 0.5:
            raise ValueError("`ebp_confidence` must be a float in (0, 0.5]")
        self.ebp_confidence = ebp_confidence

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        if not self.prune:
            super(PruneableDecisionTreeClassifier, self).fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
        elif self.prune == 'rep':
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.rep_val_percentage,
                random_state=self.random_state,
                stratify=y)
            super(PruneableDecisionTreeClassifier, self).fit(
                X_train, y_train,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            ReducedErrorPruner(self.tree_).prune(X_val, y_val)
        elif self.prune == 'ebp':
            super(PruneableDecisionTreeClassifier, self).fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            ErrorBasedPruner(self.tree_, self.ebp_confidence).prune()
        else:
            raise ValueError("Unknown pruning method: " + str(self.prune))

        # print("Tree with {} nodes and {} leaves.".format(self.n_actual_nodes, self.n_leaves))
        return self

    @property
    def n_actual_nodes(self):
        return Pruner.num_actual_nodes(self.tree_)

    @property
    def n_leaves(self):
        return Pruner.num_leaves(self.tree_)
