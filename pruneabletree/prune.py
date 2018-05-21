"""Module containing PruneableDecisionTreeClassifier class.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from .pruner import Pruner
from .pruner_rep import ReducedErrorPruner
from .pruner_ebp import ErrorBasedPruner

class PruneableDecisionTreeClassifier(DecisionTreeClassifier):
    """A pruneable decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float,
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19 and will be removed in 0.21.
           Use ``min_impurity_decrease`` instead.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    prune : string, optional (default=None)
        Determines the pruning strategy. Options are None for no pruning, 
        'rep' for Reduced Error Pruning and 'ebp' for Error Based Pruning.

    rep_val_percentage : float (default=0.1)
        Determines which percentage of the training set can be used as 
        validation set for Reduced Error Pruning. Only valid if `prune='rep'`.

    ebp_confidence : float (default=0.25)
        The confidence value that determines the upper bound on the training error.
        It must be in the (0, 0.5] interval. Only valid if `prune='ebp'`.

    Attributes
    ----------
    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_classes_ : int or list
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    Notes
    -----
    The original `DecisionTreeClassifier` only provided some simple early
    stopping criteria to limit the size of the induced tree. This 
    `PruneableDecisionTreeClassifier` additionally includes two pruning 
    strategies: Reduced Error Pruning (REP) and Error Based Pruning (EBP). 
    If `prune=None` this class acts like a regular `DecisionTreeClassifier`.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    References
    ----------

    .. [1] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [2] J Ross Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.

    .. [3] J. Ross Quinlan. Simplifying decision trees. International journal of
           man-machine studies, 27(3):221-234, 1987.

    .. [4] Tapio Elomaa and Matti Kaariainen. An analysis of reduced error
           pruning. Journal of Artificial Intelligence Research, 15:163–187, 2001.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from pruneabletree import PruneableDecisionTreeClassifier
    >>> clf = PruneableDecisionTreeClassifier(random_state=0, prune='rep')
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([1.         0.93333333 1.         0.93333333 0.93333333 0.86666667
           0.86666667 1.         1.         1.        ])
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
        """Build a decision tree classifier from the training set (X, y).

        The tree is pruned afterwards using the given pruning strategy.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
        """
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
        """Returns the actual number of nodes after pruning.
        
        This differs from the `tree_.node_count` property, which contains the 
        number of nodes before pruning. Updating this value would break other 
        features since the underlying data structures still have their original 
        sizes.
        """
        return Pruner.num_actual_nodes(self.tree_)

    @property
    def n_leaves(self):
        """Returns the number of leaves.
        """
        return Pruner.num_leaves(self.tree_)
