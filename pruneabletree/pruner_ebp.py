"""Module containing `ErrorBasedPruner` class.
"""

from math import log, pi, sqrt

import numpy as np

from .pruner import Pruner


class ErrorBasedPruner(Pruner):
    """Pruner for decision trees that uses the Error Based Pruning (EBP) technique [1]_.

    Note that the given tree is modified in place. 
    To keep a copy of the original, clone it first.
    
    Parameters
    ----------
    tree : Tree object
        The underlying tree object of a DecisionTreeClassifier (e.g. `clf.tree_`).

    ebp_confidence : float
        The confidence value that determines the upper bound on the training error.
        It must be in the (0, 0.5] interval.

    See also
    --------
    :class:`pruneabletree.prune.PruneableDecisionTreeClassifier`
    :class:`pruneabletree.pruner_rep.ReducedErrorPruner`
    

    References
    ----------

    .. [1] J Ross Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufmann, 1993.
    """
    def __init__(self, tree, ebp_confidence):
        super().__init__(tree)
        self.ebp_confidence = ebp_confidence

    def prune(self):
        """Prunes the given tree.
        """
        depths = np.zeros(self.tree.node_count)
        self._prune_ebp(0, depths, 0)
        self.tree.max_depth = depths.max() #TODO check if still works

    def _prune_ebp(self, node_id, depths, depth):
        depths[node_id] = depth
        if self.is_leaf(node_id):
            return

        self._prune_ebp(self.tree.children_left[node_id], depths, depth+1)
        self._prune_ebp(self.tree.children_right[node_id], depths, depth+1)

        error_asleaf = self._calculate_leaf_error_ebp(node_id)
        error_assubtree = self._calculate_tree_error_ebp(node_id)

        if error_asleaf <= error_assubtree + 0.1:
            # print("Pruning node {} because leaf error {} <= subtree error {}".format(node_id, error_asleaf, error_assubtree+0.1))
            self.to_leaf(node_id, depths)
        # else:
            # print("Node {} does not need to be pruned".format(node_id))

    def _calculate_leaf_error_ebp(self, node_id):
        n_instances = self.num_instances(node_id)
        if n_instances == 0:
            return 0
        y_idx_pred = self.leaf_prediction(node_id)
        n_correct = self.num_instances(node_id, y_idx_pred)
        n_incorrect = n_instances - n_correct
        errors = add_errors(n_instances, n_incorrect, self.ebp_confidence)
        return n_incorrect + errors

    def _calculate_tree_error_ebp(self, node_id):
        if self.is_leaf(node_id):
            return self._calculate_leaf_error_ebp(node_id)

        return self._calculate_tree_error_ebp(self.tree.children_left[node_id]) + \
               self._calculate_tree_error_ebp(self.tree.children_right[node_id])



# Translated from weka.classifiers.trees.j48.Stats.addErrs
def add_errors(num_instances, observed_error, confidence):
    """
    Computes estimated extra error for given total number of instances
    and error using normal approximation to binomial distribution
    (and continuity correction).
    """

    if confidence > 0.5:
        raise ValueError("Confidence value too high: {}".format(confidence))

    # Check for extreme cases at the low end because the normal approximation won't work
    if observed_error < 1:
    #   Base case (i.e. observed_error == 0) from Geigy Scientific Tables, 6th edition, page 185
        base = num_instances * (1 - pow(confidence, 1 / num_instances))
        # Use linear interpolation between 0 and 1 like C4.5 does
        return base + observed_error * (add_errors(num_instances, 1, confidence) - base)

    # Use linear interpolation at the high end because of the continuity correction
    if observed_error + 0.5 >= num_instances:
        # Make sure that we never return anything smaller than zero
        return max(num_instances - observed_error, 0)

    # Get z-score corresponding to CF
    z = normal_inverse(1 - confidence)

    # Compute upper limit of confidence interval
    f = (observed_error + 0.5) / num_instances
    r = (f + (z * z) / (2 * num_instances) + \
        z * sqrt((f / num_instances) - \
            (f * f / num_instances) + \
            (z * z / (4 * num_instances * num_instances)))) / \
        (1 + (z * z) / num_instances)

    return (r * num_instances) - observed_error

# Translated from weka.core.Statistics.normalInverse
def normal_inverse(y0):
    """
    Returns the value, <tt>x</tt>, for which the area under the Normal
    (Gaussian) probability density function (integrated from minus infinity to
    <tt>x</tt>) is equal to the argument <tt>y</tt> (assumes mean is zero,
    variance is one).
    <p>
    For small arguments <tt>0 < y < exp(-2)</tt>, the program computes
    <tt>z = sqrt( -2.0 * log(y) )</tt>; then the approximation is
    <tt>x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z)</tt>. There are two rational
    functions P/Q, one for <tt>0 < y < exp(-32)</tt> and the other for
    <tt>y</tt> up to <tt>exp(-2)</tt>. For larger arguments,
    <tt>w = y - 0.5</tt>, and <tt>x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2))</tt>.

    @param y0 the area under the normal pdf
    @return the z-value
    """

    # approximation for 0 <= |y - 0.5| <= 3/8
    P0 = [-5.99633501014107895267E1,
          9.80010754185999661536E1, -5.66762857469070293439E1,
          1.39312609387279679503E1, -1.23916583867381258016E0]
    Q0 = [1.95448858338141759834E0, 4.67627912898881538453E0,
          8.63602421390890590575E1, -2.25462687854119370527E2,
          2.00260212380060660359E2, -8.20372256168333339912E1,
          1.59056225126211695515E1, -1.18331621121330003142E0]

    # Approximation for interval z = sqrt(-2 log y ) between 2 and 8 i.e., y
    # between exp(-2) = .135 and exp(-32) = 1.27e-14.
    P1 = [4.05544892305962419923E0,
          3.15251094599893866154E1, 5.71628192246421288162E1,
          4.40805073893200834700E1, 1.46849561928858024014E1,
          2.18663306850790267539E0, -1.40256079171354495875E-1,
          -3.50424626827848203418E-2, -8.57456785154685413611E-4]
    Q1 = [1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1,
          1.50425385692907503408E1, 2.50464946208309415979E0,
          -1.42182922854787788574E-1, -3.80806407691578277194E-2,
          -9.33259480895457427372E-4]

    # Approximation for interval z = sqrt(-2 log y ) between 8 and 64 i.e., y
    # between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
    P2 = [3.23774891776946035970E0,
          6.91522889068984211695E0, 3.93881025292474443415E0,
          1.33303460815807542389E0, 2.01485389549179081538E-1,
          1.23716634817820021358E-2, 3.01581553508235416007E-4,
          2.65806974686737550832E-6, 6.23974539184983293730E-9]
    Q2 = [6.02427039364742014255E0, 3.67983563856160859403E0,
          1.37702099489081330271E0, 2.16236993594496635890E-1,
          1.34204006088543189037E-2, 3.28014464682127739104E-4,
          2.89247864745380683936E-6, 6.79019408009981274425E-9]

    s2pi = sqrt(2.0 * pi)
    EXP_MIN2 = 0.13533528323661269189 #exp(-2)

    if y0 <= 0.0 or y0 >= 1.0:
        raise ValueError()

    code = 1
    y = y0
    if y > (1.0 - EXP_MIN2):
        y = 1.0 - y
        code = 0

    if y > EXP_MIN2:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8))
        x = x * s2pi
        return x

    x = sqrt(-2.0 * log(y))
    x0 = x - log(x) / x

    z = 1.0 / x
    if x < 8.0:
        x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
    else:
        x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8)
    x = x0 - x1
    if code != 0:
        x = -x
    return x

# Translated from weka.core.Statistics.polevl
def polevl(x, coef, N):
    """
    Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
    Coefficients are stored in reverse order.
    In the interest of speed, there are no checks for out of bounds arithmetic.
    """
    ans = coef[0]
    for i in range(1, N+1):
        ans = ans * x + coef[i]
    return ans

# Translated from weka.core.Statistics.p1evl
def p1evl(x, coef, N):
    """
    Evaluates the given polynomial of degree <tt>N</tt> at <tt>x</tt>.
    Evaluates polynomial when coefficient of N is 1.0. Otherwise same as
    <tt>polevl()</tt>.

    Coefficients are stored in reverse order.

    The function <tt>p1evl()</tt> assumes that <tt>coef[N] = 1.0</tt> and is
    omitted from the array. Its calling arguments are otherwise the same as
    <tt>polevl()</tt>.
    <p>
    """
    ans = x + coef[0]
    for i in range(1, N):
        ans = ans * x + coef[i]
    return ans
