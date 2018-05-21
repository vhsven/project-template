import graphviz

from sklearn import tree as tree_pkg

from mysklearn.tree import PruneableDecisionTreeClassifier

def plot_tree(clf, name):
    dot_data = tree_pkg.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(name)

def compare_prepost_pruning(X, y, random_state):
    clf = PruneableDecisionTreeClassifier(
        random_state=random_state,
        prune=None).fit(X, y)
    print(clf.tree_.node_count)
    print(clf.tree_.n_node_samples)
    print(clf.tree_.weighted_n_node_samples)
    print(clf.tree_.value)
    print(clf.tree_.__getstate__())
    plot_tree(clf, "none")
    print(clf.predict(X))

    clf = PruneableDecisionTreeClassifier(
        random_state=random_state,
        prune='rep',
        rep_val_percentage=0.1).fit(X, y)
    plot_tree(clf, "rep")
    print(clf.predict(X))

    clf = PruneableDecisionTreeClassifier(
        random_state=random_state,
        prune='ebp',
        ebp_confidence=0.25).fit(X, y)
    plot_tree(clf, "ebp")
    print(clf.predict(X))