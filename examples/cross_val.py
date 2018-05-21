"""
============================================
Perform cross validation on a pruneable tree
============================================

Calculate the cross validation score of a
:class:`pruneabletree.prune.PruneableDecisionTreeClassifier`
using the Reduced Error Pruning strategy trained on pairs
of features of the iris dataset.
"""
print(__doc__)

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from pruneabletree import PruneableDecisionTreeClassifier
clf = PruneableDecisionTreeClassifier(random_state=0, prune='rep')
iris = load_iris()
print(cross_val_score(clf, iris.data, iris.target, cv=10))