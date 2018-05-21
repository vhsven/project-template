"""
=======================================
Fit a prunable tree on the iris dataset
=======================================

Fit a prunable tree on the iris dataset and compare the results of various pruning strategies.
"""
print(__doc__)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pruneabletree import PruneableDecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_none = PruneableDecisionTreeClassifier(prune=None).fit(X_train, y_train)
clf_rep = PruneableDecisionTreeClassifier(prune='rep', rep_val_percentage=.1).fit(X_train, y_train)
clf_ebp = PruneableDecisionTreeClassifier(prune='ebp', ebp_confidence=1e-4).fit(X_train, y_train)

print("Number of nodes remaining with pruning strategies 'none', 'rep' and 'ebp':")
print(clf_none.n_actual_nodes)
print(clf_rep.n_actual_nodes)
print(clf_ebp.n_actual_nodes)

print("Real vs predicted classes of the three classifiers:")
y_none = clf_none.predict(X_test)
y_rep = clf_rep.predict(X_test)
y_ebp = clf_ebp.predict(X_test)
print(y_test)
print(y_none)
print(y_rep)
print(y_ebp)

print("Accuracy scores of the three classifiers:")
print(accuracy_score(y_test, y_none))
print(accuracy_score(y_test, y_rep))
print(accuracy_score(y_test, y_ebp))
