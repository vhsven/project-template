"""
============================================================
Read and pre-process a CSV file for usage in a decision tree
============================================================

The example uses the :class:`pruneabletree.csv_importer.CsvImporter` to read 
a CSV file containing a data set with missing values and categorical attributes.
Observations with missing values will be discarded and categorical attributes 
will be encoded. This makes the data ready to be used by a decision tree 
classifier.
"""
print(__doc__)


from sklearn.model_selection import cross_val_score
from pruneabletree import CsvImporter, PruneableDecisionTreeClassifier

importer = CsvImporter()
clf = PruneableDecisionTreeClassifier(prune='ebp', ebp_confidence=0.001, random_state=0)

filepath = "pruneabletree/tests/datasets/dataset_56_vote.csv"
X, y = importer.fit_transform_both(filepath)
# Or:
# X = importer.fit_transform(filepath)
# y = importer.y
print(cross_val_score(clf, X, y, cv=10))
