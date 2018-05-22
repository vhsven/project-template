from sklearn.utils.testing import assert_equal, ignore_warnings

from pruneabletree import CsvImporter

def test_transformer_1():
    importer = CsvImporter(na_values=['?'], class_index=-1)
    with ignore_warnings(category=(UserWarning)):
        X = importer.fit_transform("pruneabletree/tests/datasets/dataset_55_hepatitis.csv")
        y = importer.y

        print(X)
        print(y)

        assert_equal(2, len(X.shape))
        assert_equal(1, len(y.shape))
        assert_equal(X.shape[0], y.shape[0])

def test_transformer_2():
    importer = CsvImporter(na_values=['?'], class_index=-1)
    with ignore_warnings(category=(UserWarning)):
        X = importer.fit_transform("pruneabletree/tests/datasets/dataset_61_iris.csv")
        y = importer.y

        print(X)
        print(y)

        assert_equal(2, len(X.shape))
        assert_equal(1, len(y.shape))
        assert_equal(X.shape[0], y.shape[0])

def test_transformer_3():
    importer = CsvImporter(na_values=['?'], class_index=-1)
    with ignore_warnings(category=(UserWarning)):
        X = importer.fit_transform("pruneabletree/tests/datasets/dataset_56_vote.csv")
        y = importer.y

        print(X)
        print(y)

        assert_equal(2, len(X.shape))
        assert_equal(1, len(y.shape))
        assert_equal(X.shape[0], y.shape[0])
