from sklearn.utils.testing import assert_true

from pruneabletree.tests.weka_tools import import_weka_results

def test_confidence_parser():
    df = import_weka_results("pruneabletree/tests/results/pruning/weka_nomissings.csv")
    print(df)
    unique_c_values = set(df.p_ebp_confidence.dropna().values)
    assert_true(0.1 in unique_c_values)
    assert_true(0.001 in unique_c_values)
    assert_true(0.00001 in unique_c_values)
