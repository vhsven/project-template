import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.utils.testing import ignore_warnings

from pruneabletree.tests.weka_tools import import_weka_results
from pruneabletree import PruneableDecisionTreeClassifier, CsvImporter

def main(n_repeats=10):
    df_scikit = run_all_scikit_benchmarks(n_repeats)
    df_scikit.to_csv("pruneabletree/tests/results/pruning/scikit.csv")
    df_weka = import_weka_results("pruneabletree/tests/results/pruning/weka_nomissings.csv")
    df_all = pd.concat([df_scikit, df_weka], sort=True)
    df_all.to_csv("pruneabletree/tests/results/pruning/all.csv")

def run_all_scikit_benchmarks(n_repeats=10):
    random_state = 42
    dfs = []
    datasets = {
        "activity": "pruneabletree/tests/datasets/activity.csv",
        "diabetes": "pruneabletree/tests/datasets/dataset_37_diabetes.csv",
        "ionosphere": "pruneabletree/tests/datasets/dataset_59_ionosphere.csv",
        "iris": "pruneabletree/tests/datasets/dataset_61_iris.csv",
        "wine": "pruneabletree/tests/datasets/dataset_187_wine.csv",
        "wdbc": "pruneabletree/tests/datasets/dataset_1510_wdbc.csv",
        "letter": "pruneabletree/tests/datasets/dataset_6_letter.csv",
        "german_credit": "pruneabletree/tests/datasets/dataset_31_credit-g.csv",
        "tic-tac-toe": "pruneabletree/tests/datasets/dataset_50_tic-tac-toe.csv",
        "heart-statlog": "pruneabletree/tests/datasets/dataset_53_heart-statlog.csv",
        "houses": "pruneabletree/tests/datasets/dataset_823_houses.csv",
        "hepatitis": "pruneabletree/tests/datasets/dataset_55_hepatitis.csv",
        "vote": "pruneabletree/tests/datasets/dataset_56_vote.csv",
        "monks-problems-2": "pruneabletree/tests/datasets/dataset_334_monks.csv"
    }
    for name, filename in datasets.items():
        print(name)
        X, y = CsvImporter(na_values=['?']).fit_transform_both(filename)
        df = run_scikit_benchmark(X, y, n_repeats=n_repeats, dataset=name, random_state=random_state)
        dfs.append(df)
    return pd.concat(dfs)

def run_scikit_benchmark(X, y, n_repeats, dataset, random_state):
    def n_nodes_scorer(clf, X, y):
        return clf.n_actual_nodes

    def n_leaves_scorer(clf, X, y):
        return clf.n_leaves

    n_classes = len(np.unique(y))
    param_grid = [
        {
            "prune": [None]
        },
        {
            "prune": [None],
            "min_samples_leaf": [0.5 / n_classes]
        },
        {
            "prune": ['rep'],
            "rep_val_percentage": [0.1, 0.2, 0.5]
        },
        {
            "prune": ['ebp'],
            "ebp_confidence": [1e-5, 1e-3, 1e-1]
        }
    ]
    scorers = {
        "n_nodes": n_nodes_scorer,
        "n_leaves": n_leaves_scorer,
        "accuracy": "accuracy",
        "f1": "f1_weighted"
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=random_state)
    clf = PruneableDecisionTreeClassifier(criterion="entropy", random_state=random_state)
    gsclf = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, refit=False, cv=cv, return_train_score=False)
    with ignore_warnings(category=(UndefinedMetricWarning)):
        gsclf.fit(X, y)
    return _store_results(gsclf.cv_results_, dataset)

def _store_results(cv_results, dataset):
    df = pd.DataFrame(cv_results)
    df.drop(columns=df.columns[df.columns.str.startswith("rank")], inplace=True)
    df.drop(columns=df.columns[df.columns.str.startswith("mean_test")], inplace=True)
    df.drop(columns=df.columns[df.columns.str.startswith("std_test")], inplace=True)
    df.mean_fit_time = df.mean_fit_time * 1000
    df.mean_score_time = df.mean_score_time * 1000
    df.columns = df.columns.str.replace("test_", "")
    df.columns = df.columns.str.replace("param_", "p_")
    df.columns = df.columns.str.replace("mean_(.*_time)", r"\1_ms")
    df["p_prune"] = df.params.apply(_get_prune_method)
    df["config"] = df.apply(_get_prune_method2, axis=1)
    df["tool"] = "scikit"
    df["dataset"] = dataset
    df.set_index("config", inplace=True)
    selection = df[["fit_time_ms", "score_time_ms", "p_prune", "p_min_samples_leaf", "p_rep_val_percentage", "p_ebp_confidence", "dataset", "tool"]]

    x = df.filter(regex="split").unstack().reset_index()
    x["param"] = x.level_0.str.replace("split\d+_(.*)", r"\1")
    x["run"] = x.level_0.str.replace("split(\d+)_.*", r"\1")
    x = x.drop(columns='level_0').set_index(["config", "run", "param"]).unstack().reset_index(col_level=1)
    x.columns = x.columns.get_level_values(1).values
    x.drop(columns="run", inplace=True)
    x.set_index('config', inplace=True)
    result = pd.merge(x, selection, left_index=True, right_index=True)
    return result

def _get_prune_method(params):
    if "min_samples_leaf" in params: 
        return 'stop'
    if "rep_val_percentage" in params: 
        return 'rep'
    if "ebp_confidence" in params: 
        return 'ebp'
    # if "'prune': None" in params: 
    #     return 'none'
    return 'none'

def _get_prune_method2(row):
    if "min_samples_leaf" in row.params: 
        return 'early_stop'
    if "ebp_confidence" in row.params: 
        return 'ebp_{:.5f}'.format(row.p_ebp_confidence)
    if "rep_val_percentage" in row.params: 
        return 'rep_{}'.format(row.p_rep_val_percentage)
    # if "'prune': None" in row.params: 
    #     return 'none'
    return 'none'

if __name__ == "__main__":
    main(10)
