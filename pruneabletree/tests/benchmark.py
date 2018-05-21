import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pruneabletree.tests.weka_tools import import_weka_results
from pruneabletree import PruneableDecisionTreeClassifier

#TODO update weka test config
def main(n_repeats=100):
    df_scikit = run_all_scikit_benchmarks(n_repeats)
    df_scikit.to_csv("pruneabletree/tests/results/scikit.csv")

    df_weka = import_weka_results("pruneabletree/tests/results/weka.csv")
    df_all = pd.concat([df_scikit, df_weka], sort=True)
    df_all.columns = [c + "_ms" if "time" in c else c for c in df_all.columns]
    df_all.to_csv("pruneabletree/tests/results/all.csv")

def run_all_scikit_benchmarks(n_repeats=100):
    random_state = 42
    dfs = []
    datasets = {
        "activity": "pruneabletree/tests/datasets/activity.csv",
        "diabetes": "pruneabletree/tests/datasets/dataset_37_diabetes.csv",
        #"soybean":  "pruneabletree/tests/datasets/dataset_42_soybean.csv",
        #"hypothyroid":  "pruneabletree/tests/datasets/dataset_57_hypothyroid.csv",
        "ionosphere": "pruneabletree/tests/datasets/dataset_59_ionosphere.csv",
        "iris": "pruneabletree/tests/datasets/dataset_61_iris.csv",
        #"adult":  "pruneabletree/tests/datasets/dataset_183_adult.csv",
        "wine": "pruneabletree/tests/datasets/dataset_191_wine.csv",
        "wdbc": "pruneabletree/tests/datasets/dataset_1510_wdbc.csv"
    }
    for name, filename in datasets.items():
        print(name)
        X, y = _csv_to_xy(filename)
        df = run_scikit_benchmark(X, y, n_repeats=n_repeats, dataset=name, random_state=random_state)
        dfs.append(df)
    return pd.concat(dfs)

def run_scikit_benchmark(X, y, n_repeats, dataset, random_state):
    def n_nodes_scorer(clf, X, y):
        return -clf.n_actual_nodes

    def n_leaves_scorer(clf, X, y):
        return -clf.n_leaves

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
            "rep_val_percentage": [1/10, 1/5, 1/3, 1/2]
        },
        {
            "prune": ['ebp'],
            "ebp_confidence": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1/20, 1/2]
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
    gsclf = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, refit="f1", cv=cv, return_train_score=False)
    gsclf.fit(X, y)
    clf = gsclf.best_estimator_
    df = pd.DataFrame(gsclf.cv_results_)
    df.drop(columns=df.columns[df.columns.str.startswith("split")], inplace=True)
    df.drop(columns=df.columns[df.columns.str.startswith("rank")], inplace=True)
    df.rename(columns=lambda c: c.replace("test_", "").replace("param_", "p_"), inplace=True)
    df.mean_fit_time = df.mean_fit_time * 1000
    df.mean_score_time = df.mean_score_time * 1000
    df.std_fit_time = df.std_fit_time * 1000
    df.std_score_time = df.std_score_time * 1000
    df.mean_n_leaves = df.mean_n_leaves * -1
    df.mean_n_nodes = df.mean_n_nodes * -1
    df["tool"] = "scikit"
    df["dataset"] = dataset
    df['params'] = df['params'].astype('str') #to use in MultiIndex
    df.set_index(["dataset", "params"], inplace=True)
    df["p_prune"] = [_get_prune_method(v) for v in df.index.get_level_values(1)]
    return df

def _csv_to_xy(filename):
    df = pd.read_csv(filename)

    for c in df.columns[df.dtypes == object]:
        #TODO deal with missings ("?") first
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values
    return X, y

def _get_prune_method(params):
    if "min_samples_leaf" in params: return 'min_samples_leaf'
    if "rep_val_percentage" in params: return 'rep'
    if "ebp_confidence" in params: return 'ebp'
    if "'prune': None" in params: return 'none'
    return 'unknown'

if __name__ == "__main__":
    main(1)
