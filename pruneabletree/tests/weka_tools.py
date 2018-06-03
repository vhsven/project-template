import numpy as np
import pandas as pd

def import_weka_results(filename):
    df = pd.read_csv(filename, encoding="utf-8", parse_dates=["Date_time"])
    df = df[["Key_Dataset", "Key_Scheme_options", "Percent_correct", "Weighted_avg_F_measure", "Elapsed_Time_training", "Elapsed_Time_testing", "measureTreeSize", "measureNumLeaves"]]
    df.rename(columns={
        "Key_Dataset": "dataset",
        "Key_Scheme_options": "params",
        "Elapsed_Time_training": "fit_time",
        "Elapsed_Time_testing": "score_time",
        "Percent_correct": "accuracy",
        "Weighted_avg_F_measure": "f1",
        "measureNumLeaves": "n_leaves",
        "measureTreeSize": "n_nodes",
    }, inplace=True)
    df.accuracy = df.accuracy / 100.0
    df.fit_time = df.fit_time * 1000
    df.score_time = df.score_time * 1000
    df["tool"] = "weka"
    df["p_rep_val_percentage"] = np.reciprocal(pd.to_numeric(df.params.str.extract(r"-N (\d+)", expand=False)).values) #np.reciprocal(pd.to_numeric(gdf.index.get_level_values(1).str.extract(r"-N (\d+)", expand=False)).values)
    df["p_ebp_confidence"] = pd.to_numeric(df.params.str.extract(r"-C ([^ ]+)", expand=False)).values
    df["p_prune"] = df.params.apply(_get_prune_method) #[_get_prune_method(v) for v in df.index.get_level_values(1)]
    df["config"] = df.apply(_get_prune_method2, axis=1)
    df.columns = [c + "_ms" if "time" in c else c for c in df.columns]
    df.drop(columns="params", inplace=True)
    df.set_index("config", inplace=True)
    return df

def _get_prune_method(params):
    if "-C" in params: 
        return 'ebp'
    if "-R" in params: 
        return 'rep'
    if "-U" in params: 
        return 'none'
    return 'unknown'

def _get_prune_method2(row):
    if "-C" in row.params: 
        return 'ebp_{:.5f}'.format(row.p_ebp_confidence)
    if "-R" in row.params: 
        return 'rep_{}'.format(row.p_rep_val_percentage)
    if "-U" in row.params: 
        return 'none'
    return 'unknown'
