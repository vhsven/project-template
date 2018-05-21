import numpy as np
import pandas as pd

def import_weka_results(filename):
    wdf = pd.read_csv(filename, encoding="utf-8", parse_dates=["Date_time"])
    wdf = wdf[["Key_Dataset", "Key_Scheme_options", "Percent_correct", "Weighted_avg_F_measure", "Elapsed_Time_training", "Elapsed_Time_testing", "measureTreeSize", "measureNumLeaves"]]
    wdf.rename(columns={
        "Key_Dataset": "dataset",
        "Key_Scheme_options": "params",
        "Elapsed_Time_training": "fit_time",
        "Elapsed_Time_testing": "score_time",
        "Percent_correct": "accuracy",
        "Weighted_avg_F_measure": "f1",
        "measureNumLeaves": "n_leaves",
        "measureTreeSize": "n_nodes",
    }, inplace=True)
    wdf.accuracy = wdf.accuracy / 100.0
    wdf.fit_time = wdf.fit_time * 1000
    wdf.score_time = wdf.score_time * 1000
    gwdf = wdf.groupby(["dataset", "params"]).agg(["mean", "std"])
    gwdf.columns = ["_".join(reversed(t)) for t in gwdf.columns]
    gwdf["tool"] = "weka"
    gwdf["p_prune"] = [_get_prune_method(v) for v in gwdf.index.get_level_values(1)]
    gwdf["p_rep_val_percentage"] = np.reciprocal(pd.to_numeric(gwdf.index.get_level_values(1).str.extract(r"-N (\d+)", expand=False)).values)
    gwdf["p_ebp_confidence"] = pd.to_numeric(gwdf.index.get_level_values(1).str.extract(r"-C ([\d\.]+)", expand=False)).values
    return gwdf

def _get_prune_method(params):
    if "-C" in params: return 'ebp'
    if "-R" in params: return 'rep'
    if "-U" in params: return 'none'
    return 'unknown'
