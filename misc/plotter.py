import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from matplotlib.pyplot import Axes

df = pd.read_csv("pruneabletree/tests/results/all.csv")
filtered = df[(df["dataset"] != "soybean") & (df["dataset"] != "hypothyroid")]
# grouped = filtered.groupby(["dataset", "tool", "prune_method"])
# result = grouped.mean().unstack() #"mean_accuracy", 
# err = result[["std_f1"]]
# ax = result["mean_f1"].plot(kind="bar", figsize=(14, 5), ylim=(0, 1.1), yerr=err, rot=30, width=0.75)
# plt.title("Mean F1 scores")
# plt.legend(loc='upper left') #, bbox_to_anchor=(1.0, 0.5)
# plt.savefig("pruneabletree/tests/results/f1.pdf", bbox_inches="tight")

sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_accuracy").set(ylim=(0, 1))
plt.savefig("pruneabletree/tests/results/acc.pdf", bbox_inches="tight")
sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_f1").set(ylim=(0, 1))
plt.savefig("pruneabletree/tests/results/f1.pdf", bbox_inches="tight")
p = sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_n_nodes")
p.set(ylim=(1, None))
p.axes[0][0].set_yscale("log", basey=2)
plt.savefig("pruneabletree/tests/results/nodes.pdf", bbox_inches="tight")
p = sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_n_leaves")
p.set(ylim=(1, None))
p.axes[0][0].set_yscale("log", basey=2)
plt.savefig("pruneabletree/tests/results/leaves.pdf", bbox_inches="tight")
sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_fit_time_ms")
plt.savefig("pruneabletree/tests/results/fit.pdf", bbox_inches="tight")
sns.factorplot(col="dataset", hue="prune_method", x="tool", data=filtered, kind="bar", ci=None, y="mean_score_time_ms")
plt.savefig("pruneabletree/tests/results/score.pdf", bbox_inches="tight")