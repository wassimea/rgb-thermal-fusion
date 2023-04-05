import matplotlib
import json
import os

matplotlib.use("AGG")

import matplotlib.pyplot as plt
import matplotlib.colors as pltc

from sklearn.metrics import auc
from util_config import config as util_cfg


DETECTION_METHOD = util_cfg.METHOD
METADATA_DIR = util_cfg.metadata_path_name
METADATA_PATH = os.path.join(METADATA_DIR, DETECTION_METHOD)

os.makedirs(METADATA_PATH, exist_ok=True)


results = json.load(open(util_cfg.pr_json_results_out_file, "r"))
color_map = json.load(open(util_cfg.color_map_path, "r"))
# results = json.load(open(config.pr_json_results_out_file, "r"))

sequence = "2"

experiments = results[sequence]


def get_experiment_precision_recall(exp_name, experiments):
    for exp in experiments:
        if exp["name"] == exp_name:
            return {
                "recall": exp["recall"],
                "precision": exp["precision"],
            }


all_exp = [
    # "MUL",
    # "MUL_ENHANCED_GATES",
    # "ADD",
    # "ADD_ENHANCED_GATES",
    "MFB",
    "MFB_ENHANCED_GATES",
    # "BGF",
    # "BGF_ENHANCED_GATES",
    "RGB",
    "THERMAL",
]
cnxt_exp = [
    "MUL_CNXT",
    "MUL_CNXT_ENHANCED_GATES",
    "ADD_CNXT",
    "ADD_CNXT_ENHANCED_GATES",
    "MFB_CNXT",
    "MFB_CNXT_ENHANCED_GATES",
    "BGF_CNXT",
    "BGF_CNXT_ENHANCED_GATES",
    "RGB_CNXT",
    "THERMAL_CNXT",
]
# all_exp = cnxt_exp
general_exps = [
    # "MUL",
    # "ADD",
    # "MFB",
    "BGF",
    "RGB",
    "THERMAL",
]
save_dir = os.path.join(util_cfg.plots_save_path, DETECTION_METHOD)
print("save_dir", save_dir)
os.makedirs(save_dir, exist_ok=True)
# all_colors = [k for k, v in pltc.cnames.items()]
all_colors = [
    "blue",
    "black",
    "green",
    "cyan",
    "red",
    "orange",
    "pink",
    "purple",
    "blueviolet",
    "brown",
    "darkseagreen",
]

for exp in experiments:
    print(exp['name'], len(exp['all_tp_scores']), exp['mean_tp_conf_score'])


fig, ax = plt.subplots()
# ax.set_aspect('equal', adjustable='box')


color_idx = 0
for exp in experiments:
    if exp["name"] not in all_exp:
        continue
    all_tp_scores = sorted(exp["all_tp_scores"])
    # all_tp_scores = [round(s, 2) for s in all_tp_scores]
    ax.plot(
        all_tp_scores,
        color=all_colors[color_idx],
        marker=".",
        label=exp['name'],# + "=" + str(round(all_auc[exp_name], 3)),
        markersize=1,
        linewidth=2,
        alpha=0.8,
    )
    color_idx += 1

# add axis labels to plot
# ax.set_xlim([0.0, 1])
# ax.set_ylim([0.0, 1])
ax.set_title("Confidence Scores of True Positives")
ax.set_ylabel("True Positive Scores")
ax.set_xlabel("Number of True Positives")
fig.legend(loc="center right", prop={"size": 7})
# display plot
# plt.show()
fig.savefig(
    "{}/true-positive.png".format(save_dir),
    dpi=300,
)
plt.close(fig)
