import matplotlib
import json
import os

matplotlib.use("AGG")

import matplotlib.pyplot as plt
import matplotlib.colors as pltc

from sklearn.metrics import auc
from util_config import config as util_cfg


"""
https://sanchom.wordpress.com/tag/average-precision/
https://glassboxmedicine.com/2020/07/14/the-complete-guide-to-auc-and-average-precision-simulations-and-visualizations/#:~:text=Average%20precision%20is%20one%20way,many%20negative%20examples%20as%20positive.
https://stats.stackexchange.com/questions/157012/area-under-precision-recall-curve-auc-of-pr-curve-and-average-precision-ap
"""

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
    "MUL",
    "MUL_ENHANCED_GATES",
    "ADD",
    "ADD_ENHANCED_GATES",
    "MFB",
    "MFB_ENHANCED_GATES",
    "BGF",
    "BGF_ENHANCED_GATES",
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
all_exp = cnxt_exp
general_exps = [
    # "MUL",
    # "ADD",
    # "MFB",
    "BGF",
    "RGB",
    "THERMAL",
]
plot_name = "-".join(general_exps)
save_dir = os.path.join(util_cfg.plots_save_path, DETECTION_METHOD, plot_name)
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

plotting_exp = []
for general_exp in general_exps:
    for exp in all_exp:
        if general_exp in exp:
            plotting_exp.append(exp)

fig, ax = plt.subplots()
# ax.set_aspect('equal', adjustable='box')

padded_recall = {}
padded_precision = {}
all_auc = {}


for exp_name in plotting_exp:
    padded_recall[exp_name] = (
        [1]
        + get_experiment_precision_recall(exp_name, experiments=experiments)["recall"]
        + [0]
    )
    padded_precision[exp_name] = (
        [0]
        + get_experiment_precision_recall(exp_name, experiments=experiments)[
            "precision"
        ]
        + [1]
    )
    all_auc[exp_name] = auc(padded_recall[exp_name], padded_precision[exp_name])

sorted_plotted_exp = sorted(plotting_exp, key=lambda x: all_auc[x], reverse=True)

color_idx = 0
for exp_name in sorted_plotted_exp:
    ax.plot(
        padded_recall[exp_name],
        padded_precision[exp_name],
        color=color_map.get(exp_name, all_colors[color_idx]),
        marker=".",
        label=exp_name,# + "=" + str(round(all_auc[exp_name], 3)),
        markersize=4,
        linewidth=2,
        alpha=0.8,
    )
    color_idx += 1

# add axis labels to plot
ax.set_xlim([0.0, 1])
ax.set_ylim([0.0, 1])
ax.set_title("Precision-Recall Curve")
ax.set_ylabel("Precision")
ax.set_xlabel("Recall")
fig.legend(loc="lower left", prop={"size": 10})
# display plot
# plt.show()
fig.savefig(
    "{}/pr-curve-{}.png".format(save_dir, plot_name),
    # config.plots_save_path + "/pr-curve-"
    # # + str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))
    # + "-".join(plotting_exp) + ".png",
    dpi=300,
)
plt.close(fig)
