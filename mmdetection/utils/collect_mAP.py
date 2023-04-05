import os, json, matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util_config import config as util_cfg

matplotlib.use("AGG")

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
    "teal",
    "gray",
    "chocolate",
    "slateblue",
]

DETECTION_METHOD = util_cfg.METHOD
METADATA_DIR = "metadata"
METADATA_PATH = os.path.join(METADATA_DIR, DETECTION_METHOD)

os.makedirs(METADATA_PATH, exist_ok=True)


def collect_mean_average_precision_data(fusion="", color_map={}, save_plots=True):
    exps = {
        #------TOOD-ResNet50------#
        "MUL": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_MUL"),
        },
        "MUL_ENHANCED_GATES": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_MUL_enhanced_gates"),
        },
        "ADD": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_ADD"),
        },
        "ADD_ENHANCED_GATES": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_ADD_enhanced_gates"),
        },
        "MFB": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_MFB"),
        },
        "MFB_ENHANCED_GATES": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_MFB_enhanced_gates"),
        },
        "BGF": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_BGF"),
        },
        "BGF_ENHANCED_GATES": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_BGF_enhanced_gates"),
        },
        # "BON": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_pbvs_BON"),
        # },
        "RGB": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_rgb"),
        },
        "THERMAL": {
            "ckpt_path": os.path.join(util_cfg.work_dirs_path, "tood_thermal"),
        },
        # #------TOOD-ConvNext------#
        # "MUL_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_MUL"),
        # },
        # "MUL_CNXT_ENHANCED_GATES": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_MUL_enhanced_gates"),
        # },
        # "ADD_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_ADD"),
        # },
        # "ADD_CNXT_ENHANCED_GATES": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_ADD_enhanced_gates"),
        # },
        # "MFB_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_MFB"),
        # },
        # "MFB_CNXT_ENHANCED_GATES": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_MFB_enhanced_gates"),
        # },
        # "BGF_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_BGF"),
        # },
        # "BGF_CNXT_ENHANCED_GATES": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_BGF_enhanced_gates"),
        # },
        # "RGB_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_rgb"),
        # },
        # "THERMAL_CNXT": {
        #     "ckpt_path": os.path.join(util_cfg.work_dirs_path, "toodcnext_pbvs_thermal"),
        # },
    }

    main_exps =  ["RGB", "THERMAL"] # ["RGB_CNXT", "THERMAL_CNXT"] #

    mean_average_precision_data = {}

    if save_plots:
        cur_exp = [fusion] + main_exps
        save_dir = os.path.join("plots", DETECTION_METHOD, "-".join(cur_exp))
        os.makedirs(save_dir, exist_ok=True)

    for exp_name, exp_data in exps.items():
        if fusion in exp_name or exp_name in main_exps:
            ckpt_path = exp_data["ckpt_path"]
            log_file = list(filter(lambda k: ".json" in k, os.listdir(ckpt_path)))[0]
            log_file_path = os.path.join(ckpt_path, log_file)

            mean_average_precision_data[exp_name] = []

            with open(log_file_path) as f:
                for json_obj in f:
                    record = json.loads(json_obj)
                    if record.get("mode", "") == "val":
                        mean_average_precision_data[exp_name].append(record)

    with open(util_cfg.mean_ap_json_path, "w") as f:
        # with open(util_cfg.mean_ap_json_path, "w") as f:
        json.dump(mean_average_precision_data, f, indent=4)

    fusion_op = "all" if fusion == "" else fusion

    if save_plots:
        plot_map_curve(
            mean_average_precision_data,
            iou=50,
            fusion_op=fusion_op,
            color_map=color_map,
            save_dir=save_dir,
        )
        plot_map_curve(
            mean_average_precision_data,
            iou=75,
            fusion_op=fusion_op,
            color_map=color_map,
            save_dir=save_dir,
        )

    best_map_data = generate_best_performing_epochs(mean_average_precision_data)

    if save_plots:
        plot_best_performing_epochs(
            best_map_data,
            iou=50,
            fusion_op=fusion_op,
            color_map=color_map,
            save_dir=save_dir,
        )
        plot_best_performing_epochs(
            best_map_data,
            iou=75,
            fusion_op=fusion_op,
            color_map=color_map,
            save_dir=save_dir,
        )


def generate_best_performing_epochs(mean_average_precision_data):
    # generate a json file that contains the best performing epoch for each experiment
    best_map_data = {}
    for exp_name, exp_data in mean_average_precision_data.items():
        best_map_data[exp_name] = max(exp_data, key=lambda k: k["bbox_mAP_50"])

    with open(util_cfg.best_performing_epochs_json_path, "w") as f:
        # with open(util_cfg.best_performing_epochs_json_path, "w") as f:
        json.dump(best_map_data, f, indent=4)

    return best_map_data


def plot_best_performing_epochs(
    best_map_data, iou=50, fusion_op="all", color_map={}, save_dir="plots"
):
    fig, ax = plt.subplots()

    metric = "bbox_mAP_{}".format(iou)

    ax.set_title("Best Performing Checkpoints (IoU={}%)".format(iou))
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")
    ax.yaxis.set_ticks(np.arange(0, 1, 0.02))
    # ax.set_ylim([0, 1])

    best_map_data = dict(
        sorted(best_map_data.items(), key=lambda item: item[1][metric], reverse=True)
    )

    color_idx = 0
    for exp_name, exp_data in best_map_data.items():
        ax.scatter(
            exp_data["epoch"],
            exp_data[metric],
            s=(exp_data[metric] * 30) ** 1.25,
            label=exp_name + "={:.3f}".format(exp_data[metric]),
            color=color_map.get(exp_name, all_colors[color_idx]),
        )
        ax.axhline(
            y=exp_data[metric],
            linestyle="--",
            color=color_map.get(exp_name, all_colors[color_idx]),
            alpha=0.3,
        )
        color_idx += 1

    fig.legend(loc="lower right", prop={"size": 5.5})
    fig.savefig(
        "{}/best_performing_checkpoints_{}_fusion_{}.png".format(
            save_dir, metric, fusion_op
        ),
        # util_cfg.plots_save_path + "best_performing_checkpoints_{}_fusion_{}.png".format(metric, fusion_op),
        dpi=300,
    )
    plt.close(fig)


def plot_map_curve(
    mean_average_precision_data, iou=50, fusion_op="all", color_map={}, save_dir="plots"
):
    fig, ax = plt.subplots()

    metric = "bbox_mAP_{}".format(iou)

    ax.set_title("mAP Curve (IoU={}%)".format(iou))
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")

    color_idx = 0
    for exp_name, exp_data in mean_average_precision_data.items():
        x = []
        y = []
        for record in exp_data:
            x.append(record["epoch"])
            y.append(record[metric])

        ax.plot(
            x,
            y,
            label=exp_name,
            color=color_map.get(exp_name, all_colors[color_idx]),
        )

    fig.legend(loc="lower right", prop={"size": 5.5})
    fig.savefig(
        "{}/map_curve_{}_fusion_{}.png".format(save_dir, metric, fusion_op), dpi=300
    )
    # fig.savefig(util_cfg.plots_save_path + "map_curve_{}_fusion_{}.png".format(metric, fusion_op), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    color_map = json.load(open(util_cfg.color_map_path))

    collect_mean_average_precision_data(fusion="MUL", color_map=color_map)
    collect_mean_average_precision_data(fusion="ADD", color_map=color_map)
    collect_mean_average_precision_data(fusion="MFB", color_map=color_map)
    collect_mean_average_precision_data(fusion="BGF", color_map=color_map)
    # collect_mean_average_precision_data(fusion="BON", color_map=color_map)
    # collect_mean_average_precision_data(fusion="RGB_SCRATCH", color_map=color_map)
    # collect_mean_average_precision_data(fusion="THERMAL_SCRATCH", color_map=color_map)
    collect_mean_average_precision_data(
        fusion="", color_map=color_map, save_plots=True
    )
