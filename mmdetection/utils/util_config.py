# coding:utf-8

from easydict import EasyDict as edict

config = edict()

MMDET_PATH = "/home/yalaa/yahya/pbvs23/mmdetection/"

# TOOD = "TOOD"

config.METHOD = 'TOOD' # 'TOOD-CNXT' # 
config.metadata_path_name = "metadata"

config.work_dirs_path = MMDET_PATH + "work_dirs/"
config.mmdet_configs_path = MMDET_PATH + "configs/"

config.mean_ap_json_path = (
    MMDET_PATH
    + "utils/{}/{}/mean_average_precision_data.json".format(
        config.metadata_path_name,
        config.METHOD,
    )
)
config.best_performing_epochs_json_path = (
    MMDET_PATH
    + "utils/{}/{}/best_performing_epochs.json".format(
        config.metadata_path_name,
        config.METHOD,
    )
)
config.plots_save_path = MMDET_PATH + "utils/plots/"
config.color_map_path = MMDET_PATH + "utils/{}/color_map.json".format(
    config.metadata_path_name
)


config.ds_thermal_val_dir = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/thermal/val/"
config.ds_rgb_val_dir = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/rgb/val/"
config.ds_val_gt_path = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/coco_val.json"

config.pr_txt_results_out_file = MMDET_PATH + "utils/results_files/pr_results.txt"
config.pr_json_results_out_file = MMDET_PATH + "utils/{}/{}/pr_results.json".format(
    config.metadata_path_name,
    config.METHOD,
)

config.minimum_area = 0
