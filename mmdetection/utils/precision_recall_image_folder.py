import cv2

# import tensorflow as tf
import numpy as np
import time

import os
import json

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from utils.collect_mAP import collect_mean_average_precision_data
from torch_inference import Detector
from util_config import config as util_cfg


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_json_data(path):
    f = open(path)
    data = json.load(f)
    return data


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_metrics_normal(
    pred_boxes, gt_boxes, pred_classes, gt_classes, pred_scores, confidence
):
    tp = 0
    fp = 0
    fn = 0

    tp_scores = []

    filtered_pred_boxes = []
    filtered_pred_scores = []
    filtered_pred_classes = []

    adjusted_gt_boxes = []  # from xywh to x1y1x2y2

    used_gt_indices = []

    for i in range(len(pred_scores)):
        if pred_scores[i] > confidence:
            filtered_pred_boxes.append(pred_boxes[i])
            filtered_pred_scores.append(pred_scores[i])
            filtered_pred_classes.append(pred_classes[i])

    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        adjusted_gt_boxes.append([x, y, x + w, y + h])

    for p in range(len(filtered_pred_boxes)):
        found = False
        for q in range(len(adjusted_gt_boxes)):
            if q not in used_gt_indices:
                iou = bb_intersection_over_union(
                    filtered_pred_boxes[p], adjusted_gt_boxes[q]
                )
                if iou > 0.5:  # and filtered_pred_classes[p] == gt_classes[q]:
                    found = True
                    used_gt_indices.append(q)
                    break
        if found:
            tp += 1
            tp_scores.append(filtered_pred_scores[p].item())
        else:
            fp += 1
    fn = len(adjusted_gt_boxes) - len(used_gt_indices)

    return tp, fp, fn, tp_scores


def get_image_gt(sample_name, gt_data, min_area=0):
    rgb_image_id = -1
    for image_entry_rgb in gt_data["images"]:
        if image_entry_rgb["file_name"] == sample_name:
            rgb_image_id = image_entry_rgb["id"]
            break
    if rgb_image_id == -1:
        return [], []

    rgb_annotations = []
    thermal_annotations = []

    for rgb_annotation in gt_data["annotations"]:
        if rgb_annotation["image_id"] == rgb_image_id:
            rgb_annotations.append(rgb_annotation)

    rgb_boxes = []
    rgb_classes = []

    for rgb_ann in rgb_annotations:
        x, y, w, h = rgb_ann["bbox"]
        if w * h >= min_area:
            if rgb_ann["category_id"] != 1:
                continue
            rgb_boxes.append(rgb_ann["bbox"])
            rgb_classes.append(rgb_ann["category_id"])

    return rgb_boxes, rgb_classes


def fetch_best_performing_epoch(data, exp_name, default_epoch=2):
    if exp_name in data:
        return data[exp_name].get("epoch", default_epoch)
    return default_epoch


def process_image_folder(recompute_exps=[]):
    color_map = json.load(open(util_cfg.color_map_path))
    collect_mean_average_precision_data(
        fusion="", color_map=color_map, save_plots=False
    )

    minimum_area = util_cfg.minimum_area
    thermal_images_folder = util_cfg.ds_thermal_val_dir
    rgb_images_folder = util_cfg.ds_rgb_val_dir

    if os.path.exists(util_cfg.best_performing_epochs_json_path):
        best_performing_epochs = json.load(
            open(util_cfg.best_performing_epochs_json_path)
        )
    else:
        best_performing_epochs = {}

    exps = {
        "THERMAL": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_thermal.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_thermal/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "THERMAL")
            ),
        },
        "RGB": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_rgb.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_rgb/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "RGB")
            ),
        },
        "MUL": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_MUL.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_MUL/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "MUL")
            ),
        },
        "MUL_ENHANCED_GATES": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_MUL_enhanced_gates.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_MUL_enhanced_gates/epoch_{}.pth".format(
                fetch_best_performing_epoch(
                    best_performing_epochs, "MUL_ENHANCED_GATES"
                )
            ),
        },
        "ADD": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_ADD.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_ADD/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "ADD")
            ),
        },
        "ADD_ENHANCED_GATES": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_ADD_enhanced_gates.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_ADD_enhanced_gates/epoch_{}.pth".format(
                fetch_best_performing_epoch(
                    best_performing_epochs, "ADD_ENHANCED_GATES"
                )
            ),
        },
        "BGF": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_BGF.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_BGF/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "BGF")
            ),
        },
        "BGF_ENHANCED_GATES": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_BGF_enhanced_gates.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_BGF_enhanced_gates/epoch_{}.pth".format(
                fetch_best_performing_epoch(
                    best_performing_epochs, "BGF_ENHANCED_GATES"
                )
            ),
        },
        "MFB": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_MFB.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_MFB/epoch_{}.pth".format(
                fetch_best_performing_epoch(best_performing_epochs, "MFB")
            ),
        },
        "MFB_ENHANCED_GATES": {
            "config": util_cfg.mmdet_configs_path + "tood/tood_MFB_enhanced_gates.py",
            "ckpt": util_cfg.work_dirs_path
            + "tood_MFB_enhanced_gates/epoch_{}.pth".format(
                fetch_best_performing_epoch(
                    best_performing_epochs, "MFB_ENHANCED_GATES"
                )
            ),
        },
    }
    # toodcnext_pbvs_exps = {
    #     "THERMAL_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_thermal.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_thermal/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "THERMAL_CNXT")
    #         ),
    #     },
    #     "RGB_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_rgb.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_rgb/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "RGB_CNXT")
    #         ),
    #     },
    #     "MUL_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_MUL.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_MUL/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "_CNXTMUL")
    #         ),
    #     },
    #     "MUL_CNXT_ENHANCED_GATES": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_MUL_enhanced_gates.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_MUL_enhanced_gates/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(
    #                 best_performing_epochs, "MUL_CNXT_ENHANCED_GATES"
    #             )
    #         ),
    #     },
    #     "ADD_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_ADD.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_ADD/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "ADD_CNXT")
    #         ),
    #     },
    #     "ADD_CNXT_ENHANCED_GATES": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_ADD_enhanced_gates.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_ADD_enhanced_gates/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(
    #                 best_performing_epochs, "ADD_CNXT_ENHANCED_GATES"
    #             )
    #         ),
    #     },
    #     "BGF_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_BGF.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_BGF/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "BGF_CNXT")
    #         ),
    #     },
    #     "BGF_CNXT_ENHANCED_GATES": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_BGF_enhanced_gates.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_BGF_enhanced_gates/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(
    #                 best_performing_epochs, "BGF_CNXT_ENHANCED_GATES"
    #             )
    #         ),
    #     },
    #     "MFB_CNXT": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_MFB.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_MFB/epoch_{}.pth".format(
    #             fetch_best_performing_epoch(best_performing_epochs, "MFB_CNXT")
    #         ),
    #     },
    #     "MFB_CNXT_ENHANCED_GATES": {
    #         "config": util_cfg.mmdet_configs_path + "tood/toodcnext_pbvs_MFB_enhanced_gates.py",
    #         "ckpt": util_cfg.work_dirs_path
    #         + "toodcnext_pbvs_MFB_enhanced_gates/epoch_{}.pth".format(  
    #             fetch_best_performing_epoch(
    #                 best_performing_epochs, "MFB_CNXT_ENHANCED_GATES"
    #             )
    #         ),
    #     }
    # }
    # exps = toodcnext_pbvs_exps
    # sequences = ["21-08-05_17-10-07", "21-08-05_19-33-00", "21-08-05_21-20-46", "2"]
    sequences = ["2"]

    device = "cuda:0"
    # init a detector

    image_filenames = [
        f
        for f in os.listdir(thermal_images_folder)
        if os.path.isfile(os.path.join(thermal_images_folder, f))
    ]

    gt_path = util_cfg.ds_val_gt_path

    confidences = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.95,
    ]

    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    file_stream = open(util_cfg.pr_txt_results_out_file, "w")

    if os.path.exists(util_cfg.pr_json_results_out_file):
        cur_results_json = json.load(open(util_cfg.pr_json_results_out_file))
    else:
        cur_results_json = {}
    results_json = {}

    for sequence in sequences:
        file_stream.write("\n")
        file_stream.write("\n")
        file_stream.write("sequence = " + '"' + sequence + '"')
        file_stream.write("\n")

        print("===========================")

        processed_exp_names = [
            entry["name"] for entry in cur_results_json.get(sequence, [])
        ]
        results_json[sequence] = cur_results_json.get(sequence, [])

        for exp in exps:
            if exp in processed_exp_names and exp not in recompute_exps:
                continue

            if exp in processed_exp_names and exp in recompute_exps:
                # remove all old entries that has name exp
                results_json[sequence] = [
                    entry for entry in results_json[sequence] if entry["name"] != exp
                ]

            try:
                detector = Detector(exps[exp]["config"], exps[exp]["ckpt"], 0)
            except:
                print("exp", exp, exps[exp]["ckpt"], exps[exp]["config"])
                continue
            tps = [0] * len(confidences)
            fps = [0] * len(confidences)
            fns = [0] * len(confidences)

            precisions = [0] * len(confidences)
            recalls = [0] * len(confidences)

            all_tp_scores = []

            for image_filename in image_filenames:
                if sequence in image_filename:
                    gt_boxes, gt_classes = get_image_gt(
                        image_filename, gt_data, min_area=minimum_area
                    )

                    thermal_in_image = cv2.imread(
                        os.path.join(thermal_images_folder, image_filename)
                    )
                    rgb_in_image = cv2.imread(
                        os.path.join(rgb_images_folder, image_filename)
                    )
                    in_arr = np.concatenate((thermal_in_image, rgb_in_image), 2)

                    img_draw = cv2.imread(
                        os.path.join(rgb_images_folder, image_filename)
                    )
                    img_path = os.path.join(thermal_images_folder, image_filename)
                    detections, scores, classes_ids, classes_txt = detector.predict(
                        in_arr, min_area=minimum_area
                    )

                    for w in range(len(confidences)):
                        tp, fp, fn, tp_scores = get_metrics_normal(
                            detections,
                            gt_boxes,
                            classes_ids,
                            gt_classes,
                            scores,
                            confidences[w],
                        )

                        # print('tp, fp, fn', tp, fp, fn, tp_scores, len(tp_scores) == tp)

                        tps[w] += tp
                        fps[w] += fp
                        fns[w] += fn

                        all_tp_scores.extend(tp_scores)

            for k in range(len(tps)):
                precisions[k] = tps[k] / max((tps[k] + fps[k]), 0.0000001)
                recalls[k] = tps[k] / max((tps[k] + fns[k]), 0.0000001)

            precisions_index_0 = -1
            recalls_index_0 = -1

            if 0.0 in precisions:
                precisions_index_0 = precisions.index(0.0)

            if 0.0 in recalls:
                recalls_index_0 = recalls.index(0.0)

            precisions = precisions[:precisions_index_0]
            recalls = recalls[:recalls_index_0]

            file_stream.write("precision_" + exp + " = " + str(precisions))
            file_stream.write("\n")
            file_stream.write("recall_" + exp + " = " + str(recalls))
            file_stream.write("\n")

            results_json[sequence].append(
                {
                    "name": exp,
                    "precision": precisions,
                    "recall": recalls,
                    "epoch": fetch_best_performing_epoch(best_performing_epochs, exp),
                    "mean_tp_conf_score": np.mean(all_tp_scores).item(),
                    "90_percentile_tp_conf_score": np.percentile(all_tp_scores, 90).item(),
                    "95_percentile_tp_conf_score": np.percentile(all_tp_scores, 95).item(),
                    "75_percentile_tp_conf_score": np.percentile(all_tp_scores, 75).item(),
                    "50_percentile_tp_conf_score": np.percentile(all_tp_scores, 25).item(),
                    "all_tp_scores": all_tp_scores,
                }
            )
    file_stream.close()

    # save json file to results_data.json file
    with open(util_cfg.pr_json_results_out_file, "w") as outfile:
        json.dump(results_json, outfile, indent=4)


if __name__ == "__main__":
    recompute_exps = []
    process_image_folder(recompute_exps=recompute_exps)
