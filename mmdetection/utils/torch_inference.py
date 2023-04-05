import random
import cv2
import numpy as np
import time
import bbox_visualizer as bbv
import os

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import CocoDataset


class Detector:
    def __init__(self, config_file, model_dir, confidence):
        # print('Loading model...', end='')
        start_time = time.time()
        device = "cuda:0"
        # init a detector
        self.model = init_detector(config_file, model_dir, device=device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print('Done! Took {} seconds'.format(elapsed_time))

        self.confidence = confidence

    def predict(self, img_path, min_area=0):
        classes_interest = {0: "car"}
        outs = inference_detector(self.model, img_path)

        detections = []
        scores = []
        classes_ids = []
        classes_txt = []

        for class_id in classes_interest:
            dets = outs[class_id]
            for det in dets:
                x1, y1, x2, y2, score = det
                w = x2 - x1
                h = y2 - y1
                if w * h >= min_area:
                    if score > self.confidence:
                        detections.append([x1, y1, x2, y2])
                        scores.append(score)
                        classes_ids.append(class_id)
                        classes_txt.append(classes_interest[class_id])

        detections = np.array(detections)
        scores = np.array(scores)
        classes_ids = np.array(classes_ids)
        classes_txt = np.array(classes_txt)

        return detections, scores, classes_ids, classes_txt

    def predict_fairmot(self, img_path):
        retlist = []
        detections, scores, classes_ids, classes_txt = self.predict(input)
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections[i]
            preds = [x1, y1, x2, y2, scores[i], classes_ids[i]]
            retlist.append(preds)
        return np.array(retlist)


def process():
    minimum_area = 150
    thermal_images_folder = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/thermal/val/"
    rgb_images_folder = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/rgb/val/"

    annotations_path = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/coco_val.json"

    # method, ckpt_name, epoch = 'rgb-only', 'tood_rgb', 2
    # method, ckpt_name, epoch = 'thermal-only', 'tood_thermal', 8
    # method, ckpt_name, epoch = 'add', 'tood_ADD', 2
    # method, ckpt_name, epoch = 'mul', 'tood_MUL', 3
    # method, ckpt_name, epoch = 'mfb', 'tood_MFB', 1
    # method, ckpt_name, epoch = 'bgf', 'tood_BGF', 4

    all_exp_triplets = [
        ('rgb-only', 'tood_rgb', 2),
        ('thermal-only', 'tood_thermal', 8),
        # ('add', 'tood_ADD', 2),
        # ('mul', 'tood_MUL', 3),
        # ('mfb', 'tood_MFB', 1),
        ('bgf', 'tood_BGF', 4),
        # ('enhanced_bgf', 'tood_BGF_enhanced_gates', 1),
    ]

    det_thresh = 0.4

    for method, ckpt_name, epoch in all_exp_triplets:
        config_file = "/home/yalaa/yahya/pbvs23/mmdetection/work_dirs/{}/{}.py".format(ckpt_name, ckpt_name)
        checkpoint_file = "/home/yalaa/yahya/pbvs23/mmdetection/work_dirs/{}/epoch_{}.pth".format(ckpt_name, epoch)
        device = "cuda:0"

        output_dir = 'vis_results'

        dataset = CocoDataset(annotations_path, [])



        # init a detector
        detector = Detector(config_file, checkpoint_file, det_thresh)
        x = 1

        image_filenames = [
            f
            for f in os.listdir(thermal_images_folder)
            if os.path.isfile(os.path.join(thermal_images_folder, f))
        ]
        chosen_images = [
            # '21-08-05_17-10-07_000036.png',
            '21-08-05_17-10-07_000081.png',
            # '21-08-05_19-33-00_000128.png', # very good!!
            '21-08-05_21-20-46_000003.png', # dark - very good!!
        ]
        chosen_images_ids = [
            # '210805171007000035',
            '210805171007000080',
            # '210805193300000127',
            '210805212046000002'
        ]

        from pprint import pprint


        # coco_imgs = dataset.coco.load_imgs(chosen_images_ids)
        # pprint(dataset.coco.get_ann_ids(chosen_images_ids))
        # # pprint(dataset.get_ann_info())

        # return


        for image_filename, image_coco_id in zip(chosen_images, chosen_images_ids):
            thermal_in_image = cv2.imread(
                os.path.join(thermal_images_folder, image_filename)
            )
            rgb_in_image = cv2.imread(os.path.join(rgb_images_folder, image_filename))
            in_arr = np.concatenate((thermal_in_image, rgb_in_image), 2)

            rgb_img_draw = cv2.imread(os.path.join(rgb_images_folder, image_filename))
            thermal_img_draw = cv2.imread(os.path.join(thermal_images_folder, image_filename))
            # img_path = os.path.join(thermal_images_folder, image_filename)
            detections, scores, classes_ids, classes_txt = detector.predict(
                in_arr, min_area=minimum_area
            )

            bboxes_ids = dataset.coco.get_ann_ids([image_coco_id])
            bboxes = []
            for bbox_id in bboxes_ids:
                bbox = dataset.coco.loadAnns([bbox_id])[0]
                bboxes.append(bbox['bbox']) if bbox['category_id'] == 1 else None

            line_thickness = 2

            ### GT BOXES
            for i in range(len(bboxes)):
                x, y, w, h = bboxes[i]
                if w * h <= minimum_area:
                    continue
                x1, y1, x2, y2 = x, y, x + w, y + h
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                #
                # cv2.rectangle(rgb_img_draw, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
                # cv2.rectangle(thermal_img_draw, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

                bbox = [x1, y1, x2, y2]
                rgb_img_draw = bbv.draw_rectangle(rgb_img_draw, bbox, bbox_color=(0, 255, 0), thickness=2)
                thermal_img_draw = bbv.draw_rectangle(thermal_img_draw, bbox, bbox_color=(0, 255, 0), thickness=2)

            ## PREDICTIONS
            cv2.rectangle(thermal_img_draw, (0, 0), (100, 22*len(detections)), (255, 255, 255), -1)
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections[i]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                #
                # cv2.rectangle(rgb_img_draw, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
                # # write score as text
                text_id_pos = (x2 + 5, y2 - 5) #if random.random() < 0.5 else (x1, y1 - 5)
                cv2.putText(
                    rgb_img_draw,
                    str(i), #"{}%".format(int(scores[i] * 100)),
                    text_id_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    thermal_img_draw,
                    str(i), #"{}%".format(int(scores[i] * 100)),
                    text_id_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                # write score as text in the top right corner of image
                text_pos = (10, 20+i*21) #if random.random() < 0.5 else (x1, y1 - 5)
                cv2.putText(
                    thermal_img_draw,
                    "{}: {}%".format(i, int(scores[i] * 100)),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                )
                # cv2.rectangle(thermal_img_draw, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)

                bbox = [x1, y1, x2, y2]
                rgb_img_draw = bbv.draw_rectangle(rgb_img_draw, bbox, bbox_color=(0, 0, 255), thickness=2)
                # rgb_img_draw = bbv.add_label(rgb_img_ddraw, str(int(scores[i] * 100)), bbox, size=1)
                thermal_img_draw = bbv.draw_rectangle(thermal_img_draw, bbox, bbox_color=(0, 0, 255), thickness=2)

            # concatenate both rgb_img_draw and thermal_img_draw images horizontally
            img_draw = np.concatenate((rgb_img_draw, thermal_img_draw), 1)
            cv2.imwrite(os.path.join(output_dir, method+image_filename), img_draw)

            # cv2.imwrite(os.path.join(output_dir, method+'-R-'+image_filename), rgb_img_draw)
            # cv2.imwrite(os.path.join(output_dir, method+'-T-'+image_filename), thermal_img_draw)

            # cv2.imshow("img_draw", img_draw)
            # cv2.waitKey()


if __name__ == "__main__":
    process()
