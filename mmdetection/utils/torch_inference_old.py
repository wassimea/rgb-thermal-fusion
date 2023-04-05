import cv2
import numpy as np
import time

import os

from mmdet.apis import init_detector, inference_detector, show_result_pyplot

class Detector():
    def __init__(self, config_file, model_dir, confidence):
        #print('Loading model...', end='')
        start_time = time.time()
        device = 'cuda:0'
        # init a detector
        self.model = init_detector(config_file, model_dir, device=device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print('Done! Took {} seconds'.format(elapsed_time))

        self.confidence = confidence
    
    def predict(self, img_path, min_area=0):
        classes_interest = {
            0: "car",
        }
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
    minimum_area = 300
    thermal_images_folder = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/thermal/val/"
    rgb_images_folder = "/media/wassimea/Storage/datasets/Thermal_RGB_SC/rgb/val/"

    config_file = '/home/yalaa/yahya/pbvs23/mmdetection/work_dirs/tood_rgb/tood_rgb.py'
    checkpoint_file = '/home/yalaa/yahya/pbvs23/mmdetection/work_dirs/tood_rgb/epoch_12.pth'
    device = 'cuda:0'
    # init a detector
    detector = Detector(config_file, checkpoint_file, 0.4)
    x = 1

    image_filenames = [f for f in os.listdir(thermal_images_folder) if os.path.isfile(os.path.join(thermal_images_folder, f))]

    for image_filename in image_filenames:
        thermal_in_image = cv2.imread(os.path.join(thermal_images_folder, image_filename))
        rgb_in_image = cv2.imread(os.path.join(rgb_images_folder, image_filename))
        in_arr = np.concatenate((thermal_in_image, rgb_in_image), 2)

        img_draw = cv2.imread(os.path.join(rgb_images_folder, image_filename))
        img_path = os.path.join(thermal_images_folder, image_filename)
        detections, scores, classes_ids, classes_txt = detector.predict(in_arr, min_area=minimum_area)

        for i in range(len(detections)):
            x1, y1, x2, y2 = detections[i]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
#
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(img_draw,str(classes_txt[i]), (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        cv2.imshow("img_draw", img_draw)
        cv2.waitKey()



if __name__ == "__main__":
    process()
