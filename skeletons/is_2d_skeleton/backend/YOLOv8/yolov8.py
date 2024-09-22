import numpy as np
from ultralytics import YOLO

from utils.interfaces.person_detector import PersonDetectorInterface


class YOLOv8(PersonDetectorInterface):

    def __init__(self, model, device='cuda', score_threshold=0.3, iou=0.7):
        self._detector = YOLO(model)
        self._device = device
        self._score_threshold = score_threshold
        self._iou = iou

    def __call__(self, image_list):
        predicts = self._detector.predict(image_list,
                                          device=self._device,
                                          conf=self._score_threshold,
                                          iou=self._iou,
                                          classes=0,
                                          verbose=False)
        results = []
        for i in range(len(predicts)):
            h, w, _ = image_list[i].shape
            boxes = predicts[i].boxes
            bboxes = boxes.data.cpu().numpy()

            img_results = []
            for bbox in bboxes:
                x1 = max(0, bbox[0])
                y1 = max(0, bbox[1])
                x2 = min(bbox[2], w)
                y2 = min(bbox[3], h)
                score = bbox[4]

                person_dict = [x1, y1, x2 - x1, y2 - y1, score]
                img_results.append(person_dict)

            results.append(img_results)
        return results
