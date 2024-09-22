import numpy as np
from mmdeploy_runtime import PoseDetector

from utils.interfaces.pose_detector import PoseDetectorInterface


class RTMPose(PoseDetectorInterface):

    def __init__(self, model_path, device_name='cuda', device_id=0):
        self._detector = PoseDetector(model_path=model_path, device_name=device_name, device_id=device_id)

    def __call__(self, image_list, bbox_list):
        results = []
        n_cams = len(bbox_list)
        for c in range(n_cams):
            image = image_list[c]
            cam_results = []
            n_persons = len(bbox_list[c])
            for p in range(n_persons):
                bbox = bbox_list[c][p]
                bbox = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
                keypoints = self._detector(image, bbox)
                keypoints = keypoints.reshape(-1, 3)
                result = keypoints.round(3)
                cam_results.append(result)

            results.append(cam_results)
        return results
