import os
import torch

from backend.YOLOv3.yolov3 import YOLOv3
from backend.YOLOv8.yolov8 import YOLOv8
from backend.RTMDet.rtmdet import RTMDet
from backend.RTMPose.rtmpose import RTMPose
from is_2d_skeleton.utils.is_annotations import Skeleton2DAnnotations


def get_gpu_name():
    if not torch.cuda.is_available():
        raise Exception("Cannot find gpu device!")
    name = torch.cuda.get_device_name(0)
    print("Found gpu={}", name)
    return name.lower().replace(" ", "-")


class Skeleton2D:

    def __init__(self, service_cfg):
        self._person_det_cfg = None
        self._pose_det_cfg = None
        self._person_det = None
        self._pose_det = None
        self._load_conf(service_cfg)
        self.sks_annotations = Skeleton2DAnnotations()

    def _load_conf(self, cfg):
        pipeline = cfg.PIPELINE_COMBINATION
        self._person_det_cfg = cfg.DETECT_MODELS[pipeline["DETECT_MODEL"].upper()]
        self._pose_det_cfg = cfg.POSE_MODELS[pipeline["POSE_MODEL"].upper()]

        self._person_det = self._get_person_detector()
        self._pose_det = self._get_pose_detector_2d()

    def _get_person_detector(self):
        assert self._person_det_cfg is not None

        if self._person_det_cfg.NAME == "YOLOv3":
            person_detector = YOLOv3(
                self._person_det_cfg.CFG,
                self._person_det_cfg.WEIGHT,
                self._person_det_cfg.CLASS_NAMES,
                score_thresh=self._person_det_cfg.SCORE_THRESH,
                nms_thresh=self._person_det_cfg.NMS_THRESH,
                use_cuda=self._person_det_cfg.USE_CUDA,
            )
            print("Person Detector : ", self._person_det_cfg.NAME)
            return person_detector

        elif self._person_det_cfg.NAME == "RTMDet":
            person_detector = RTMDet(
                model_path=self._person_det_cfg.MODEL,
                device_name=self._person_det_cfg.DEVICE_NAME,
                device_id=self._person_det_cfg.DEVICE_ID,
            )
            print("Person Detector : ", self._person_det_cfg.NAME)
            return person_detector

        elif self._person_det_cfg.NAME == "YOLOv8":
            person_detector = YOLOv8(
                model=self._person_det_cfg.MODEL,
                device=self._person_det_cfg.DEVICE,
                score_threshold=self._person_det_cfg.SCORE_THRESH,
                iou=self._person_det_cfg.IOU,
            )
            print("Person Detector : ", self._person_det_cfg.NAME)
            return person_detector

        else:
            raise NotImplementedError(
                f"{self._person_det_cfg.NAME} Person Detector has not yet been implemented."
            )

    def _get_pose_detector_2d(self):
        assert self._pose_det_cfg is not None

        if self._pose_det_cfg.NAME == "RTMPose":
            gpu_name = get_gpu_name()
            model_path = os.path.join(
                self._pose_det_cfg.MODEL,
                gpu_name,
            )
            pose_detector = RTMPose(
                model_path=model_path,
                device_name=self._pose_det_cfg.DEVICE_NAME,
                device_id=self._pose_det_cfg.DEVICE_ID,
            )
            print("Pose Detector : ", self._pose_det_cfg.NAME)
            return pose_detector

        else:
            raise NotImplementedError(
                f"{self._pose_det_cfg.NAME} Pose Detector has not yet been implemented."
            )

    def estimate(self, image_list):
        person_bbox = self._person_det(image_list)
        result_pose = self._pose_det(image_list, person_bbox)

        obs2d_dict = {}
        for c_id, pose in enumerate(result_pose):
            im_height, im_width = image_list[0].shape[:2]
            obs = self.sks_annotations.to_object_annotations(pose, im_width, im_height)
            obs2d_dict[c_id] = obs
        return obs2d_dict, result_pose, person_bbox
