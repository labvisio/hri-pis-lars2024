import cv2
import os
from os import path
import time
import numpy as np
from easydict import EasyDict as edict
import yaml

from is_skeleton_2d import Skeleton2D
from utils.plot_2d_pose import Plot2DPose
from utils.is_annotations import Skeleton2DAnnotations

from is_wire.core import Channel, Message


def load_conf(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return exp_config


def main(conf, show_2d=False):
    is_skeleton = Skeleton2D(conf)
    plot_2d = Plot2DPose()
    sks_annotations = Skeleton2DAnnotations()

    # carregando imagens do dataset antigo da viros
    dataset_dir = "/home/lucasfoll/Desktop/Datasets/new_visio/p001g10"  # p001g01 - p003g01
    img_paths = os.listdir(dataset_dir)
    cam_0 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c00s" in img_path])
    cam_1 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c01s" in img_path])
    cam_2 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c02s" in img_path])
    cam_3 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c03s" in img_path])

    for frame_id in range(len(cam_0)):
        print(f"FRAME_ID: {frame_id}")
        image_list = [cv2.imread(cam_0[frame_id]), cv2.imread(cam_1[frame_id]), cv2.imread(cam_2[frame_id]),
                      cv2.imread(cam_3[frame_id])]

        result_pose, person_bbox = is_skeleton.estimate(image_list)

        if any(result_pose):
            im_height, im_width = image_list[0].shape[:2]
            obj_annotations = {}
            for cam_id, poses_2d in enumerate(result_pose):
                obs = sks_annotations.to_object_annotations(poses_2d, im_width, im_height)
                obj_annotations[cam_id] = obs  # TODO: Colocar para enviar para o broker, e não guardar em um dict
            if show_2d:
                stacked_img = plot_2d.get_stacked_images(cam_imgs=image_list, human_poses=result_pose, person_bbox=person_bbox)
                cv2.imshow("2D POSES", stacked_img)

        cv2.waitKey(1)


if __name__ == "__main__":
    # TODO: Isso pode ser passado como variável de ambiente
    cfg = load_conf('configs/service_configs_lab.yaml')
    main(cfg, show_2d=True)
