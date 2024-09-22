import cv2
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import time
from easydict import EasyDict as edict
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "is_2d_skeleton"))
sys.path.append(os.path.join(os.path.dirname(__file__), "is_3d_skeleton"))

from is_2d_skeleton.is_skeleton_2d import Skeleton2D
from is_2d_skeleton.utils.plot_2d_pose import Plot2DPose

from is_3d_skeleton.is_skeleton_3d import Skeleton3D
from is_3d_skeleton.utils.get_cam_params import get_camera_calibration
from is_3d_skeleton.utils.plot_3d_pose import Plot3DPose

from is_broker import ISBroker


def load_conf(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return exp_config


def main(conf, show_2d=False, show_3d=False):
    calib_file_path = conf.PIPELINE_COMBINATION.CAMERA_CALIBRATION_PATH
    cameras = get_camera_calibration(calib_file_path)

    broker_conf = conf.IS_BROKER
    is_broker = ISBroker(broker_conf)

    is_skeleton_2d = Skeleton2D(conf)
    is_skeleton_3d = Skeleton3D(cameras, conf)

    if show_2d:
        plot_2d = Plot2DPose()
    if show_3d:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection="3d")
        plot_3d = Plot3DPose(ax, cameras)

    while True:
        init_time = time.time()

        frame_id, image_list = is_broker.get_images(period=conf.IS_BROKER.PERIOD)

        if len(image_list) == 0:
            continue

        print(f"FRAME_ID: {frame_id}  -  IMAGES: {len(image_list)}")
        obs2d_dict, result_pose, person_bbox = is_skeleton_2d.estimate(image_list)

        if any(result_pose):
            if show_2d:
                stacked_img = plot_2d.get_stacked_images(
                    cam_imgs=image_list,
                    human_poses=result_pose,
                    person_bbox=person_bbox,
                )
                cv2.imshow("2D POSES", stacked_img)

            pose_3d_time = time.time()

            obs, pts3d, person3d_ids = is_skeleton_3d.estimate(
                frame_id=frame_id, obs2d_dict=obs2d_dict, result_pose=result_pose
            )
            print(f"POSE 3D AND TRACKING TIME: {time.time() - pose_3d_time} s")

            is_broker.publish_annotations(obs)

            if show_3d:
                plot_3d.plot(person3d_ids, pts3d)
                fig.canvas.draw()

                # generate a image from a matplotlib figure
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
                view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow("3D", view_3d)

        diff_time = time.time() - init_time
        print(f"Exec. Time: {diff_time} s")

        # cv2.waitKey(1)


if __name__ == "__main__":
    cfg = load_conf("configs/service_configs_lab.yaml")
    main(cfg, show_2d=False, show_3d=False)
