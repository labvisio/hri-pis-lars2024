import cv2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import yaml
import os
from os import path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "is_2d_skeleton"))
sys.path.append(os.path.join(os.path.dirname(__file__), "is_3d_skeleton"))

from is_2d_skeleton.is_skeleton_2d import Skeleton2D
from is_2d_skeleton.utils.plot_2d_pose import Plot2DPose

from is_3d_skeleton.is_skeleton_3d import Skeleton3D
from is_3d_skeleton.utils.get_cam_params import get_camera_calibration
from is_3d_skeleton.utils.plot_3d_pose import Plot3DPose
from view_skeletons import SkeletonsViewer


# TODO:
#  - [ ] Criar um arquivo de configuração para o serviço 2D e 3D
#  - [ ] Colocar no main a iniciação do broker e a captura de imagens (no main_is.py)
#  - [ ] Enviar o object annotation para o broker (no main_is.py)


def load_conf(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return exp_config

def main(conf, show_2d=False, show_3d=False):
    calib_file_path = conf.PIPELINE_COMBINATION.CAMERA_CALIBRATION_PATH
    cameras = get_camera_calibration(calib_file_path)

    is_skeleton_2d = Skeleton2D(conf)
    is_skeleton_3d = Skeleton3D(cameras, conf)

    if show_2d:
        plot_2d = Plot2DPose()
    if show_3d:
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        plot_3d = Plot3DPose(ax, cameras)
        # plot_3d = SkeletonsViewer()

    # carregando imagens do dataset antigo da viros
    dataset_dir = "../dataset/p001g15"  # p001g10 -  p001g15

    img_paths = os.listdir(dataset_dir)
    cam_0 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c00s" in img_path])
    cam_1 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c01s" in img_path])
    cam_2 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c02s" in img_path])
    cam_3 = sorted([path.join(dataset_dir, img_path) for img_path in img_paths if "c03s" in img_path])

    # for frame_id in range(len(cam_0)):
    for frame_id in range(215, 375):  # 75, 375
        print(f"FRAME_ID: {frame_id}")

        image_list = [cv2.imread(cam_0[frame_id]), cv2.imread(cam_1[frame_id]), cv2.imread(cam_2[frame_id]),
                      cv2.imread(cam_3[frame_id])]
        init_time = time.time()

        obs2d_dict, result_pose, person_bbox = is_skeleton_2d.estimate(image_list)
        print(f"POSE 2D TIME: {1000 * (time.time() - init_time)} ms")

        if any(result_pose):
            if show_2d:
                stacked_img = plot_2d.get_stacked_images(cam_imgs=image_list, human_poses=result_pose,
                                                         person_bbox=person_bbox)
                cv2.imshow("2D POSES", stacked_img)

            pose_3d_time = time.time()

            obs, pts3d, person3d_ids = is_skeleton_3d.estimate(frame_id=frame_id, obs2d_dict=obs2d_dict, result_pose=result_pose)
            print(f"POSE 3D AND TRACKING TIME: {time.time() - pose_3d_time} s")

            print(f"Person 3D IDs: {person3d_ids}")

            if show_3d:
                plot_3d.plot(person3d_ids, pts3d)
                fig.canvas.draw()

                # generate a image from a matplotlib figure
                data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow("3D", view_3d)

                # plot_3d.plot(obs)


        diff_time = time.time() - init_time
        print(f"Exec. Time: {diff_time} s")

        cv2.waitKey(1)

    # plot_3d.get_poses_graph()
    # cv2.waitKey(0)


if __name__ == "__main__":
    # TODO: Isso pode ser passado como variável de ambiente
    cfg = load_conf('configs/service_configs_lab.yaml')
    main(cfg, show_2d=True, show_3d=True)
