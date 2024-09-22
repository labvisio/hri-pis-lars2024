import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt


class Plot2DPose:

    def __init__(self, threshold=0.5, resize=640):
        self._threshold = threshold
        self._resize = resize
        self._skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                          (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                          (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
        self._palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                         (255, 153, 255), (153, 204, 255), (255, 102, 255),
                         (255, 51, 255), (102, 178, 255),
                         (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
                         (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                         (0, 0, 255), (255, 0, 0), (255, 255, 255)]
        self._link_color = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
        self._point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

    def _draw_skeleton(self, frame, keypoints, scores, bboxes, plot_bbox=False):
        scale = self._resize / max(frame.shape[0], frame.shape[1])
        keypoints = (keypoints[..., :2] * scale).astype(int)
        bboxes = (bboxes[...] * scale).astype(int)

        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        for kpts, score, bbox in zip(keypoints, scores, bboxes):
            if plot_bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            show = [0] * len(kpts)
            for (u, v), color in zip(self._skeleton, self._link_color):
                if score[u] > self._threshold and score[v] > self._threshold:
                    cv2.line(img, tuple(kpts[u]), tuple(kpts[v]), self._palette[color], 1, cv2.LINE_AA)
                    show[u] = show[v] = 1
            for kpt, show, color in zip(kpts, show, self._point_color):
                if show:
                    cv2.circle(img, kpt, 1, self._palette[color], 2, cv2.LINE_AA)
        return img

    def get_stacked_images(self, cam_imgs, human_poses, person_bbox, plot_bbox=True):
        skeleton_imgs = []
        for img, poses, bboxes in zip(cam_imgs, human_poses, person_bbox):
            bboxes = np.array(bboxes)
            if bboxes.size == 0:
                skeleton_imgs.append(img)
                continue
            bboxes = np.array([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2]+bboxes[:, 0], bboxes[:, 3]+bboxes[:, 1]]).T
            poses = np.array(poses)

            n_poses = poses.shape[0]
            kpts = poses[:, :, 0:2]
            scores = poses[:, :, 2]
            scores = scores.reshape((n_poses, 17))

            skeleton_img = self._draw_skeleton(img, kpts, scores, bboxes, plot_bbox)
            skeleton_imgs.append(skeleton_img)

        skeleton_imgs = [imutils.resize(img, width=300) for img in skeleton_imgs]
        stacked_img = np.hstack(skeleton_imgs)
        return stacked_img
