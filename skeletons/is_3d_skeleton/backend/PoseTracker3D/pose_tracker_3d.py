import numpy as np
from is_msgs.image_pb2 import ObjectAnnotations, ObjectLabels, HumanKeypoints

from is_3d_skeleton.utils.is_annotations import Skeleton3DAnnotations


class PoseTracker3D:

    def __init__(self, tracker, cameras, build_3d='SVD'):
        self._tracker = tracker
        self._cameras = cameras
        self._build_3d = build_3d
        self._poses = []
        self.sks_annotations = Skeleton3DAnnotations()
        self._to_coco_idx = {
            HumanKeypoints.Value('NOSE'): 0,
            HumanKeypoints.Value('LEFT_EYE'): 1,
            HumanKeypoints.Value('RIGHT_EYE'): 2,
            HumanKeypoints.Value('LEFT_EAR'): 3,
            HumanKeypoints.Value('RIGHT_EAR'): 4,
            HumanKeypoints.Value('LEFT_SHOULDER'): 5,
            HumanKeypoints.Value('RIGHT_SHOULDER'): 6,
            HumanKeypoints.Value('LEFT_ELBOW'): 7,
            HumanKeypoints.Value('RIGHT_ELBOW'): 8,
            HumanKeypoints.Value('LEFT_WRIST'): 9,
            HumanKeypoints.Value('RIGHT_WRIST'): 10,
            HumanKeypoints.Value('LEFT_HIP'): 11,
            HumanKeypoints.Value('RIGHT_HIP'): 12,
            HumanKeypoints.Value('LEFT_KNEE'): 13,
            HumanKeypoints.Value('RIGHT_KNEE'): 14,
            HumanKeypoints.Value('LEFT_ANKLE'): 15,
            HumanKeypoints.Value('RIGHT_ANKLE'): 16,
        }

    def _to_tracking_format(self, obs2d_dict):
        poses = []
        for c_id, obs2d in obs2d_dict.items():
            n_persons = len(obs2d.objects)
            pts = np.zeros((n_persons, 17, 3), dtype=np.float32)
            for i, skeleton in enumerate(obs2d.objects):
                for part in skeleton.keypoints:
                    pts[i, self._to_coco_idx[part.id], 0] = part.position.y
                    pts[i, self._to_coco_idx[part.id], 1] = part.position.x
                    pts[i, self._to_coco_idx[part.id], 2] = part.score
            poses.append(pts)
        return poses

    def __call__(self, frame_id, obs2d_dict, result_pose):
        self._poses = self._to_tracking_format(obs2d_dict)
        self._tracker.tracking(frame_id, self._cameras, self._poses, self._build_3d)
        self._poses.clear()

        pts3d = []
        person3d_ids = []
        for track in self._tracker.tracks:
            if track.time_since_update > 0 or not track.is_confirmed():
                continue

            poses2d, pose3d, joints_views = track.poses2d, track.poses3d[-1]['pose3d'], track.poses3d[-1][
                'joints_views']
            pts3d.append(np.transpose(pose3d))
            person3d_ids.append(track.track_id)

        pts3d = np.array(pts3d)
        person3d_ids = np.array(person3d_ids)

        obs = self.sks_annotations.to_object_annotations(points_3d=pts3d,
                                                         person3d_ids=person3d_ids)

        return obs, pts3d, person3d_ids
