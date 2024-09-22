from is_msgs.image_pb2 import Image, ObjectAnnotations, ObjectLabels, HumanKeypoints


class Skeleton3DAnnotations:

    def __init__(self):
        self._to_sks_part = {
            0: HumanKeypoints.Value('NOSE'),
            1: HumanKeypoints.Value('LEFT_EYE'),
            2: HumanKeypoints.Value('RIGHT_EYE'),
            3: HumanKeypoints.Value('LEFT_EAR'),
            4: HumanKeypoints.Value('RIGHT_EAR'),
            5: HumanKeypoints.Value('LEFT_SHOULDER'),
            6: HumanKeypoints.Value('RIGHT_SHOULDER'),
            7: HumanKeypoints.Value('LEFT_ELBOW'),
            8: HumanKeypoints.Value('RIGHT_ELBOW'),
            9: HumanKeypoints.Value('LEFT_WRIST'),
            10: HumanKeypoints.Value('RIGHT_WRIST'),
            11: HumanKeypoints.Value('LEFT_HIP'),
            12: HumanKeypoints.Value('RIGHT_HIP'),
            13: HumanKeypoints.Value('LEFT_KNEE'),
            14: HumanKeypoints.Value('RIGHT_KNEE'),
            15: HumanKeypoints.Value('LEFT_ANKLE'),
            16: HumanKeypoints.Value('RIGHT_ANKLE'),
        }

    def to_object_annotations(self, points_3d, person3d_ids):
        obs = ObjectAnnotations()
        for person_id, points in zip(person3d_ids, points_3d):
            ob = obs.objects.add()
            for part_id, (x, y, z) in enumerate(points.T):
                part = ob.keypoints.add()
                part.id = self._to_sks_part[part_id]
                part.position.x = x
                part.position.y = y
                part.position.z = z
            ob.label = "3d_human_skeleton"
            ob.id = person_id
        return obs
