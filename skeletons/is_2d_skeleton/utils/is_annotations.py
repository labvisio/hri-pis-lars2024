from is_msgs.image_pb2 import ObjectAnnotations, ObjectLabels, HumanKeypoints


class Skeleton2DAnnotations:

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

    def to_object_annotations(self, points_2d, im_width, im_height):
        obs = ObjectAnnotations()
        for points in points_2d:
            ob = obs.objects.add()
            for part_id, (x, y, score) in enumerate(points):
                part = ob.keypoints.add()
                part.id = self._to_sks_part[part_id]
                part.position.x = x
                part.position.y = y
                part.score = score
            ob.label = "human_skeleton"
            ob.id = ObjectLabels.Value('HUMAN_SKELETON')
        obs.resolution.width = im_width
        obs.resolution.height = im_height
        return obs
