from enum import Enum
from statistics import mode
from typing import Dict, Tuple, List

from is_msgs.image_pb2 import HumanKeypoints as HKP, ObjectAnnotations


class Gesture(Enum):
    LEFT_UP = 1
    RIGHT_UP = 2
    BOTH_UP = 3
    NONE_UP = 4


class GestureRecognizer:

    def __init__(self):
        self.window = 5
        self.gestures: Dict[int, List[int]] = {}

    def predict_parts(
        self,
        parts: Dict[int, Tuple[float, float, float]],
    ) -> int:
        if parts[HKP.Value("NOSE")][2] > parts[HKP.Value("LEFT_WRIST")][2]:
            if parts[HKP.Value("NOSE")][2] > parts[HKP.Value("RIGHT_WRIST")][2]:
                return 0
            else:
                return 1
        else:
            if parts[HKP.Value("NOSE")][2] > parts[HKP.Value("RIGHT_WRIST")][2]:
                return 2
            else:
                return 3

    def predict(
        self,
        skeletons: ObjectAnnotations,
    ) -> Dict[int, int]:
        positions = {}
        for skeleton in skeletons.objects:
            parts: Dict[int, float] = {}
            x_mean = 0
            y_mean = 0

            for part in skeleton.keypoints:
                parts[part.id] = (part.position.x, part.position.y, part.position.z)
                x_mean += part.position.x
                y_mean += part.position.y

            x_mean /= len(parts)
            y_mean /= len(parts)
            positions[skeleton.id] = (x_mean, y_mean)
            gesture = self.predict_parts(parts)

            if skeleton.id not in self.gestures:
                self.gestures[skeleton.id] = [gesture]
                continue
            self.gestures[skeleton.id].append(gesture)

            while len(self.gestures[skeleton.id]) > self.window:
                self.gestures[skeleton.id].pop(0)

        result = {}
        for key, value in self.gestures.items():
            if key in positions:
                result[key] = mode(value)

        return result, positions
