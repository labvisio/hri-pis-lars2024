from abc import ABC, abstractmethod


class PoseDetectorInterface(ABC):

    @abstractmethod
    def __call__(self, bbox_list):
        # Should return an array of dicts with the person detect score, person bbox, keypoints and keypoints score
        raise NotImplementedError("Subclasses should implement this!")
