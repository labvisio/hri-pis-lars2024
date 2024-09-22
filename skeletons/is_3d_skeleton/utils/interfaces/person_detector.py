from abc import ABC, abstractmethod


class PersonDetectorInterface(ABC):

    @abstractmethod
    def __call__(self, image_list):
        # Should return an array of dicts with the score and bbox
        raise NotImplementedError("Subclasses should implement this!")
