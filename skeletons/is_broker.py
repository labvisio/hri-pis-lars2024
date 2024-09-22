from datetime import datetime
import socket
import re
from typing import List, Dict
from easydict import EasyDict
import time
import cv2
import numpy as np
import numpy.typing as npt

from is_msgs.image_pb2 import Image

from is_wire.core import Channel, Message, Subscription
from is_wire.core.utils import now


class CustomChannel(Channel):
    def __init__(self, uri, exchange) -> None:
        super().__init__(uri=uri, exchange=exchange)
        self._deadline = now()
        self._running = False

    def consume_for(self, period: float) -> List[Message]:
        if not self._running:
            self._deadline = now()
            self._running = True
        self._deadline = self._deadline + period
        messages = []
        while True:
            try:
                message = self.consume_until(deadline=self._deadline)
                messages.append(message)
            except socket.timeout:
                break
        return messages

    def consume_until(self, deadline: float) -> Message:
        timeout = max([deadline - now(), 0.0])
        return self.consume(timeout=timeout)


class ISBroker(object):
    def __init__(self, configuration: EasyDict) -> None:
        self.channel = CustomChannel(uri=configuration["URI"], exchange="is")
        self.subscription = Subscription(channel=self.channel, name="is-skeletons")
        self._pub_topic = configuration["TOPIC_TO_PUBLISH"]
        self._frame_id = 0

        cameras = configuration["CAMERAS"]
        for camera in cameras:
            self.subscription.subscribe(f"CameraGateway.{camera}.Frame")

    @staticmethod
    def to_np(image: Image) -> npt.NDArray[np.uint8]:
        buffer = np.frombuffer(image.data, dtype=np.uint8)
        output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
        return output

    @staticmethod
    def get_topic_id(topic: str) -> str:
        re_topic = re.compile(r"CameraGateway.(\d+).Frame")
        result = re_topic.match(topic)
        if result:
            return result.group(1)

    def get_images(self, period=0.1):
        messages = self.channel.consume_for(period=period)
        images_dict: Dict[str, npt.NDArray[np.uint8]] = {}

        for message in messages:
            camera_id = self.get_topic_id(message.topic)
            image = self.to_np(message.unpack(Image))
            images_dict[camera_id] = image

        print(images_dict.keys())
        cam_ids = sorted(images_dict.keys(), key=int)
        images = [images_dict[cam_id] for cam_id in cam_ids]

        self._frame_id += 1
        return self._frame_id, images

    def publish_annotations(self, annotations):
        message = Message()
        message.topic = "SkeletonsGrouper.0.Localization"
        message.pack(annotations)
        self.channel.publish(message)
