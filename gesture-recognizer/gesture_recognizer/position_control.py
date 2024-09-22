import time
from typing import Literal

from is_wire.core import Channel, Message
from .is_to_ros import get_ros_message


class PositionControl:

    def __init__(self, channel: Channel, interval: float = 10):
        self._channel = channel

        self._interval = interval
        self._now = time.time()
        self._last = time.time()

    def sent_to(self, x: float, y: float):
        self._now = time.time()
        if (self._now - self._last) > self._interval:
            ros_message = get_ros_message(x, y)
            message = Message(content=ros_message)
            self._channel.publish(message, topic="ros.send_goal_nav2")
            self._last = time.time()
