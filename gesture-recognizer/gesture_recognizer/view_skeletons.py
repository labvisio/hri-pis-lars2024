import socket
import colorsys
from itertools import permutations
from typing import List

import cv2
from google.protobuf.json_format import ParseDict


from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_msgs.image_pb2 import ObjectAnnotations
from is_wire.core import Channel, Message, Subscription
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use("Agg")

from .recognizer import GestureRecognizer
from .position_control import PositionControl


class CustomChannel(Channel):
    def __init__(
        self,
        uri: str = "amqp://guest:guest@localhost:5672",
        exchange: str = "is",
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    def consume_all(self) -> List[Message]:
        messages = []
        message = super().consume()
        messages.append(message)
        while True:
            try:
                message = super().consume(timeout=0.0)
                messages.append(message)
            except socket.timeout:
                return messages


class App(object):

    def __init__(
        self,
        group_id: int = 0,
        broker_uri: str = "amqp://guest:guest@localhost:5672",
    ) -> None:
        self.channel = CustomChannel(uri=broker_uri, exchange="is")
        self.subscription = Subscription(channel=self.channel, name="GestureRecognizer")
        self.control = PositionControl(channel=self.channel, interval=15)
        self.subscription.subscribe(f"SkeletonsGrouper.{group_id}.Localization")

        self.fig = plt.figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot()

        self.colors = list(permutations([0, 255, 85, 170], 3))
        self.links = [
            (HKP.Value("LEFT_SHOULDER"), HKP.Value("RIGHT_SHOULDER")),
            (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_HIP")),
            (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_HIP")),
            (HKP.Value("LEFT_HIP"), HKP.Value("RIGHT_HIP")),
            (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_EAR")),
            (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_EAR")),
            (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_ELBOW")),
            (HKP.Value("LEFT_ELBOW"), HKP.Value("LEFT_WRIST")),
            # (HKP.Value("NECK"), HKP.Value("LEFT_HIP")),
            (HKP.Value("LEFT_HIP"), HKP.Value("LEFT_KNEE")),
            (HKP.Value("LEFT_KNEE"), HKP.Value("LEFT_ANKLE")),
            # (HKP.Value("NECK"), HKP.Value("RIGHT_SHOULDER")),
            (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_ELBOW")),
            (HKP.Value("RIGHT_ELBOW"), HKP.Value("RIGHT_WRIST")),
            # (HKP.Value("NECK"), HKP.Value("RIGHT_HIP")),
            (HKP.Value("RIGHT_HIP"), HKP.Value("RIGHT_KNEE")),
            (HKP.Value("RIGHT_KNEE"), HKP.Value("RIGHT_ANKLE")),
            (HKP.Value("NOSE"), HKP.Value("LEFT_EYE")),
            (HKP.Value("LEFT_EYE"), HKP.Value("LEFT_EAR")),
            (HKP.Value("NOSE"), HKP.Value("RIGHT_EYE")),
            (HKP.Value("RIGHT_EYE"), HKP.Value("RIGHT_EAR")),
        ]
        self.classifier = GestureRecognizer()

    def _id_to_rgb_color(self, id):
        hue = (id % 20) / 20
        saturation = 0.8
        luminance = 0.6
        r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]
        return r, g, b

    def render_skeletons_3d(self, skeletons):
        results, positions = self.classifier.predict(skeletons=skeletons)
        colors = {0: "red", 1: "blue", 2: "pink", 3: "green"}

        gestures = {}
        for key, value in results.items():
            if value not in gestures:
                gestures[value] = {"x": [], "y": []}
            gestures[value]["x"].append(positions[key][0])
            gestures[value]["y"].append(positions[key][1])

        for key, value in results.items():
            if value == 1:
                x = positions[key][0]
                y = positions[key][1]

                # print("Person: ", positions[key][0], positions[key][1])
                if x > 0 and y > 0:
                    x = positions[key][0] - 0.3
                    y = positions[key][1] - 0.3
                elif x > 0 and y < 0:
                    x = positions[key][0] - 0.3
                    y = positions[key][1] + 0.3
                elif x < 0 and y > 0:
                    x = positions[key][0] + 0.3
                    y = positions[key][1] - 0.3
                elif x < 0 and y < 0:
                    x = positions[key][0] + 0.3
                    y = positions[key][1] + 0.3
                # print("Robot: ", x, y)
                self.control.sent_to(x, y)
                break
            elif value == 2:
                self.control.sent_to(x=1.53, y=-6.52)
                break
            elif value == 3:
                self.control.sent_to(x=0, y=0)
                break

        for color in colors:
            if color in gestures:
                self.ax.scatter(
                    gestures[color]["x"],
                    gestures[color]["y"],
                    linewidth=20,
                    color=colors[color],
                )

    def run(self) -> None:
        plt.ioff()
        while True:
            messages = self.channel.consume_all()
            for message in messages:
                objs = message.unpack(ObjectAnnotations)
                self.ax.clear()
                self.ax.set_xlim(-4, 4)
                self.ax.set_xticks(np.arange(-4, 4, 1))
                self.ax.set_ylim(-4, 4)
                self.ax.set_yticks(np.arange(-4, 4, 1))
                self.ax.set_xlabel("X", labelpad=20)
                self.ax.set_ylabel("Y", labelpad=10)

                self.render_skeletons_3d(objs)

                self.fig.canvas.draw()

                data = np.fromstring(
                    self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep=""
                )
                view_3d = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow("", view_3d)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    return


if __name__ == "__main__":
    app = App(broker_uri="amqp://guest:guest@10.20.5.2:30000")
    app.run()
