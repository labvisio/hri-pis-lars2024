import numpy as np
from ros_pb2 import ROSMessage
from google.protobuf.struct_pb2 import Struct


def get_quaternion_from_euler(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def get_ros_message(x, y, z=0, yaw=0, frame_id="Map"):
    qx, qy, qz, qw = get_quaternion_from_euler(0, 0, yaw)
    ros_message = ROSMessage()
    ros_message.type = "geometry_msgs/PoseWithCovarianceStamped"
    PoseWithCovariance_dict = {
        "header": {"frame_id": frame_id},
        "pose": {
            "pose": {
                "position": {"y": y, "z": z, "x": x},
                "orientation": {
                    "z": qz,
                    "w": qw,
                },
            },
        },
    }

    PoseWithCovarianceStamped = Struct()
    PoseWithCovarianceStamped.update(PoseWithCovariance_dict)

    ros_message = ROSMessage(content=PoseWithCovarianceStamped)
    ros_message.type = "geometry_msgs/PoseWithCovarianceStamped"
    return ros_message
