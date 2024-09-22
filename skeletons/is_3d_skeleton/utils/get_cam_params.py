from typing import Any

import torch
import os
from os import path
import numpy as np
import numpy.typing as npt

from google.protobuf.json_format import Parse
from is_msgs.camera_pb2 import CameraCalibration
from is_msgs.common_pb2 import Tensor, DataType


class Camera(object):
    def __init__(self, cid, P, K, RT, F):
        self.cid = cid
        self.P = P
        self.K = K
        self.RT = RT
        self.RK_INV = np.linalg.inv(self.RT[:, :3]) @ np.linalg.inv(self.K)
        self.F = F
        RT_inv = np.linalg.inv(np.vstack([self.RT, [0, 0, 0, 1]]))
        self.position = RT_inv[:3, 3]

    def undistort(self, im):
        """ undistorts the image
		:param im: {h x w x c}
		:return:
		"""
        return im

    def undistort_points(self, points2d):
        """
		:param points2d: [ (x,y,w), ...]
		:return:
		"""
        return points2d

    def projectPoints_undist(self, points3d):
        """
			projects 3d points into 2d ones with
			no distortion
		:param points3d: {n x 3}
		:return:
		"""

        points2d = np.zeros((len(points3d), 2))
        for i, (x, y, z) in enumerate(points3d):
            p3d = np.array([x, y, z, 1])
            a, b, c = self.P @ p3d
            # assert c != 0 , print(self.P, p3d,self.P @ p3d)
            c = 10e-6 if c == 0 else c
            points2d[i, 1] = a / c if a is not None else None
            points2d[i, 0] = b / c if b is not None else None
        return points2d

    def projectPoints(self, points3d):
        """
			projects 3d points into 2d with
			distortion being considered
		:param points3d: {n x 3}
		:param withmask: {boolean} if True return mask that tells if a point is in the view or not
		:return:
		"""
        pts2d = self.projectPoints_undist(points3d)
        return pts2d

    def projectPoints_parallel(self, points3d):
        points3d = np.concatenate([points3d, np.ones((points3d.shape[0], points3d.shape[1], 1))], axis=2)
        homo_reproj = np.transpose(self.P @ points3d.reshape(-1, 4).T)
        reproj = homo_reproj[:, :2] / homo_reproj[:, 2].reshape(-1, 1)
        reproj = np.flip(reproj, axis=1)
        reproj = reproj.reshape(-1, 17, 2)

        return reproj

def get_camera_parameters(camera_parameter):
    P = camera_parameter['P'].astype(np.float32)
    K = camera_parameter['K'].astype(np.float32)
    RT = camera_parameter['RT'].astype(np.float32)
    skew_op = lambda x: torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: torch.inverse(K_0).t() @ (
            R_0 @ R_1.t()) @ K_1.t() @ skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))
    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1, RT_1[:, :3],
                                                                    RT_1[:, 3])
    camera_num = len(P)
    F = torch.zeros(camera_num, camera_num, 3, 3)  # NxNx3x3 matrix

    # Matriz fundamental que relaciona cada par de cameras
    for x in range(camera_num):
        for y in range(camera_num):
            F[x, y] += fundamental_RT_op(torch.tensor(K[x]), torch.tensor(RT[x]), torch.tensor(K[y]),
                                         torch.tensor(RT[y]))
            if F[x, y].sum() == 0:
                F[x, y] += 1e-12  # to avoid nan
    F = F.numpy()
    cameras = []
    for j in range(camera_num):
        cameras.append(Camera(j, P[j], K[j], RT[j], F[j]))
    return cameras

def load(filepath: str) -> CameraCalibration:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                calib = Parse(f.read(), CameraCalibration())
                # print(f"CameraCalibration: \n{calib}")
                return calib
            except Exception as ex:
                print(f"Unable to load options from '{filepath}'. \n{ex}")
    except Exception:
        print(f"Unable to open file '{filepath}'")

def tensor2array(tensor: Tensor) -> npt.NDArray[Any]:
    if len(tensor.shape.dims) != 2 or tensor.shape.dims[0].name != "rows":
        return np.array([])
    shape = (tensor.shape.dims[0].size, tensor.shape.dims[1].size)
    if tensor.type == DataType.Value("INT32_TYPE"):
        return np.array(tensor.ints32, dtype=np.int32, copy=False).reshape(shape)
    if tensor.type == DataType.Value("INT64_TYPE"):
        return np.array(tensor.ints64, dtype=np.int64, copy=False).reshape(shape)
    if tensor.type == DataType.Value("FLOAT_TYPE"):
        return np.array(tensor.floats, dtype=np.float32, copy=False).reshape(shape)
    if tensor.type == DataType.Value("DOUBLE_TYPE"):
        return np.array(tensor.doubles, dtype=np.float64, copy=False).reshape(shape)
    return np.array([])

def get_camera_calibration(cam_calib_dir):
    calib_files = sorted([path.join(cam_calib_dir, calib) for calib in os.listdir(cam_calib_dir) if path.isfile(path.join(cam_calib_dir, calib))])
    n_cams = len(calib_files)

    K = np.zeros((n_cams, 3, 3), dtype=np.float64)         # transforma para pixel
    RT = np.zeros((n_cams, 3, 4), dtype=np.float64)        # posição e orientação da câmera
    P = np.zeros((n_cams, 3, 4), dtype=np.float64)         # KRT
    F = np.zeros((n_cams, n_cams, 3, 3), dtype=np.float64)  # Matriz fundamental que relaciona cada par de cameras

    for i in range(n_cams):
        calibration = load(calib_files[i])
        intrinsic = tensor2array(calibration.intrinsic)
        distortion = tensor2array(calibration.distortion)
        ext = calibration.extrinsic[0]  # ext do mundo para camera
        # extrinsic = np.linalg.inv(tensor2array(ext.tf))
        extrinsic = tensor2array(ext.tf)

        K[i] = intrinsic
        RT[i] = extrinsic[0:3, :]
        P[i] = K[i] @ RT[i]

    skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
    fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1, RT_1[:, :3], RT_1[:, 3])

    for x in range(n_cams):
        for y in range(n_cams):
            F[x, y] += fundamental_RT_op(K[x], RT[x], K[y], RT[y])
            if F[x, y].sum() == 0:
                F[x, y] += 1e-12  # to avoid nan

    cameras = []
    for j in range(n_cams):
        cameras.append(Camera(j, P[j], K[j], RT[j], F[j]))
    return cameras
