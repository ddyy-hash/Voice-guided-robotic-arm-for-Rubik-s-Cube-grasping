"""Robot-side planning and optional Piper SDK execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import load_yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics used for depth projection."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class GraspPlan:
    """End-effector target pose for one grasp attempt."""

    approach_position_m: np.ndarray
    target_rotation: np.ndarray
    object_position_m: np.ndarray


def load_handeye_matrix(path: str | Path) -> np.ndarray:
    """Load a 4x4 tool-to-camera transform from the calibration YAML file."""
    data = load_yaml(path)
    transform = data.get("transformation", {})
    if "matrix_4x4" in transform:
        return np.array(transform["matrix_4x4"], dtype=float)
    if "transform_matrix" in transform:
        return np.array(transform["transform_matrix"], dtype=float)
    if "T_ee_cam" in data and "matrix" in data["T_ee_cam"]:
        return np.array(data["T_ee_cam"]["matrix"], dtype=float)
    raise ValueError(f"Unsupported hand-eye calibration format: {path}")


def pixel_to_camera(point_xy: tuple[int, int], depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Project one image pixel and depth value into camera coordinates."""
    x, y = point_xy
    return np.array(
        [
            (x - intrinsics.cx) * depth_m / intrinsics.fx,
            (y - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=float,
    )


def compute_grasp_plan(
    center_xy: tuple[int, int],
    depth_m: float,
    intrinsics: CameraIntrinsics,
    transform_base_tool: np.ndarray,
    transform_tool_camera: np.ndarray,
    approach_distance_m: float,
) -> GraspPlan:
    """Compute a pose that points the gripper toward the detected object center."""
    object_camera = pixel_to_camera(center_xy, depth_m, intrinsics)
    transform_camera_object = np.eye(4)
    transform_camera_object[:3, 3] = object_camera
    transform_camera_object[:3, 0] = [1.0, 0.0, 0.0]
    transform_camera_object[:3, 1] = [0.0, -1.0, 0.0]
    transform_camera_object[:3, 2] = [0.0, 0.0, -1.0]

    transform_base_object = (
        transform_base_tool @ transform_tool_camera @ transform_camera_object
    )
    object_position = transform_base_object[:3, 3]
    object_z_axis = transform_base_object[:3, :3][:, 2]
    approach_position = object_position + object_z_axis * approach_distance_m

    z_axis = object_position - approach_position
    z_axis = z_axis / np.linalg.norm(z_axis)
    current_y = transform_base_tool[:3, 1]
    projected_y = current_y - np.dot(current_y, z_axis) * z_axis
    if np.linalg.norm(projected_y) < 1e-6:
        current_x = transform_base_tool[:3, 0]
        x_axis = current_x - np.dot(current_x, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
    else:
        y_axis = projected_y / np.linalg.norm(projected_y)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return GraspPlan(approach_position, rotation, object_position)


class PiperArm:
    """Piper SDK adapter with dry-run as the default operating mode."""

    def __init__(self, can_interface: str, execute: bool = False, speed: int = 30) -> None:
        self.can_interface = can_interface
        self.execute = execute
        self.speed = speed
        self.piper = None

    def connect(self) -> None:
        if not self.execute:
            print("Dry-run mode: Piper SDK connection is skipped.")
            return

        from piper_sdk import C_PiperInterface, LogLevel

        self.piper = C_PiperInterface(
            can_name=self.can_interface,
            judge_flag=False,
            can_auto_init=True,
            dh_is_offset=1,
            logger_level=LogLevel.INFO,
        )
        self.piper.ConnectPort()
        time.sleep(0.5)
        for _ in range(10):
            self.piper.EnableArm(7)
            time.sleep(0.2)

    def current_tool_pose(self) -> np.ndarray:
        if not self.execute or self.piper is None:
            return np.eye(4)

        pose = self.piper.GetArmEndPoseMsgs().end_pose
        position = np.array([pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=float) / 1e6
        euler_deg = np.array([pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=float) / 1000.0
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        transform[:3, 3] = position
        return transform

    def move_to_pose(self, position_m: np.ndarray, rotation: np.ndarray) -> None:
        if not self.execute or self.piper is None:
            print(
                "Dry-run target pose: "
                f"x={position_m[0]:.3f}m y={position_m[1]:.3f}m z={position_m[2]:.3f}m"
            )
            return

        euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        self.piper.MotionCtrl_2(0x01, 0x00, self.speed, 0x00)
        self.piper.EndPoseCtrl(
            int(position_m[0] * 1e6),
            int(position_m[1] * 1e6),
            int(position_m[2] * 1e6),
            int(euler[0] * 1000),
            int(euler[1] * 1000),
            int(euler[2] * 1000),
        )

    def control_gripper(self, open_mm: float, force: float) -> None:
        if not self.execute or self.piper is None:
            print(f"Dry-run gripper command: open={open_mm:.1f}mm force={force:.2f}")
            return
        self.piper.GripperCtrl(int(open_mm * 1000), int(force * 1000), 0x01, 0)

    def close(self) -> None:
        if self.piper is None:
            return
        try:
            self.piper.DisableArm(7)
            self.piper.DisconnectPort()
        finally:
            self.piper = None
