import numpy as np

from voice_rubik_grasping.robot import CameraIntrinsics, compute_grasp_plan


def test_compute_grasp_plan_returns_finite_pose():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    plan = compute_grasp_plan(
        center_xy=(320, 240),
        depth_m=0.35,
        intrinsics=intrinsics,
        transform_base_tool=np.eye(4),
        transform_tool_camera=np.eye(4),
        approach_distance_m=0.10,
    )

    assert plan.approach_position_m.shape == (3,)
    assert plan.target_rotation.shape == (3, 3)
    assert np.isfinite(plan.approach_position_m).all()
    assert np.isclose(np.linalg.det(plan.target_rotation), 1.0)
