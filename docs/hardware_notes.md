# Hardware Notes

The hardware demo was organized around:

- Piper robotic arm with a gripper.
- Eye-in-hand RGB/depth camera.
- Hand-eye calibration from the camera frame to the end-effector frame.
- Chinese voice command input mapped to detector prompts.
- A Rubik's Cube as the target object.

The public repository keeps motion disabled by default. Use dry-run output first, inspect the target pose, and only then enable `--execute` on a controlled workspace.

Before hardware execution:

1. Confirm the CAN interface name in `config/grasp_profile.yaml`.
2. Replace `config/T_ee_cam_example.yaml` with the calibrated `T_tool_camera` transform.
3. Confirm camera intrinsics and depth units.
4. Confirm the cube is reachable and the gripper aperture is large enough.
5. Keep emergency stop access available.
