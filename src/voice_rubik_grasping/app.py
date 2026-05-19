"""Command-line entry point for voice-guided Rubik's Cube grasping."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_ALIASES, DEFAULT_PROFILE, load_yaml, resolve_repo_path
from .perception import OpenVocabularyDetector, annotate_detection
from .robot import CameraIntrinsics, PiperArm, compute_grasp_plan, load_handeye_matrix
from .voice import VoiceCommandParser, WhisperVoiceRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Voice-guided Piper robotic arm demo for Rubik's Cube grasping."
    )
    parser.add_argument("--config", default=str(DEFAULT_PROFILE))
    parser.add_argument("--aliases", default=str(DEFAULT_ALIASES))
    parser.add_argument("--target", default="cube")
    parser.add_argument("--voice", action="store_true", help="Listen for one voice command")
    parser.add_argument("--image", help="Use a still image for offline detection")
    parser.add_argument("--video", help="Use the first readable frame from a video")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--depth-m", type=float, default=0.35)
    parser.add_argument("--output", default="")
    parser.add_argument("--execute", action="store_true", help="Enable Piper SDK motion")
    parser.add_argument("--load-models", action="store_true", help="Load GroundingDINO when available")
    return parser


def load_input_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        return image

    if args.video:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        try:
            if args.frame_index > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {args.frame_index}")
            return frame
        finally:
            capture.release()

    if args.camera_index is not None:
        capture = cv2.VideoCapture(args.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index: {args.camera_index}")
        try:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read camera frame")
            return frame
        finally:
            capture.release()

    raise RuntimeError("Provide --image, --video, or --camera-index")


def choose_target(args: argparse.Namespace) -> str:
    parser = VoiceCommandParser(args.aliases)
    if not args.voice:
        return args.target

    recognizer = WhisperVoiceRecognizer()
    command = recognizer.listen_once(parser)
    if command.is_exit:
        raise SystemExit("Exit command received.")
    if command.target is None:
        raise RuntimeError("No target object was recognized from voice input.")
    print(f"Voice command: '{command.raw_text}' -> target '{command.target}'")
    return command.target


def main() -> None:
    args = build_parser().parse_args()
    profile = load_yaml(args.config)
    target = choose_target(args)
    frame = load_input_frame(args)

    vision_config = profile.get("vision", {})
    detector = OpenVocabularyDetector(
        box_threshold=float(vision_config.get("box_threshold", 0.35)),
        text_threshold=float(vision_config.get("text_threshold", 0.25)),
        grounding_dino_weights=resolve_repo_path(
            vision_config.get("grounding_dino_weights", "weights/groundingdino_swint_ogc.pth")
        ),
        efficient_sam_weights=resolve_repo_path(
            vision_config.get("efficient_sam_weights", "weights/efficient_sam_vits.pt")
        ),
    )
    if args.load_models:
        detector.load_optional_models()

    detection = detector.detect(frame, target)
    annotated = annotate_detection(frame, detection)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)

    if detection is None:
        raise RuntimeError(f"Target '{target}' was not detected.")

    camera_config = profile.get("camera", {})
    intrinsics_config = camera_config.get("intrinsics", {})
    intrinsics = CameraIntrinsics(
        fx=float(intrinsics_config.get("fx", 487.85)),
        fy=float(intrinsics_config.get("fy", 487.85)),
        cx=float(intrinsics_config.get("cx", 318.40)),
        cy=float(intrinsics_config.get("cy", 216.84)),
    )

    robot_config = profile.get("robot", {})
    grasp_config = profile.get("grasp", {})
    arm = PiperArm(
        can_interface=robot_config.get("can_interface", "can0"),
        execute=args.execute,
        speed=int(robot_config.get("speed", 30)),
    )
    handeye = load_handeye_matrix(
        resolve_repo_path(robot_config.get("handeye_calibration", "config/T_ee_cam_example.yaml"))
    )

    try:
        arm.connect()
        plan = compute_grasp_plan(
            center_xy=detection.center_xy,
            depth_m=args.depth_m,
            intrinsics=intrinsics,
            transform_base_tool=arm.current_tool_pose(),
            transform_tool_camera=handeye,
            approach_distance_m=float(grasp_config.get("approach_distance_m", 0.10)),
        )
        print(
            f"Detected {detection.label} at pixel {detection.center_xy}; "
            f"planned object position {np.round(plan.object_position_m, 4).tolist()} m"
        )
        arm.control_gripper(float(grasp_config.get("gripper_open_mm", 80.0)), 0.1)
        arm.move_to_pose(plan.approach_position_m, plan.target_rotation)
        if args.execute:
            arm.control_gripper(0.0, float(grasp_config.get("gripper_force", 0.5)))
    finally:
        arm.close()


if __name__ == "__main__":
    main()
