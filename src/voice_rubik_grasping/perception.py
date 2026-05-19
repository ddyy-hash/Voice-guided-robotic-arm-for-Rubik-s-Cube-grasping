"""Vision utilities for Rubik's Cube detection and demo visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected target in image coordinates."""

    label: str
    bbox_xyxy: tuple[int, int, int, int]
    center_xy: tuple[int, int]
    confidence: float
    source: str


def detect_rubik_by_color(image_bgr: np.ndarray, label: str = "cube") -> Optional[Detection]:
    """Detect a Rubik-style cube by clustering multiple colored sticker regions."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation > 70) & (value > 90)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    components: list[tuple[float, tuple[int, int, int, int], tuple[float, float]]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(20.0, image_area * 0.00002) or area > image_area * 0.004:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / max(h, 1)
        if 0.35 <= ratio <= 2.8:
            components.append((area, (x, y, x + w, y + h), (x + w / 2.0, y + h / 2.0)))

    if not components:
        return None

    cluster_radius = max(70.0, min(image_bgr.shape[:2]) * 0.11)
    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None

    for _area, _bbox, center in components:
        nearby = [
            component
            for component in components
            if np.linalg.norm(np.array(component[2]) - np.array(center)) <= cluster_radius
        ]
        if len(nearby) < 3:
            continue

        xs1 = [component[1][0] for component in nearby]
        ys1 = [component[1][1] for component in nearby]
        xs2 = [component[1][2] for component in nearby]
        ys2 = [component[1][3] for component in nearby]
        bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
        total_area = sum(component[0] for component in nearby)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        square_bonus = 1.0 - min(abs(width / height - 1.0), 1.0)
        score = len(nearby) * 1000.0 + total_area + square_bonus * 250.0
        if score > best_score:
            best_score = score
            best_bbox = bbox

    if best_bbox is None:
        _area, best_bbox, _center = max(components, key=lambda item: item[0])

    x1, y1, x2, y2 = best_bbox
    padding = 8
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_bgr.shape[1], x2 + padding)
    y2 = min(image_bgr.shape[0], y2 + padding)
    return Detection(
        label=label,
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=((x1 + x2) // 2, (y1 + y2) // 2),
        confidence=0.65,
        source="color-fallback",
    )


class OpenVocabularyDetector:
    """Open-vocabulary detector with a Rubik-specific lightweight fallback."""

    def __init__(
        self,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        grounding_dino_weights: str | Path | None = None,
        efficient_sam_weights: str | Path | None = None,
    ) -> None:
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.grounding_dino_weights = Path(grounding_dino_weights) if grounding_dino_weights else None
        self.efficient_sam_weights = Path(efficient_sam_weights) if efficient_sam_weights else None
        self._gd_predict = None
        self._gd_transforms = None
        self._gd_model = None
        self._device = "cpu"

    def load_optional_models(self) -> bool:
        """Load GroundingDINO when its package and weight file are available."""
        if not self.grounding_dino_weights or not self.grounding_dino_weights.exists():
            return False
        try:
            import torch
            import groundingdino
            import groundingdino.datasets.transforms as transforms
            from groundingdino.util.inference import load_model, predict

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            config_path = (
                Path(groundingdino.__file__).parent
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            )
            self._gd_model = load_model(
                str(config_path), str(self.grounding_dino_weights), device=self._device
            )
            self._gd_predict = predict
            self._gd_transforms = transforms
            return True
        except Exception as exc:
            print(f"GroundingDINO is unavailable, using fallback detector: {exc}")
            return False

    def detect(self, image_bgr: np.ndarray, text_prompt: str) -> Optional[Detection]:
        prompt = text_prompt.strip() or "cube"
        if self._gd_model is not None:
            detected = self._detect_with_grounding_dino(image_bgr, prompt)
            if detected is not None:
                return detected
        return detect_rubik_by_color(image_bgr, prompt)

    def _detect_with_grounding_dino(
        self, image_bgr: np.ndarray, text_prompt: str
    ) -> Optional[Detection]:
        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        transform = self._gd_transforms.Compose(
            [
                self._gd_transforms.RandomResize([800], max_size=1333),
                self._gd_transforms.ToTensor(),
                self._gd_transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        image_tensor, _ = transform(pil_image, None)
        boxes, logits, phrases = self._gd_predict(
            model=self._gd_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self._device,
        )
        if len(boxes) == 0:
            return None

        best_idx = int(torch.argmax(logits).item())
        cx, cy, width, height = boxes[best_idx].cpu().numpy()
        img_h, img_w = image_bgr.shape[:2]
        x1 = int(max(0, (cx - width / 2.0) * img_w))
        y1 = int(max(0, (cy - height / 2.0) * img_h))
        x2 = int(min(img_w, (cx + width / 2.0) * img_w))
        y2 = int(min(img_h, (cy + height / 2.0) * img_h))
        return Detection(
            label=phrases[best_idx] if phrases else text_prompt,
            bbox_xyxy=(x1, y1, x2, y2),
            center_xy=((x1 + x2) // 2, (y1 + y2) // 2),
            confidence=float(logits[best_idx].item()),
            source="grounding-dino",
        )


def annotate_detection(image_bgr: np.ndarray, detection: Optional[Detection]) -> np.ndarray:
    """Draw a detector result for README evidence or debugging."""
    output = image_bgr.copy()
    if detection is None:
        cv2.putText(
            output,
            "target not detected",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return output

    x1, y1, x2, y2 = detection.bbox_xyxy
    cx, cy = detection.center_xy
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
    label = f"{detection.label} {detection.confidence:.2f} [{detection.source}]"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_x = min(max(12, x1), max(12, output.shape[1] - text_size[0] - 12))
    label_y = max(30, y1 - 12)
    cv2.putText(
        output,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return output
