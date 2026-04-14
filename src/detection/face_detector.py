from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np

from src.utils.types import BoundingBox, DetectionResult

try:
    import mediapipe as mp  # type: ignore
    try:
        _FACE_MESH_API = mp.solutions.face_mesh  # type: ignore[attr-defined]
        MEDIAPIPE_AVAILABLE = True
    except AttributeError:
        _FACE_MESH_API = None
        MEDIAPIPE_AVAILABLE = False
except ImportError:  # pragma: no cover
    mp = None
    _FACE_MESH_API = None
    MEDIAPIPE_AVAILABLE = False

LOGGER = logging.getLogger("afs.face_detector")
_MEDIAPIPE_WARNING_EMITTED = False


def _mediapipe_version() -> str:
    try:
        return version("mediapipe")
    except PackageNotFoundError:
        return "not-installed"


def check_mediapipe() -> bool:
    """Return whether MediaPipe FaceMesh API surface is available in this environment."""
    return MEDIAPIPE_AVAILABLE


@dataclass(slots=True)
class FaceDetectorConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


class FaceDetector:
    def __init__(self, config: FaceDetectorConfig) -> None:
        self._config = config
        self._mesh = None
        if MEDIAPIPE_AVAILABLE and _FACE_MESH_API is not None:
            try:
                self._mesh = _FACE_MESH_API.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=3,
                    refine_landmarks=True,
                    min_detection_confidence=config.min_detection_confidence,
                    min_tracking_confidence=config.min_tracking_confidence,
                )
            except Exception as exc:
                LOGGER.warning(
                    "MediaPipe FaceMesh init failed (version=%s): %s",
                    _mediapipe_version(),
                    exc,
                )
                self._mesh = None
        else:
            LOGGER.warning(
                "MediaPipe FaceMesh API unavailable (version=%s); detector will use full-frame fallback",
                _mediapipe_version(),
            )

    def _fallback_full_frame(self, frame: np.ndarray, reason: str) -> DetectionResult:
        global _MEDIAPIPE_WARNING_EMITTED
        if not _MEDIAPIPE_WARNING_EMITTED:
            LOGGER.warning(
                "Using full-frame detector fallback (%s). mediapipe_version=%s",
                reason,
                _mediapipe_version(),
            )
            _MEDIAPIPE_WARNING_EMITTED = True

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return DetectionResult(boxes=[], landmarks=[], error="invalid_frame")

        return DetectionResult(
            boxes=[BoundingBox(0, 0, w - 1, h - 1)],
            landmarks=[],
            error="mediapipe_unavailable_full_frame_fallback",
        )

    @staticmethod
    def _map_point_from_rotation(
        x_rot: float,
        y_rot: float,
        width: int,
        height: int,
        rotation: str,
    ) -> tuple[float, float]:
        if rotation == "cw":
            # Inverse of 90deg clockwise rotation.
            return y_rot, (height - 1) - x_rot
        if rotation == "ccw":
            # Inverse of 90deg counter-clockwise rotation.
            return (width - 1) - y_rot, x_rot
        if rotation == "180":
            return (width - 1) - x_rot, (height - 1) - y_rot
        return x_rot, y_rot

    def _extract_result(
        self,
        results: object,
        width: int,
        height: int,
        rotation: str,
    ) -> DetectionResult | None:
        faces = getattr(results, "multi_face_landmarks", None)
        if not faces:
            return None

        boxes: list[BoundingBox] = []
        all_landmarks: list[list[list[float]]] = []

        for face_landmarks in faces:
            coords: list[tuple[float, float]] = []
            for lm in face_landmarks.landmark:
                x_rot = lm.x * (width - 1)
                y_rot = lm.y * (height - 1)
                x, y = self._map_point_from_rotation(x_rot, y_rot, width, height, rotation)
                coords.append((x, y))

            xs = [int(x) for x, _ in coords]
            ys = [int(y) for _, y in coords]
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys)).clamp(width, height)
            boxes.append(box)
            all_landmarks.append([[float(x), float(y)] for x, y in coords])

        return DetectionResult(boxes=boxes, landmarks=all_landmarks)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._mesh is None:
            return self._fallback_full_frame(frame, "mesh_not_initialized")

        h, w = frame.shape[:2]
        attempts: list[tuple[np.ndarray, str]] = [
            (frame, "none"),
            (cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), "cw"),
            (cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE), "ccw"),
            (cv2.rotate(frame, cv2.ROTATE_180), "180"),
        ]

        for candidate, rotation in attempts:
            rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            results = self._mesh.process(rgb)
            mapped = self._extract_result(results, w, h, rotation)
            if mapped is not None and mapped.boxes:
                return mapped

        return DetectionResult(boxes=[], landmarks=[], error="no_face_detected")
