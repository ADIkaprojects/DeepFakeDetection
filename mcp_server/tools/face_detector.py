from __future__ import annotations

from src.detection.face_detector import FaceDetector
from src.utils.image_codec import decode_image_from_b64


def handle_face_detector(detector: FaceDetector, payload: dict) -> dict:
    frame = decode_image_from_b64(payload["frame_b64"])
    result = detector.detect(frame)
    return {
        "boxes": [b.to_list() for b in result.boxes],
        "landmarks": result.landmarks,
        "error": result.error,
    }
