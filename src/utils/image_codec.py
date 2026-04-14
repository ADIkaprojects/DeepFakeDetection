from __future__ import annotations

import base64

import cv2
import numpy as np


class CodecError(RuntimeError):
    pass


def encode_image_to_b64(image: np.ndarray, ext: str = ".png") -> str:
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise CodecError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_image_from_b64(payload: str) -> np.ndarray:
    try:
        raw = base64.b64decode(payload)
    except ValueError as exc:
        raise CodecError("Invalid Base64 payload") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise CodecError("Failed to decode image")
    return image
