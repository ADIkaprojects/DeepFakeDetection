from __future__ import annotations

from src.blending.frame_blender import FrameBlender
from src.utils.image_codec import decode_image_from_b64, encode_image_to_b64
from src.utils.types import BoundingBox


def handle_frame_blender(blender: FrameBlender, payload: dict) -> dict:
    frame = decode_image_from_b64(payload["frame_b64"])
    perturbation = decode_image_from_b64(payload["perturbation_b64"])
    boxes = [BoundingBox(*[int(v) for v in box]) for box in payload["boxes"]]
    shielded = blender.blend(frame, perturbation, boxes, alpha=float(payload["alpha"]))
    return {"shielded_frame_b64": encode_image_to_b64(shielded)}
