from __future__ import annotations

from src.perturbation.atn_engine import ATNEngine
from src.utils.image_codec import decode_image_from_b64, encode_image_to_b64


def handle_perturbation_generator(engine: ATNEngine, payload: dict) -> dict:
    face = decode_image_from_b64(payload["face_b64"])
    perturb, latency_ms = engine.generate(face)
    return {
        "perturbation_b64": encode_image_to_b64(perturb),
        "latency_ms": latency_ms,
    }
