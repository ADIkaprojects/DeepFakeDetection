from __future__ import annotations

import threading
from typing import Any

from src.feedback.deepsafe_engine import BaseFeedbackEngine, DeepSafeConfig, build_feedback_engine
from src.utils.image_codec import decode_image_from_b64

_ENGINE: BaseFeedbackEngine | None = None
_LOCK = threading.Lock()


def init_deepsafe(config: dict[str, Any], model_path: str) -> None:
    global _ENGINE
    with _LOCK:
        if _ENGINE is not None:
            return

        feedback_cfg = config["feedback"]
        perturb_cfg = config["perturbation"]
        strict = bool(config["models"].get("strict_startup", True))
        _ENGINE = build_feedback_engine(
            DeepSafeConfig(
                model_path=model_path,
                input_size=int(feedback_cfg.get("input_size", 224)),
                device=str(perturb_cfg["device"]),
                strict_startup=strict,
                use_ufd_backend=bool(feedback_cfg.get("use_ufd_backend", False)),
            )
        )


def handle_deepfake_feedback(payload: dict) -> dict:
    if _ENGINE is None:
        raise RuntimeError("DeepSafe engine not initialized")

    frame = decode_image_from_b64(payload["frame_b64"])
    confidence, label = _ENGINE.infer(frame)
    return {"confidence": confidence, "label": label}
