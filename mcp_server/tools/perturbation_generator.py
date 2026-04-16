from __future__ import annotations

import cv2
import threading

import torch

from src.perturbation.atn_engine import ATNEngine
from src.perturbation.atn_engine import DualHeadATNEngine
from src.utils.image_codec import decode_image_from_b64, encode_image_to_b64

_DUAL_ENGINE: DualHeadATNEngine | None = None
_LOCK = threading.Lock()


def _bgr_to_tensor(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def _tensor_to_bgr(face_tensor: torch.Tensor):
    rgb = (
        face_tensor.squeeze(0)
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        * 255.0
    ).round().astype("uint8")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _get_dual_engine(reface_engine: ATNEngine, config: dict) -> DualHeadATNEngine:
    global _DUAL_ENGINE
    with _LOCK:
        if _DUAL_ENGINE is not None:
            return _DUAL_ENGINE

        nsfw_model_cfg = config.get("models", {}).get("nsfw_trigger_atn", {})
        nsfw_ckpt = str(nsfw_model_cfg.get("path", "models/nsfw_trigger_atn.pth"))
        perturb_cfg = config.get("perturbation", {})
        nsfw_cfg = config.get("nsfw_trigger", {})
        device = str(perturb_cfg.get("device", "cpu"))
        alpha_shield = float(config.get("pipeline", {}).get("alpha_initial", 0.12))
        alpha_nsfw = float(nsfw_cfg.get("alpha", 0.05))

        _DUAL_ENGINE = DualHeadATNEngine(
            reface_engine=reface_engine,
            nsfw_checkpoint_path=nsfw_ckpt,
            device=device,
            alpha_shield=alpha_shield,
            alpha_nsfw=alpha_nsfw,
        )

        return _DUAL_ENGINE


def handle_perturbation_generator(engine: ATNEngine, payload: dict, config: dict | None = None) -> dict:
    face = decode_image_from_b64(payload["face_b64"])
    protection_profile = str(payload.get("protection_profile", "shield_only"))

    if protection_profile == "shield_only":
        perturb, latency_ms = engine.generate(face)
    else:
        dual_engine = _get_dual_engine(engine, config or {})
        face_tensor = _bgr_to_tensor(face)
        perturb_tensor = dual_engine.run(face_tensor, profile=protection_profile)
        perturb = _tensor_to_bgr(perturb_tensor)
        latency_ms = 0.0

    return {
        "perturbation_b64": encode_image_to_b64(perturb),
        "latency_ms": latency_ms,
        "protection_profile": protection_profile,
    }
