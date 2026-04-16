from __future__ import annotations

import base64
import io
import logging
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

LOGGER = logging.getLogger("afs.nsfw_feedback")

PROXY_REGISTRY: dict[str, dict[str, Any]] = {
    "falconsai": {
        "hf_id": "Falconsai/nsfw_image_detection",
        "cache_dir": "models/nsfw_proxy/falconsai_cache",
    }
}


class NSFWProxyEnsemble:
    """Ensemble wrapper for NSFW proxy classifiers used for runtime scoring and training feedback."""

    def __init__(self, device: str = "cpu", proxies: list[str] | None = None) -> None:
        self._device = torch.device(device)
        self._proxies = proxies or list(PROXY_REGISTRY.keys())
        self._models: dict[str, dict[str, Any]] = {}
        self._load_models()

    def _load_models(self) -> None:
        for proxy_name in self._proxies:
            cfg = PROXY_REGISTRY.get(proxy_name)
            if cfg is None:
                LOGGER.warning("Unknown NSFW proxy '%s'; skipping", proxy_name)
                continue

            processor = AutoImageProcessor.from_pretrained(
                cfg["hf_id"],
                cache_dir=cfg["cache_dir"],
            )
            model = AutoModelForImageClassification.from_pretrained(
                cfg["hf_id"],
                cache_dir=cfg["cache_dir"],
            )
            model = model.to(self._device).eval()
            model.requires_grad_(False)

            nsfw_index = self._resolve_nsfw_index(model.config.id2label)
            self._models[proxy_name] = {
                "processor": processor,
                "model": model,
                "nsfw_index": nsfw_index,
            }

    @staticmethod
    def _resolve_nsfw_index(id2label: dict[int, str] | dict[str, str]) -> int:
        for raw_idx, label in id2label.items():
            idx = int(raw_idx)
            normalized = str(label).lower()
            if any(token in normalized for token in ("nsfw", "explicit", "unsafe", "porn")):
                return idx

        LOGGER.warning("Unable to infer NSFW class index from id2label=%s; defaulting to class index 1", id2label)
        return 1

    def score_tensor(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Score a (B,3,H,W) float tensor in [0,1] and return ensemble NSFW probabilities with shape (B,)."""
        if not self._models:
            raise RuntimeError("No NSFW proxy models loaded")

        scores: list[torch.Tensor] = []
        for metadata in self._models.values():
            processor = metadata["processor"]
            model = metadata["model"]
            nsfw_index = int(metadata["nsfw_index"])

            size = processor.size
            if hasattr(size, "height") or hasattr(size, "width"):
                height = getattr(size, "height", None)
                width = getattr(size, "width", None)
                shortest_edge = getattr(size, "shortest_edge", None)
                target_h = int(height or shortest_edge or 224)
                target_w = int(width or shortest_edge or target_h)
            elif isinstance(size, dict):
                target_h = int(size.get("height") or size.get("shortest_edge") or 224)
                target_w = int(size.get("width") or size.get("shortest_edge") or target_h)
            else:
                target_h = int(size)
                target_w = int(size)

            x = F.interpolate(
                face_tensor.to(self._device),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

            image_mean = torch.tensor(processor.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            image_std = torch.tensor(processor.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            pixel_values = (x - image_mean) / image_std

            logits = model(pixel_values=pixel_values).logits
            probs = F.softmax(logits, dim=-1)
            scores.append(probs[:, nsfw_index])

        stacked = torch.stack(scores, dim=1)
        return stacked.prod(dim=1).pow(1.0 / stacked.shape[1])

    def score_b64(self, image_b64: str) -> dict[str, Any]:
        """Score a base64 image and return JSON-serializable result payload for MCP responses."""
        raw = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        tensor = self._pil_to_tensor(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            score = float(self.score_tensor(tensor).item())

        label = "nsfw_flagged" if score >= 0.5 else "safe"
        return {
            "nsfw_score": round(score, 6),
            "label": label,
            "proxies_used": list(self._models.keys()),
        }

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert PIL RGB image to float32 tensor in [0,1] with shape (3,H,W)."""
        import torchvision.transforms.functional as transforms_functional

        return transforms_functional.to_tensor(image)


def build_nsfw_feedback_engine(config: dict[str, Any]) -> NSFWProxyEnsemble:
    """Factory for NSFW feedback engine from config.nsfw_trigger.{device,proxies}."""
    nsfw_cfg = config.get("nsfw_trigger", {})
    device = str(nsfw_cfg.get("device", "cpu"))
    proxies = nsfw_cfg.get("proxies", ["falconsai"])
    if not isinstance(proxies, list):
        proxies = ["falconsai"]
    return NSFWProxyEnsemble(device=device, proxies=[str(proxy) for proxy in proxies])
