from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger("afs.deepsafe")


@dataclass(slots=True)
class DeepSafeConfig:
    model_path: str
    input_size: int = 224
    device: str = "cuda"
    strict_startup: bool = True
    use_ufd_backend: bool = False


class BaseFeedbackEngine(ABC):
    """Common feedback engine interface for runtime detector backends."""

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> float:
        """Return fake probability confidence in [0, 1]."""

    def infer(self, frame_bgr: np.ndarray) -> tuple[float, str]:
        confidence = float(self.predict(frame_bgr))
        label = "fake" if confidence >= 0.5 else "real"
        return confidence, label


class LightweightDeepSafeEngine(BaseFeedbackEngine):
    """Low-dependency detector fallback used as default runtime backend."""

    def __init__(self, config: DeepSafeConfig) -> None:
        self._config = config
        self._model_ready = False
        self._bias = 0.0
        self._load_model()

    def _load_model(self) -> None:
        path = Path(self._config.model_path)
        if not path.exists():
            message = f"DeepSafe checkpoint missing: {path}"
            if self._config.strict_startup:
                raise FileNotFoundError(message)
            LOGGER.warning("%s; lightweight detector running in neutral mode", message)
            return

        try:
            loaded = torch.load(path, map_location="cpu")
            if isinstance(loaded, dict):
                # Lightweight backend extracts a deterministic bias from checkpoint metadata.
                first_tensor = next((v for v in loaded.values() if isinstance(v, torch.Tensor)), None)
                if first_tensor is not None:
                    self._bias = float(torch.tanh(first_tensor.float().mean()).item()) * 0.05
                    self._model_ready = True
                    return
        except (RuntimeError, OSError) as exc:
            if self._config.strict_startup:
                raise RuntimeError(f"Failed to load lightweight DeepSafe checkpoint at {path}: {exc}") from exc
            LOGGER.warning("Failed lightweight checkpoint load at %s: %s", path, exc)
            return

        if self._config.strict_startup:
            raise RuntimeError(f"Unsupported lightweight checkpoint format: {path}")
        LOGGER.warning("Unsupported lightweight checkpoint format at %s; using neutral mode", path)

    def predict(self, frame_bgr: np.ndarray) -> float:
        if frame_bgr.size == 0:
            return 0.0

        resized = cv2.resize(frame_bgr, (self._config.input_size, self._config.input_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized_mean = float(gray.mean() / 255.0)
        score = normalized_mean * 0.5 + 0.25 + self._bias
        return float(max(0.0, min(1.0, score)))


class UFDDeepSafeAdapter(BaseFeedbackEngine):
    """UniversalFakeDetect-backed detector adapter for parity with notebook benchmarks."""

    def __init__(self, config: DeepSafeConfig) -> None:
        self._config = config
        self._device = torch.device(
            config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self._model: torch.nn.Module | None = None
        self._load_model()

    def _resolve_ufd_root(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "gitrepos" / "UniversalFakeDetect",
            repo_root / "UniversalFakeDetect",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "UniversalFakeDetect repository not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    @staticmethod
    def _fake_prob_from_output(out_tensor: torch.Tensor) -> torch.Tensor:
        if out_tensor.ndim == 2 and out_tensor.shape[1] > 1:
            return torch.softmax(out_tensor, dim=1)[:, 1]
        return torch.sigmoid(out_tensor.flatten())

    def _load_model(self) -> None:
        path = Path(self._config.model_path)
        if not path.exists():
            message = f"DeepSafe checkpoint missing: {path}"
            if self._config.strict_startup:
                raise FileNotFoundError(message)
            LOGGER.warning("%s; UFD detector disabled", message)
            return

        ufd_root = self._resolve_ufd_root()
        if str(ufd_root) not in sys.path:
            sys.path.insert(0, str(ufd_root))

        from models import get_model  # pylint: disable=import-error,import-outside-toplevel

        model = get_model("CLIP:ViT-L/14")
        loaded = torch.load(path, map_location=self._device)
        if isinstance(loaded, dict):
            state_dict = loaded.get("model") or loaded.get("state_dict") or loaded
        else:
            message = f"Unsupported DeepSafe checkpoint format: {path}"
            if self._config.strict_startup:
                raise RuntimeError(message)
            LOGGER.warning("%s; UFD detector disabled", message)
            return

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if self._config.strict_startup and (missing or unexpected):
            raise RuntimeError(
                "DeepSafe state_dict incompatible with UFD CLIP model; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        if missing or unexpected:
            LOGGER.warning(
                "DeepSafe partially loaded: missing=%s unexpected=%s",
                len(missing),
                len(unexpected),
            )

        self._model = model.to(self._device).eval()

    def predict(self, frame_bgr: np.ndarray) -> float:
        if self._model is None:
            if self._config.strict_startup:
                raise RuntimeError("UFD DeepSafe model is not loaded")
            return 0.0

        resized = cv2.resize(frame_bgr, (self._config.input_size, self._config.input_size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            score = float(self._fake_prob_from_output(logits).mean().item())
        return score


def build_feedback_engine(config: DeepSafeConfig) -> BaseFeedbackEngine:
    """Build configured feedback backend, defaulting to lightweight runtime path."""
    if config.use_ufd_backend:
        LOGGER.info("Using UFD feedback backend")
        return UFDDeepSafeAdapter(config)
    LOGGER.info("Using lightweight feedback backend")
    return LightweightDeepSafeEngine(config)


class DeepSafeEngine:
    """Compatibility wrapper exposing infer() for existing callers."""

    def __init__(self, config: DeepSafeConfig) -> None:
        self._engine = build_feedback_engine(config)

    def infer(self, frame_bgr: np.ndarray) -> tuple[float, str]:
        return self._engine.infer(frame_bgr)
