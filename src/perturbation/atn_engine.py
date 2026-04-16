from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger("afs.atn_engine")


@dataclass(slots=True)
class ATNConfig:
    model_path: str
    input_size: int = 224
    device: str = "cuda"
    epsilon: float = 0.20
    strict_startup: bool = True
    allow_identity_fallback: bool = False


class IdentityATN(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(8, channels),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(8, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ReFaceATN(torch.nn.Module):
    def __init__(self, epsilon: float = 0.20) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.GELU())
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.bottleneck = torch.nn.Sequential(
            ResidualBlock(256, dropout=0.1),
            ResidualBlock(256, dropout=0.1),
        )
        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.out = torch.nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        raw = self.out(torch.cat([d2, e1], dim=1))
        return x + self.epsilon * torch.tanh(raw)

    def arch_signature(self) -> dict[str, Any]:
        """Return a stable architecture fingerprint for checkpoint compatibility checks."""
        return {
            "activation": "gelu",
            "bottleneck_dropout": True,
            "output_bound": "eps_tanh",
            "channel_widths": [64, 128, 256],
        }


def _filter_compatible_state_dict(
    model: torch.nn.Module,
    ckpt_dict: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """Return only checkpoint tensors that are loadable by key and exact shape."""
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}

    for key, value in ckpt_dict.items():
        if key not in model_state:
            LOGGER.warning(
                "Skipping checkpoint key '%s' from %s: missing in model state",
                key,
                checkpoint_path,
            )
            continue

        if tuple(model_state[key].shape) != tuple(value.shape):
            LOGGER.warning(
                "Skipping checkpoint key '%s' from %s: shape mismatch checkpoint=%s model=%s",
                key,
                checkpoint_path,
                tuple(value.shape),
                tuple(model_state[key].shape),
            )
            continue

        compatible[key] = value

    return compatible


def compare_arch_signatures(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> list[str]:
    """Return field-level mismatch messages between checkpoint and runtime signatures."""
    mismatches: list[str] = []
    keys = sorted(set(expected.keys()) | set(actual.keys()))
    for key in keys:
        lhs = expected.get(key)
        rhs = actual.get(key)
        if lhs != rhs:
            mismatches.append(f"{key}: checkpoint={lhs} runtime={rhs}")
    return mismatches


class ATNEngine:
    def __init__(self, config: ATNConfig) -> None:
        self._config = config
        self._device = torch.device(
            config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self._model: torch.nn.Module = ReFaceATN(epsilon=config.epsilon)
        self._identity_mode = False
        self._load_model()

    def is_identity_mode(self) -> bool:
        """Return whether the engine is running in identity fallback mode."""
        return self._identity_mode

    def _enable_identity_mode(self, reason: str) -> None:
        self._model = IdentityATN().to(self._device).eval()
        self._identity_mode = True
        LOGGER.critical("%s", reason)

    def _load_model(self) -> None:
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            message = f"ATN checkpoint missing at {model_path}"
            if self._config.strict_startup:
                raise FileNotFoundError(message)
            if self._config.allow_identity_fallback:
                self._enable_identity_mode(f"{message}; identity fallback enabled")
                return
            raise RuntimeError(message)

        try:
            loaded = torch.load(model_path, map_location=self._device)
        except (RuntimeError, FileNotFoundError) as exc:
            LOGGER.exception("Failed to load ATN checkpoint from %s", model_path)
            if self._config.strict_startup:
                raise
            if self._config.allow_identity_fallback:
                self._enable_identity_mode(
                    f"ATN checkpoint load failed at {model_path}; entering identity mode: {exc}"
                )
                return
            raise

        if isinstance(loaded, torch.nn.Module):
            self._model = loaded
        elif isinstance(loaded, dict):
            model = ReFaceATN(epsilon=self._config.epsilon)
            state_dict = loaded.get("model_state_dict") or loaded.get("state_dict") or loaded
            if not isinstance(state_dict, Mapping):
                message = f"Unsupported ATN state_dict format at {model_path}"
                LOGGER.error(message)
                if self._config.strict_startup:
                    raise RuntimeError(message)
                if self._config.allow_identity_fallback:
                    self._enable_identity_mode(f"{message}; identity fallback enabled")
                    return
                raise RuntimeError(message)

            checkpoint_signature = loaded.get("arch") if isinstance(loaded.get("arch"), Mapping) else None
            runtime_signature = model.arch_signature()
            if checkpoint_signature is None:
                LOGGER.warning(
                    "Checkpoint %s has no architecture signature; proceeding with backward-compatible load",
                    model_path,
                )
            else:
                signature_mismatches = compare_arch_signatures(checkpoint_signature, runtime_signature)
                for mismatch in signature_mismatches:
                    LOGGER.warning("Architecture mismatch for %s: %s", model_path, mismatch)

            compatible_state = _filter_compatible_state_dict(model, state_dict, model_path)

            try:
                missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            except RuntimeError:
                LOGGER.exception("RuntimeError while applying filtered ATN checkpoint from %s", model_path)
                if self._config.strict_startup:
                    raise
                if self._config.allow_identity_fallback:
                    self._enable_identity_mode(
                        f"Filtered ATN checkpoint application failed for {model_path}; identity fallback enabled"
                    )
                    return
                raise

            loaded_count = len(compatible_state)
            if loaded_count == 0:
                message = (
                    f"ATN checkpoint {model_path} provided zero compatible keys; "
                    "model remains uninitialized from checkpoint"
                )
                LOGGER.error(message)
                if self._config.strict_startup:
                    raise RuntimeError(message)
                if self._config.allow_identity_fallback:
                    self._enable_identity_mode(
                        f"{message}; identity fallback enabled"
                    )
                    return

            if not missing and not unexpected and checkpoint_signature == runtime_signature:
                LOGGER.info("Checkpoint fully compatible from %s; strict mode is safe.", model_path)
            else:
                LOGGER.warning(
                    "ATN checkpoint partially loaded from %s: compatible=%s/%s, missing=%s, unexpected=%s",
                    model_path,
                    loaded_count,
                    len(state_dict),
                    len(missing),
                    len(unexpected),
                )

            if self._config.strict_startup and (missing or unexpected):
                raise RuntimeError(
                    f"ATN checkpoint incompatible at {model_path}: missing={len(missing)} unexpected={len(unexpected)}"
                )
            self._model = model
            self._identity_mode = False
        else:
            if self._config.strict_startup:
                raise RuntimeError("Unsupported ATN checkpoint format")
            if self._config.allow_identity_fallback:
                self._enable_identity_mode(
                    f"Unsupported ATN checkpoint format at {model_path}; identity fallback enabled"
                )
            else:
                raise RuntimeError("Unsupported ATN checkpoint format")

        self._model.to(self._device).eval()

    def generate(self, face_crop: np.ndarray) -> tuple[np.ndarray, float]:
        resized = cv2.resize(face_crop, (self._config.input_size, self._config.input_size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0).to(self._device)

        start = torch.cuda.Event(enable_timing=True) if self._device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self._device.type == "cuda" else None

        with torch.no_grad():
            if start and end:
                start.record()
            pred = self._model(tensor)
            if start and end:
                end.record()
                torch.cuda.synchronize()
                latency_ms = float(start.elapsed_time(end))
            else:
                latency_ms = 0.0

        out = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = cv2.resize(out, (face_crop.shape[1], face_crop.shape[0]))
        return out, latency_ms


from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN, load_nsfw_trigger_checkpoint
from src.perturbation.perturbation_combiner import PerturbationCombiner


class DualHeadATNEngine:
    """Dual-head wrapper that combines existing shield perturbation with optional NSFW trigger perturbation."""

    PROFILE_SHIELD_ONLY = "shield_only"
    PROFILE_NSFW_ONLY = "nsfw_trigger_only"
    PROFILE_COMBINED = "shield_and_nsfw"

    def __init__(
        self,
        reface_engine: ATNEngine,
        nsfw_checkpoint_path: str | None = None,
        device: str = "cpu",
        alpha_shield: float = 0.12,
        alpha_nsfw: float = 0.05,
    ) -> None:
        self._reface_engine = reface_engine
        self._device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self._nsfw_atn = NSFWTriggerATN().to(self._device).eval()
        if nsfw_checkpoint_path:
            self._nsfw_atn = load_nsfw_trigger_checkpoint(
                self._nsfw_atn,
                nsfw_checkpoint_path,
                device=str(self._device),
                strict=False,
            ).to(self._device).eval()

        self._combiner = PerturbationCombiner(
            alpha_shield=alpha_shield,
            alpha_nsfw=alpha_nsfw,
        )

    @staticmethod
    def _face_tensor_to_bgr(face_tensor: torch.Tensor) -> np.ndarray:
        face = face_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
        rgb = (face.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_face_tensor(face_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(device)

    @torch.no_grad()
    def run(self, face_tensor: torch.Tensor, profile: str = PROFILE_COMBINED) -> torch.Tensor:
        """Run selected perturbation profile and return perturbed tensor in [0,1]."""
        input_tensor = face_tensor.to(self._device)

        if profile == self.PROFILE_SHIELD_ONLY:
            bgr = self._face_tensor_to_bgr(input_tensor)
            shield_bgr, _ = self._reface_engine.generate(bgr)
            return self._bgr_to_face_tensor(shield_bgr, self._device)

        if profile == self.PROFILE_NSFW_ONLY:
            perturbed, _ = self._nsfw_atn(input_tensor)
            return perturbed

        bgr = self._face_tensor_to_bgr(input_tensor)
        shield_bgr, _ = self._reface_engine.generate(bgr)
        shield_tensor = self._bgr_to_face_tensor(shield_bgr, self._device)
        delta_shield = shield_tensor - input_tensor

        _, delta_nsfw = self._nsfw_atn(input_tensor)
        return self._combiner.combine(input_tensor, delta_shield, delta_nsfw)
