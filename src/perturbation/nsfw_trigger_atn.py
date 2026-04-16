from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("afs.nsfw_trigger_atn")


class ResBlock(torch.nn.Module):
    """Residual block with instance normalization used by NSFWTriggerATN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self._block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            torch.nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._block(x)


class NSFWTriggerATN(torch.nn.Module):
    """Encoder-decoder ATN that predicts bounded perturbations for NSFW trigger behavior."""

    MODEL_VERSION = "NSFWTriggerATN_v1"

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_res_blocks: int = 6,
        l_inf_bound: float = 0.06,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.l_inf_bound = l_inf_bound

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            torch.nn.InstanceNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(base_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(base_channels * 4),
            torch.nn.ReLU(inplace=True),
        )

        self.bottleneck = torch.nn.Sequential(
            *[ResBlock(base_channels * 4) for _ in range(num_res_blocks)]
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(base_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.InstanceNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, in_channels, kernel_size=7, padding=3, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return perturbed image and bounded perturbation delta for input tensor in [0,1]."""
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        raw_delta = self.decoder(latent)
        if raw_delta.shape[-2:] != x.shape[-2:]:
            raw_delta = F.interpolate(raw_delta, size=x.shape[-2:], mode="bilinear", align_corners=False)
        delta = torch.clamp(raw_delta * self.l_inf_bound, -self.l_inf_bound, self.l_inf_bound)
        perturbed = torch.clamp(x + delta, 0.0, 1.0)
        return perturbed, delta

    def arch_signature(self) -> dict[str, Any]:
        """Return stable architecture fingerprint for checkpoint compatibility checks."""
        return {
            "model_version": self.MODEL_VERSION,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "num_res_blocks": self.num_res_blocks,
            "l_inf_bound": self.l_inf_bound,
        }

    @staticmethod
    def compare_arch_signatures(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
        """Return field-level mismatch messages between checkpoint and runtime signatures."""
        mismatches: list[str] = []
        for key in sorted(set(expected.keys()) | set(actual.keys())):
            if expected.get(key) != actual.get(key):
                mismatches.append(f"{key}: checkpoint={expected.get(key)} runtime={actual.get(key)}")
        return mismatches


def load_nsfw_trigger_checkpoint(
    model: NSFWTriggerATN,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
) -> NSFWTriggerATN:
    """Load checkpoint weights with strict or filtered compatibility behavior."""
    path = Path(checkpoint_path)
    if not path.exists():
        LOGGER.warning("NSFW trigger checkpoint missing at %s", path)
        return model

    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, torch.nn.Module):
        loaded = loaded.state_dict()

    state_dict = loaded.get("model_state_dict") if isinstance(loaded, dict) else loaded
    if not isinstance(state_dict, dict):
        LOGGER.warning("Unsupported NSFW trigger checkpoint format at %s", path)
        return model

    if strict:
        model.load_state_dict(state_dict, strict=True)
        return model

    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)

    if skipped:
        LOGGER.warning(
            "Skipped %s incompatible NSFW trigger keys from %s",
            len(skipped),
            path,
        )

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    LOGGER.info(
        "Loaded NSFW trigger checkpoint from %s with %s/%s compatible tensors (missing=%s unexpected=%s)",
        path,
        len(compatible),
        len(state_dict),
        len(missing),
        len(unexpected),
    )
    return model
