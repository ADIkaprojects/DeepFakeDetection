from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger("afs.perturbation_combiner")


class PerturbationCombiner:
    """Combine shield and NSFW perturbation deltas with joint imperceptibility constraints."""

    def __init__(
        self,
        alpha_shield: float = 0.12,
        alpha_nsfw: float = 0.05,
        joint_l_inf_cap: float = 0.10,
        ssim_floor: float = 0.97,
    ) -> None:
        self.alpha_shield = alpha_shield
        self.alpha_nsfw = alpha_nsfw
        self.joint_l_inf_cap = joint_l_inf_cap
        self.ssim_floor = ssim_floor

    def combine(
        self,
        face_tensor: torch.Tensor,
        delta_shield: torch.Tensor,
        delta_nsfw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return combined perturbed tensor in [0,1] after L-infinity clamp and SSIM guard."""
        combined_delta = self.alpha_shield * delta_shield
        if delta_nsfw is not None:
            combined_delta = combined_delta + self.alpha_nsfw * delta_nsfw

        combined_delta = torch.clamp(combined_delta, -self.joint_l_inf_cap, self.joint_l_inf_cap)
        perturbed = torch.clamp(face_tensor + combined_delta, 0.0, 1.0)

        if delta_nsfw is not None:
            ssim_value = self._ssim_proxy(face_tensor, perturbed)
            if ssim_value < self.ssim_floor:
                LOGGER.warning(
                    "Combined SSIM proxy %.4f below floor %.4f; reducing NSFW contribution",
                    ssim_value,
                    self.ssim_floor,
                )
                reduced_delta = (self.alpha_shield * delta_shield) + ((self.alpha_nsfw * 0.5) * delta_nsfw)
                reduced_delta = torch.clamp(reduced_delta, -self.joint_l_inf_cap, self.joint_l_inf_cap)
                perturbed = torch.clamp(face_tensor + reduced_delta, 0.0, 1.0)

        return perturbed

    @staticmethod
    def _ssim_proxy(x: torch.Tensor, y: torch.Tensor) -> float:
        """Fast quality proxy in [0,1] used as a runtime SSIM guard approximation."""
        mse = float(((x - y) ** 2).mean().item())
        return max(0.0, min(1.0, 1.0 - (mse * 100.0)))
