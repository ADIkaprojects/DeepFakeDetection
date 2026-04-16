from __future__ import annotations

import torch

from src.perturbation.perturbation_combiner import PerturbationCombiner


def test_combine_shield_only_returns_valid_range() -> None:
    combiner = PerturbationCombiner(alpha_shield=0.12, alpha_nsfw=0.05)
    face = torch.rand(1, 3, 224, 224)
    delta_shield = torch.rand(1, 3, 224, 224) * 0.05

    out = combiner.combine(face, delta_shield, delta_nsfw=None)

    assert out.shape == face.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_combine_with_nsfw_returns_valid_range() -> None:
    combiner = PerturbationCombiner(alpha_shield=0.12, alpha_nsfw=0.05)
    face = torch.rand(1, 3, 224, 224)
    delta_shield = torch.rand(1, 3, 224, 224) * 0.05
    delta_nsfw = torch.rand(1, 3, 224, 224) * 0.05

    out = combiner.combine(face, delta_shield, delta_nsfw=delta_nsfw)

    assert out.shape == face.shape
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
