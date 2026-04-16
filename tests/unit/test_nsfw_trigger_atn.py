from __future__ import annotations

import torch

from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN


def test_output_shape_matches_input() -> None:
    model = NSFWTriggerATN(l_inf_bound=0.06)
    x = torch.rand(2, 3, 224, 224)
    perturbed, delta = model(x)

    assert perturbed.shape == x.shape
    assert delta.shape == x.shape


def test_delta_respects_l_inf_bound() -> None:
    l_inf_bound = 0.06
    model = NSFWTriggerATN(l_inf_bound=l_inf_bound)
    x = torch.rand(2, 3, 224, 224)
    _, delta = model(x)

    assert delta.abs().max().item() <= l_inf_bound + 1e-5


def test_perturbed_values_are_clamped() -> None:
    model = NSFWTriggerATN(l_inf_bound=0.06)
    x = torch.rand(2, 3, 224, 224)
    perturbed, _ = model(x)

    assert perturbed.min().item() >= -1e-5
    assert perturbed.max().item() <= 1.0 + 1e-5


def test_arch_signature_contains_required_fields() -> None:
    model = NSFWTriggerATN(l_inf_bound=0.06)
    signature = model.arch_signature()

    assert "model_version" in signature
    assert "l_inf_bound" in signature
