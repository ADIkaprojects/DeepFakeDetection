from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as transforms_functional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble
from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN, load_nsfw_trigger_checkpoint

LOGGER = logging.getLogger("afs.validate_nsfw_trigger")


class SSIMLoss(torch.nn.Module):
    """Differentiable SSIM approximation used for validation metrics."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self._window_size = window_size
        self._c1 = 0.01 ** 2
        self._c2 = 0.03 ** 2
        self.register_buffer("_window", self._create_window(window_size), persistent=False)

    @staticmethod
    def _create_window(size: int) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * (1.5 ** 2)))
        gaussian = gaussian / gaussian.sum()
        kernel = gaussian.unsqueeze(1) @ gaussian.unsqueeze(0)
        return kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, size, size).contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        window = self._window.to(dtype=x.dtype, device=x.device)
        padding = self._window_size // 2
        mu_x = torch.nn.functional.conv2d(x, window, padding=padding, groups=3)
        mu_y = torch.nn.functional.conv2d(y, window, padding=padding, groups=3)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x = torch.nn.functional.conv2d(x * x, window, padding=padding, groups=3) - mu_x2
        sigma_y = torch.nn.functional.conv2d(y * y, window, padding=padding, groups=3) - mu_y2
        sigma_xy = torch.nn.functional.conv2d(x * y, window, padding=padding, groups=3) - mu_xy
        numerator = (2 * mu_xy + self._c1) * (2 * sigma_xy + self._c2)
        denominator = (mu_x2 + mu_y2 + self._c1) * (sigma_x + sigma_y + self._c2)
        return numerator.div(denominator.clamp_min(1e-8)).mean()


def parse_args() -> argparse.Namespace:
    """Parse validation CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate NSFW trigger checkpoint behavior")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="models/nsfw_trigger_atn.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    """Run validation and return 0 on pass, 1 on metric failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    image = Image.open(args.image).convert("RGB").resize((args.image_size, args.image_size))
    x = transforms_functional.to_tensor(image).unsqueeze(0).to(args.device)

    model = NSFWTriggerATN().to(args.device)
    model = load_nsfw_trigger_checkpoint(model, args.checkpoint, device=args.device, strict=False).to(args.device)
    model.eval()

    proxy = NSFWProxyEnsemble(device=args.device)
    ssim_metric = SSIMLoss().to(args.device)

    with torch.no_grad():
        before = float(proxy.score_tensor(x).item())
        perturbed, delta = model(x)
        after = float(proxy.score_tensor(perturbed).item())
        ssim = float(ssim_metric(x, perturbed).item())
        l_inf = float(delta.abs().max().item())

    print("========== NSFWTriggerATN Validation ==========")
    print(f"NSFW score before: {before:.6f}")
    print(f"NSFW score after : {after:.6f}")
    print(f"SSIM             : {ssim:.6f}")
    print(f"L_inf            : {l_inf:.6f}")

    ok_after = after > 0.5
    ok_ssim = ssim > 0.97
    ok_linf = l_inf <= 0.06

    if ok_after and ok_ssim and ok_linf:
        print("PASS: trigger raises NSFW score while remaining imperceptible.")
        return 0

    print("FAIL: one or more validation thresholds were not met.")
    if not ok_after:
        print("- NSFW score after perturbation is <= 0.5. Suggestion: train longer or strengthen classifier loss.")
    if not ok_ssim:
        print("- SSIM is <= 0.97. Suggestion: increase SSIM weight or lower blending strength.")
    if not ok_linf:
        print("- L_inf is > 0.06. Suggestion: reduce l_inf_bound in the trigger model.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
