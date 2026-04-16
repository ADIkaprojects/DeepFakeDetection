from __future__ import annotations

import io
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as transforms_functional

from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble
from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN


class FaceCropDataset(Dataset[torch.Tensor]):
    """Dataset that loads face crops from a folder and returns tensors in [0,1]."""

    def __init__(self, data_dir: str, image_size: int = 224) -> None:
        root = Path(data_dir)
        self._paths = sorted(
            list(root.glob("*.jpg"))
            + list(root.glob("*.jpeg"))
            + list(root.glob("*.png"))
        )
        if not self._paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self._paths[index]).convert("RGB")
        return self._transform(image)


def jpeg_augment(batch: torch.Tensor, quality_range: tuple[int, int] = (70, 95)) -> torch.Tensor:
    """Apply per-image random JPEG compression/decompression to a batch in [0,1]."""
    quality = random.randint(*quality_range)
    augmented: list[torch.Tensor] = []
    for tensor in batch:
        image = transforms_functional.to_pil_image(tensor.clamp(0.0, 1.0))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decoded = Image.open(buffer).convert("RGB")
        augmented.append(transforms_functional.to_tensor(decoded))
    return torch.stack(augmented, dim=0).to(batch.device)


def random_resize_augment(batch: torch.Tensor, scale_range: tuple[float, float] = (0.85, 1.15)) -> torch.Tensor:
    """Apply random resize and restore operation to improve preprocessing robustness."""
    _, _, height, width = batch.shape
    scale = random.uniform(*scale_range)
    target_h = max(16, int(height * scale))
    target_w = max(16, int(width * scale))
    resized = torch.nn.functional.interpolate(
        batch,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return torch.nn.functional.interpolate(
        resized,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss returning 1-SSIM for optimization."""

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
        ssim = (numerator / denominator.clamp_min(1e-8)).mean()
        return 1.0 - ssim


class NSFWTriggerTrainer:
    """Training loop for NSFWTriggerATN using NSFW proxy supervision and imperceptibility losses."""

    def __init__(
        self,
        atn: NSFWTriggerATN,
        proxy_ensemble: NSFWProxyEnsemble,
        device: str = "cpu",
        lr: float = 1e-4,
        lambda_ssim: float = 2.0,
        lambda_tv: float = 0.1,
        use_jpeg_aug: bool = True,
        use_resize_aug: bool = True,
    ) -> None:
        self._device = torch.device(device)
        self._atn = atn.to(self._device)
        self._proxy = proxy_ensemble
        self._lambda_ssim = lambda_ssim
        self._lambda_tv = lambda_tv
        self._use_jpeg_aug = use_jpeg_aug
        self._use_resize_aug = use_resize_aug

        for model_bundle in self._proxy._models.values():
            model_bundle["model"].requires_grad_(False)

        self._optimizer = optim.Adam(self._atn.parameters(), lr=lr)
        self._ssim_loss = SSIMLoss()

    @staticmethod
    def _tv_loss(delta: torch.Tensor) -> torch.Tensor:
        """Total variation regularization to discourage high-frequency artifacts."""
        dy = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
        dx = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        return dx + dy

    def train_epoch(self, dataloader: DataLoader[torch.Tensor]) -> dict[str, float]:
        """Train for one epoch and return aggregated metrics."""
        self._atn.train()
        total_loss = 0.0
        total_nsfw = 0.0
        total_ssim = 0.0
        batches = 0

        for batch in dataloader:
            x = batch.to(self._device)
            x_perturbed, delta = self._atn(x)

            x_aug = x_perturbed
            if self._use_jpeg_aug:
                x_aug = jpeg_augment(x_aug)
            if self._use_resize_aug:
                x_aug = random_resize_augment(x_aug)

            nsfw_scores = self._proxy.score_tensor(x_aug)
            cls_loss = -torch.log(nsfw_scores.clamp(1e-7, 1.0 - 1e-7)).mean()
            ssim_loss = self._ssim_loss(x, x_perturbed)
            tv_loss = self._tv_loss(delta)

            loss = cls_loss + (self._lambda_ssim * ssim_loss) + (self._lambda_tv * tv_loss)

            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._atn.parameters(), max_norm=1.0)
            self._optimizer.step()

            total_loss += float(loss.item())
            total_nsfw += float(nsfw_scores.mean().item())
            total_ssim += float((1.0 - ssim_loss).item())
            batches += 1

        denom = max(1, batches)
        return {
            "loss": total_loss / denom,
            "nsfw_score_mean": total_nsfw / denom,
            "ssim_mean": total_ssim / denom,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict[str, float]) -> None:
        """Persist trainer checkpoint containing model weights, architecture signature, and metrics."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._atn.state_dict(),
            "arch": self._atn.arch_signature(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
