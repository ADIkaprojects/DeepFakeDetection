from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble
from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN
from src.training.nsfw_trigger_trainer import FaceCropDataset, NSFWTriggerTrainer

LOGGER = logging.getLogger("afs.train_nsfw_trigger")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for NSFW trigger model training."""
    parser = argparse.ArgumentParser(description="Train NSFWTriggerATN")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-ssim", type=float, default=2.0)
    parser.add_argument("--lambda-tv", type=float, default=0.1)
    parser.add_argument("--l-inf-bound", type=float, default=0.06)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--no-jpeg-aug", action="store_true")
    parser.add_argument("--no-resize-aug", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Train and export NSFW trigger model checkpoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FaceCropDataset(args.data_dir, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    atn = NSFWTriggerATN(l_inf_bound=args.l_inf_bound)
    proxy = NSFWProxyEnsemble(device=args.device)
    trainer = NSFWTriggerTrainer(
        atn=atn,
        proxy_ensemble=proxy,
        device=args.device,
        lr=args.lr,
        lambda_ssim=args.lambda_ssim,
        lambda_tv=args.lambda_tv,
        use_jpeg_aug=not args.no_jpeg_aug,
        use_resize_aug=not args.no_resize_aug,
    )

    best_nsfw = float("-inf")
    last_metrics: dict[str, float] = {"loss": 0.0, "nsfw_score_mean": 0.0, "ssim_mean": 0.0}

    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader)
        last_metrics = metrics
        LOGGER.info(
            "Epoch %s/%s | loss=%.4f | nsfw_score_mean=%.4f | ssim_mean=%.4f",
            epoch,
            args.epochs,
            metrics["loss"],
            metrics["nsfw_score_mean"],
            metrics["ssim_mean"],
        )

        if epoch % args.save_every == 0:
            trainer.save_checkpoint(str(output_dir / f"nsfw_trigger_atn_epoch{epoch:03d}.pth"), epoch, metrics)

        if metrics["nsfw_score_mean"] > best_nsfw:
            best_nsfw = metrics["nsfw_score_mean"]
            trainer.save_checkpoint(str(output_dir / "nsfw_trigger_atn_best.pth"), epoch, metrics)

    trainer.save_checkpoint(str(output_dir / "nsfw_trigger_atn.pth"), args.epochs, last_metrics)
    LOGGER.info("Training complete. Best nsfw_score_mean=%.4f", best_nsfw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
