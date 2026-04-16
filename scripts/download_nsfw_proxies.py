from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

LOGGER = logging.getLogger("afs.download_nsfw_proxies")


def download_falconsai(target_dir: Path) -> Path:
    """Download and cache the Falconsai NSFW proxy model into the given directory."""
    cache_dir = target_dir / "falconsai_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading Falconsai NSFW image classifier to %s", cache_dir)
    AutoImageProcessor.from_pretrained(
        "Falconsai/nsfw_image_detection",
        cache_dir=str(cache_dir),
    )
    model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        cache_dir=str(cache_dir),
    )
    model.eval()

    output_path = target_dir / "falconsai_nsfw_vit.pth"
    torch.save(model.state_dict(), output_path)
    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    proxy_dir = Path("models") / "nsfw_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    try:
        output_path = download_falconsai(proxy_dir)
        print(f"Saved NSFW proxy weights: {output_path}")
        return 0
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Failed to download NSFW proxy weights")
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
