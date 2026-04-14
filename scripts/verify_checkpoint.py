from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.perturbation.atn_engine import ATNConfig, ATNEngine

LOGGER = logging.getLogger("afs.verify_checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ATN checkpoint compatibility without starting server")
    parser.add_argument("--checkpoint", default="models/reface_atn.pth", help="Checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Runtime device")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict startup checks (raises on incompatibilities)",
    )
    parser.add_argument(
        "--allow-identity-fallback",
        action="store_true",
        help="Allow identity fallback in non-strict mode",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        LOGGER.error("Checkpoint file does not exist: %s", checkpoint_path)
        return 1

    try:
        engine = ATNEngine(
            ATNConfig(
                model_path=str(checkpoint_path),
                device=args.device,
                strict_startup=bool(args.strict),
                allow_identity_fallback=bool(args.allow_identity_fallback),
            )
        )
    except (RuntimeError, FileNotFoundError) as exc:
        LOGGER.error("Compatibility verification failed for %s: %s", checkpoint_path, exc)
        return 1

    if engine.is_identity_mode():
        LOGGER.warning("Checkpoint %s loaded in identity fallback mode", checkpoint_path)
    else:
        LOGGER.info("Checkpoint %s loaded with active perturbation model", checkpoint_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
