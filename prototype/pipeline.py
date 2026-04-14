from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from src.blending.frame_blender import BlendConfig, FrameBlender
from src.detection.face_detector import FaceDetector, FaceDetectorConfig
from src.perturbation.atn_engine import ATNConfig, ATNEngine
from src.utils.config import load_config
from src.utils.logging_utils import configure_logging
from src.utils.model_registry import validate_registry

LOGGER = logging.getLogger("afs.prototype")


def resolve_model_path(registry_path: str, model_path: str) -> str:
    path = Path(model_path)
    if path.is_absolute():
        return str(path)

    registry_file = Path(registry_path)
    candidate_from_root = registry_file.parent.parent / path
    candidate_from_models = registry_file.parent / path
    if candidate_from_root.exists() or not candidate_from_models.exists():
        return str(candidate_from_root)
    return str(candidate_from_models)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monolithic AFS prototype")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--source", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg["logging"]["level"], cfg["logging"]["output"])

    registry = validate_registry(cfg["models"]["registry_path"], strict=False)
    atn_path = resolve_model_path(
        cfg["models"]["registry_path"],
        registry[cfg["perturbation"]["model_key"]]["path"],
    )

    detector = FaceDetector(
        FaceDetectorConfig(
            min_detection_confidence=cfg["detection"]["min_detection_confidence"],
            min_tracking_confidence=cfg["detection"]["min_tracking_confidence"],
        )
    )
    engine = ATNEngine(
        ATNConfig(
            model_path=atn_path,
            input_size=cfg["perturbation"]["atn_input_size"],
            device=cfg["perturbation"]["device"],
        )
    )
    blender = FrameBlender(BlendConfig(alpha=cfg["pipeline"]["alpha_initial"]))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        start = time.perf_counter()
        det = detector.detect(frame)
        out = frame.copy()
        for box in det.boxes:
            crop = frame[box.y1:box.y2, box.x1:box.x2]
            if crop.size == 0:
                continue
            pert, _lat = engine.generate(crop)
            out = blender.blend(out, pert, [box])

        latency_ms = (time.perf_counter() - start) * 1000
        cv2.putText(out, f"latency={latency_ms:.1f}ms", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("AFS Prototype", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
