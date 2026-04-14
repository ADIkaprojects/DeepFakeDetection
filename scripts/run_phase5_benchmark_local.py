from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--frames", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.blending.frame_blender import BlendConfig, FrameBlender
    from src.detection.face_detector import FaceDetector, FaceDetectorConfig
    from src.feedback.deepsafe_engine import DeepSafeConfig, DeepSafeEngine
    from src.perturbation.atn_engine import ATNConfig, ATNEngine
    from src.utils.config import load_config
    from src.utils.types import BoundingBox

    cfg = load_config(args.config)
    detector = FaceDetector(
        FaceDetectorConfig(
            min_detection_confidence=float(cfg["detection"]["min_detection_confidence"]),
            min_tracking_confidence=float(cfg["detection"]["min_tracking_confidence"]),
        )
    )
    atn = ATNEngine(
        ATNConfig(
            model_path="models/reface_atn.pth",
            input_size=int(cfg["perturbation"]["atn_input_size"]),
            device=str(cfg["perturbation"]["device"]),
            strict_startup=False,
            allow_identity_fallback=True,
        )
    )
    deep = DeepSafeEngine(
        DeepSafeConfig(
            model_path="models/deepsafe.pth",
            input_size=int(cfg["feedback"].get("input_size", 224)),
            device=str(cfg["perturbation"]["device"]),
            strict_startup=False,
        )
    )
    blender = FrameBlender(BlendConfig(alpha=float(cfg["pipeline"]["alpha_initial"])))

    latencies = []
    scores = []

    for _ in range(args.frames):
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        cv2.circle(frame, (320, 240), 90, (180, 160, 140), -1)
        cv2.ellipse(frame, (320, 240), (110, 130), 0, 0, 360, (170, 150, 130), 2)

        t0 = time.perf_counter()
        det = detector.detect(frame)
        boxes = [b.to_list() for b in det.boxes] if det.boxes else [[210, 110, 430, 350]]
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]

        crop = frame[y1:y2, x1:x2]
        perturb, _ = atn.generate(crop)
        shielded = blender.blend(frame, perturb, [BoundingBox(x1, y1, x2, y2)], alpha=0.12)
        _confidence, _label = deep.infer(shielded)

        latencies.append((time.perf_counter() - t0) * 1000.0)

        gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(shielded, cv2.COLOR_BGR2GRAY)
        scores.append(float(ssim(gray_a, gray_b, data_range=255)))

    results = {
        "frames": args.frames,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": statistics.mean(latencies),
        },
        "ssim": {
            "mean": statistics.mean(scores),
            "min": min(scores),
            "max": max(scores),
        },
        "notes": "Local synthetic benchmark without commercial attack API or MOS study.",
    }

    out = repo_root / "validation" / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
