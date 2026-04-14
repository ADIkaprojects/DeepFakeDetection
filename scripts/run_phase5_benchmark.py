from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080/rpc")
    parser.add_argument("--api-key", default="afs-local-dev-key")
    parser.add_argument("--frames", type=int, default=120)
    return parser.parse_args()


def rpc(endpoint: str, api_key: str, method: str, params: dict, request_id: int, timeout: int = 90) -> dict:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    res = requests.post(endpoint, json=payload, headers={"x-api-key": api_key}, timeout=timeout)
    res.raise_for_status()
    body = res.json()
    if "error" in body:
        raise RuntimeError(body["error"]["message"])
    return body["result"]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return sorted(values)[idx]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.utils.image_codec import decode_image_from_b64, encode_image_to_b64

    proc = subprocess.Popen(
        [args.python, "-m", "mcp_server.server", "--config", args.config, "--transport", "http"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        health = args.endpoint.replace("/rpc", "/health")
        ready = False
        for _ in range(20):
            try:
                r = requests.get(health, timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"Server failed to start: {stderr[:2000]}")

        e2e_latencies: list[float] = []
        ssim_scores: list[float] = []

        for i in range(args.frames):
            frame = np.full((480, 640, 3), 120, dtype=np.uint8)
            cv2.circle(frame, (320, 240), 90, (180, 160, 140), -1)
            cv2.ellipse(frame, (320, 240), (110, 130), 0, 0, 360, (170, 150, 130), 2)

            frame_b64 = encode_image_to_b64(frame)
            t0 = time.perf_counter()

            try:
                det = rpc(args.endpoint, args.api_key, "face_detector", {"frame_b64": frame_b64}, i * 10 + 1)
            except Exception:
                continue
            boxes = det.get("boxes") or [[210, 110, 430, 350]]
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            crop = frame[y1:y2, x1:x2]

            try:
                pert = rpc(
                args.endpoint,
                args.api_key,
                "perturbation_generator",
                {"face_b64": encode_image_to_b64(crop)},
                i * 10 + 2,
            )
                blended = rpc(
                args.endpoint,
                args.api_key,
                "frame_blender",
                {
                    "frame_b64": frame_b64,
                    "perturbation_b64": pert["perturbation_b64"],
                    "boxes": boxes,
                    "alpha": 0.12,
                },
                i * 10 + 3,
            )
                _feedback = rpc(
                args.endpoint,
                args.api_key,
                "deepfake_feedback",
                {"frame_b64": blended["shielded_frame_b64"]},
                i * 10 + 4,
            )
            except Exception:
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            e2e_latencies.append(elapsed_ms)

            shielded = decode_image_from_b64(blended["shielded_frame_b64"])
            gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(shielded, cv2.COLOR_BGR2GRAY)
            score = float(ssim(gray_a, gray_b, data_range=255))
            ssim_scores.append(score)

        if not e2e_latencies:
            raise RuntimeError("No successful benchmark frames collected")

        results = {
            "frames": args.frames,
            "frames_successful": len(e2e_latencies),
            "latency_ms": {
                "p50": percentile(e2e_latencies, 50),
                "p95": percentile(e2e_latencies, 95),
                "p99": percentile(e2e_latencies, 99),
                "mean": statistics.mean(e2e_latencies),
            },
            "ssim": {
                "mean": statistics.mean(ssim_scores),
                "min": min(ssim_scores),
                "max": max(ssim_scores),
            },
            "notes": "Synthetic-frame benchmark via live HTTP pipeline; attack API and MOS not run in this environment.",
        }

        out_dir = repo_root / "validation"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
