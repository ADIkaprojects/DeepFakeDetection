from __future__ import annotations

import statistics
import time

import cv2


def benchmark_loop(source: int = 0, frames: int = 120) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Camera source unavailable")

    latencies = []
    try:
        for _ in range(frames):
            start = time.perf_counter()
            ok, _frame = cap.read()
            if not ok:
                continue
            latencies.append((time.perf_counter() - start) * 1000)
    finally:
        cap.release()

    if not latencies:
        raise RuntimeError("No frames captured")

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    p99 = latencies[int(len(latencies) * 0.99) - 1]

    print(f"capture_p50_ms={p50:.2f}")
    print(f"capture_p95_ms={p95:.2f}")
    print(f"capture_p99_ms={p99:.2f}")
    print(f"capture_mean_ms={statistics.mean(latencies):.2f}")


if __name__ == "__main__":
    benchmark_loop()
