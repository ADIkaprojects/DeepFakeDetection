from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

from src.utils.image_codec import encode_image_to_b64


def _wait_for_health(proc: subprocess.Popen[str], endpoint: str, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    health_url = endpoint.replace("/rpc", "/health")
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server exited before health check")
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError("Server did not become healthy in time and exited")
    raise RuntimeError("Server did not become healthy in time")


def test_nsfw_feedback_rpc_returns_score_in_range() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_key = "afs-local-dev-key"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    endpoint = f"http://127.0.0.1:{port}/rpc"

    smoke_config_path = repo_root / "config" / "smoke.yaml"
    config = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    config["transport"]["http_port"] = port

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(config, tmp)
        temp_config_path = tmp.name

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_server.server",
            "--config",
            temp_config_path,
            "--transport",
            "http",
        ],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_health(proc, endpoint)

        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        cv2.rectangle(frame, (24, 24), (72, 72), (180, 160, 130), -1)
        frame_b64 = encode_image_to_b64(frame)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "nsfw_feedback",
            "params": {"frame_b64": frame_b64},
        }
        body = None
        for _ in range(3):
            response = requests.post(
                endpoint,
                headers={"x-api-key": api_key},
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            candidate = response.json()
            result = candidate.get("result", {}) if isinstance(candidate, dict) else {}
            if isinstance(result, dict) and isinstance(result.get("nsfw_score"), float):
                body = candidate
                break
            time.sleep(1.0)

        assert body is not None
        assert "result" in body
        assert isinstance(body["result"]["nsfw_score"], float)
        assert 0.0 <= body["result"]["nsfw_score"] <= 1.0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        Path(temp_config_path).unlink(missing_ok=True)
