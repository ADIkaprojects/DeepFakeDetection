from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", default="config/smoke.yaml")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 4:
        return "*" * len(secret)
    return ("*" * (len(secret) - 4)) + secret[-4:]


def resolve_run_config(args: argparse.Namespace) -> dict[str, str]:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = loaded if isinstance(loaded, dict) else {}
    transport_cfg = cfg.get("transport", {}) if isinstance(cfg.get("transport", {}), dict) else {}
    auth_cfg = transport_cfg.get("auth", {}) if isinstance(transport_cfg.get("auth", {}), dict) else {}

    config_endpoint = ""
    host = transport_cfg.get("host")
    port = transport_cfg.get("http_port")
    if host and port:
        config_endpoint = f"http://{host}:{port}/rpc"

    endpoint = args.endpoint or config_endpoint or os.getenv("DPFK_ENDPOINT", "")
    if not endpoint:
        raise RuntimeError(
            "No endpoint configured. Pass --endpoint, set in config YAML, or set DPFK_ENDPOINT env var."
        )

    api_key = args.api_key or str(auth_cfg.get("api_key", "")) or os.getenv("DPFK_API_KEY", "")
    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "config_path": str(config_path),
    }


def rpc_call(endpoint: str, api_key: str, method: str, params: dict, request_id: int) -> dict:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    res = requests.post(endpoint, json=payload, headers={"x-api-key": api_key}, timeout=30)
    res.raise_for_status()
    body = res.json()
    if "error" in body:
        raise RuntimeError(body["error"]["message"])
    return body["result"]


def encode_b64(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("encode failed")
    return buffer.tobytes().hex()


def decode_hex_png(payload: str) -> np.ndarray:
    data = bytes.fromhex(payload)
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("decode failed")
    return image


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    run_cfg = resolve_run_config(args)
    endpoint = run_cfg["endpoint"]
    api_key = run_cfg["api_key"]
    print(
        "[smoke] endpoint=%s config=%s api_key=%s"
        % (endpoint, run_cfg["config_path"], _mask_secret(api_key))
    )

    # Server expects Base64 payloads; use utility from codebase via subprocess-safe JSON payload.
    py = Path(args.python)
    if not py.exists():
        raise RuntimeError(f"Python executable not found: {py}")

    health = endpoint.replace("/rpc", "/health")
    proc: subprocess.Popen[str] | None = None
    started_local_server = False

    try:
        probe = requests.get(health, timeout=2)
        ready = probe.status_code == 200
    except Exception:
        ready = False

    if not ready:
        proc = subprocess.Popen(
            [str(py), "-m", "mcp_server.server", "--config", args.config, "--transport", "http"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        started_local_server = True

    try:
        for _ in range(180):
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("MCP server exited before becoming healthy")
            try:
                r = requests.get(health, timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)

        if not ready:
            raise RuntimeError("MCP server failed to start within timeout")

        # use local module functions to avoid duplicating base64 logic
        from src.utils.image_codec import encode_image_to_b64, decode_image_from_b64

        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        cv2.rectangle(frame, (220, 120), (420, 320), (180, 160, 130), -1)
        frame_b64 = encode_image_to_b64(frame)

        det = rpc_call(endpoint, api_key, "face_detector", {"frame_b64": frame_b64}, 1)
        boxes = det.get("boxes") or [[220, 120, 420, 320]]

        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        crop = frame[y1:y2, x1:x2]

        pert = rpc_call(
            endpoint,
            api_key,
            "perturbation_generator",
            {"face_b64": encode_image_to_b64(crop)},
            2,
        )

        blended = rpc_call(
            endpoint,
            api_key,
            "frame_blender",
            {
                "frame_b64": frame_b64,
                "perturbation_b64": pert["perturbation_b64"],
                "boxes": boxes,
                "alpha": 0.12,
            },
            3,
        )

        feedback = rpc_call(
            endpoint,
            api_key,
            "deepfake_feedback",
            {"frame_b64": blended["shielded_frame_b64"]},
            4,
        )

        nsfw_feedback = rpc_call(
            endpoint,
            api_key,
            "nsfw_feedback",
            {"frame_b64": blended["shielded_frame_b64"]},
            5,
        )

        nsfw_perturb = rpc_call(
            endpoint,
            api_key,
            "perturbation_generator",
            {
                "face_b64": encode_image_to_b64(crop),
                "protection_profile": "shield_and_nsfw",
            },
            6,
        )
        if "perturbation_b64" not in nsfw_perturb:
            raise RuntimeError("NSFW perturbation call did not return perturbation_b64")

        out = decode_image_from_b64(blended["shielded_frame_b64"])
        Path("validation").mkdir(exist_ok=True)
        cv2.imwrite("validation/smoke_output.png", out)

        result = {
            "status": "ok",
            "tools_called": [
                "face_detector",
                "perturbation_generator",
                "frame_blender",
                "deepfake_feedback",
                "nsfw_feedback",
                "perturbation_generator(shield_and_nsfw)",
            ],
            "feedback": feedback,
            "nsfw_feedback": nsfw_feedback,
            "nsfw_profile": nsfw_perturb.get("protection_profile"),
        }
        Path("validation/smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        if started_local_server and proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
