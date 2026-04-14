from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import cv2
import requests

from mcp_client.alpha_controller import AlphaController, AlphaControllerConfig
from mcp_client.virt_cam_driver import VirtualCamConfig, VirtualCamDriver
from src.utils.config import load_config
from src.utils.image_codec import decode_image_from_b64, encode_image_to_b64
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger("afs.capture_client")


@dataclass(slots=True)
class RpcClient:
    endpoint: str
    api_key: str | None = None

    def call(self, method: str, params: dict, request_id: int) -> dict:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(body["error"]["message"])
        return body["result"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFS capture client")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--source", type=int, default=0, help="OpenCV camera source")
    parser.add_argument("--rpc-endpoint", default="http://127.0.0.1:8080/rpc")
    parser.add_argument("--virtual-cam", action="store_true")
    return parser.parse_args()


def run_capture_loop(config: dict, source: int, endpoint: str, use_virtual_cam: bool) -> None:
    pipeline = config["pipeline"]
    feedback = config["feedback"]

    alpha_controller = AlphaController(
        AlphaControllerConfig(
            initial=float(pipeline["alpha_initial"]),
            min_alpha=float(pipeline["alpha_min"]),
            max_alpha=float(pipeline["alpha_max"]),
            increase_step=float(feedback["increase_step"]),
            decay_rate=float(pipeline["alpha_decay_rate"]),
            high_threshold=float(feedback["high_threshold"]),
            low_threshold=float(feedback["low_threshold"]),
        )
    )

    api_key = config.get("transport", {}).get("auth", {}).get("api_key")
    rpc = RpcClient(endpoint=endpoint, api_key=api_key)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")

    target_w, target_h = config["pipeline"]["resolution"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    vcam = None
    if use_virtual_cam:
        vcam = VirtualCamDriver(VirtualCamConfig(width=target_w, height=target_h, fps=int(pipeline["fps_target"])))
        vcam.start()

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.warning("Frame capture failed")
                continue

            req_id = int(time.time() * 1000) % 1_000_000
            frame_b64 = encode_image_to_b64(frame)

            detection = rpc.call("face_detector", {"frame_b64": frame_b64}, req_id)
            boxes = detection.get("boxes", [])
            if not boxes:
                if vcam:
                    vcam.send(frame)
                cv2.imshow("AFS", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            perturb_res = rpc.call(
                "perturbation_generator",
                {"face_b64": encode_image_to_b64(face_crop)},
                req_id + 1,
            )

            shielded = rpc.call(
                "frame_blender",
                {
                    "frame_b64": frame_b64,
                    "perturbation_b64": perturb_res["perturbation_b64"],
                    "boxes": boxes,
                    "alpha": alpha_controller.alpha,
                },
                req_id + 2,
            )
            shielded_frame = decode_image_from_b64(shielded["shielded_frame_b64"])

            frame_idx += 1
            if feedback.get("enabled") and frame_idx % int(pipeline["feedback_interval_frames"]) == 0:
                fb = rpc.call("deepfake_feedback", {"frame_b64": encode_image_to_b64(shielded_frame)}, req_id + 3)
                alpha_controller.update(float(fb["confidence"]))

            if vcam:
                vcam.send(shielded_frame)

            cv2.putText(
                shielded_frame,
                f"alpha={alpha_controller.alpha:.3f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("AFS", shielded_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if vcam:
            vcam.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config["logging"]["level"], config["logging"]["output"])
    run_capture_loop(config, args.source, args.rpc_endpoint, args.virtual_cam)


if __name__ == "__main__":
    main()
