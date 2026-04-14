from __future__ import annotations

from dataclasses import dataclass

import cv2

try:
    import pyvirtualcam
except Exception:  # pragma: no cover
    pyvirtualcam = None


@dataclass(slots=True)
class VirtualCamConfig:
    width: int
    height: int
    fps: int


class VirtualCamDriver:
    def __init__(self, config: VirtualCamConfig) -> None:
        self._config = config
        self._cam = None

    def start(self) -> None:
        if pyvirtualcam is None:
            raise RuntimeError("pyvirtualcam is not installed")
        self._cam = pyvirtualcam.Camera(
            width=self._config.width,
            height=self._config.height,
            fps=self._config.fps,
        )

    def send(self, frame_bgr) -> None:
        if self._cam is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._cam.send(frame_rgb)
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None
