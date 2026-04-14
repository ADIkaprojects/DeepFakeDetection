from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.types import BoundingBox


@dataclass(slots=True)
class BlendConfig:
    alpha: float = 0.12


class FrameBlender:
    def __init__(self, config: BlendConfig) -> None:
        self._config = config

    def blend(self, frame: np.ndarray, perturbation: np.ndarray, boxes: list[BoundingBox], alpha: float | None = None) -> np.ndarray:
        out = frame.copy()
        blend_alpha = self._config.alpha if alpha is None else alpha

        for box in boxes:
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            if x2 <= x1 or y2 <= y1:
                continue

            roi = out[y1:y2, x1:x2]
            pert = cv2.resize(perturbation, (roi.shape[1], roi.shape[0]))
            mixed = cv2.addWeighted(roi, 1.0, pert, blend_alpha, 0.0)
            mixed = cv2.GaussianBlur(mixed, (3, 3), 0)
            out[y1:y2, x1:x2] = mixed

        return np.clip(out, 0, 255).astype(np.uint8)
