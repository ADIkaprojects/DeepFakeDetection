from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width - 1)),
            y2=max(0, min(self.y2, height - 1)),
        )

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class DetectionResult:
    boxes: list[BoundingBox]
    landmarks: list[list[list[float]]]
    error: str | None = None


JSONDict = dict[str, Any]
