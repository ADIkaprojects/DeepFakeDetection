import numpy as np

from src.blending.frame_blender import BlendConfig, FrameBlender
from src.utils.types import BoundingBox


def test_blend_changes_roi_pixels() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    perturbation = np.full((20, 20, 3), 255, dtype=np.uint8)

    blender = FrameBlender(BlendConfig(alpha=0.5))
    out = blender.blend(frame, perturbation, [BoundingBox(10, 10, 30, 30)])

    assert out[15, 15].sum() > 0
