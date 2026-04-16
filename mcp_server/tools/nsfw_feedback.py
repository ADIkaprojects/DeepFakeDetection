from __future__ import annotations

import logging
import threading
from typing import Any

from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble, build_nsfw_feedback_engine

LOGGER = logging.getLogger("afs.mcp_nsfw_feedback")

_ENGINE: NSFWProxyEnsemble | None = None
_LOCK = threading.Lock()


def _get_nsfw_engine(config: dict[str, Any]) -> NSFWProxyEnsemble:
    """Return singleton NSFW feedback ensemble initialized from runtime config."""
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = build_nsfw_feedback_engine(config)
    return _ENGINE


def handle_nsfw_feedback(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Score base64 frame with NSFW ensemble and return score payload."""
    frame_b64 = payload.get("frame_b64")
    if not isinstance(frame_b64, str) or not frame_b64:
        return {"error": "missing_frame_b64", "nsfw_score": None, "label": None}

    try:
        return _get_nsfw_engine(config).score_b64(frame_b64)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("nsfw_feedback failed")
        return {"error": str(exc), "nsfw_score": None, "label": None}
