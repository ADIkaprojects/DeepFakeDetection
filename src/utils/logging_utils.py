from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "component": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "latency_ms"):
            payload["latency_ms"] = getattr(record, "latency_ms")
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str, output_file: str | None = None) -> None:
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)

    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
