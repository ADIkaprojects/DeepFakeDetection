from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class TimerResult:
    elapsed_ms: float = 0.0


@contextmanager
def timer() -> Iterator[TimerResult]:
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.elapsed_ms = (end - start) * 1000.0
