from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(slots=True)
class RateLimitConfig:
    requests_per_window: int
    window_seconds: int


class SlidingWindowRateLimiter:
    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._windows[key]

        while q and (now - q[0]) > self._config.window_seconds:
            q.popleft()

        if len(q) >= self._config.requests_per_window:
            return False

        q.append(now)
        return True
