from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class AlphaControllerConfig:
    initial: float
    min_alpha: float
    max_alpha: float
    increase_step: float
    decay_rate: float
    high_threshold: float
    low_threshold: float


class AlphaController:
    def __init__(self, config: AlphaControllerConfig) -> None:
        self._config = config
        self._alpha = config.initial
        self._history: deque[float] = deque(maxlen=10)

    @property
    def alpha(self) -> float:
        return self._alpha

    def update(self, confidence: float) -> float:
        self._history.append(confidence)
        avg = sum(self._history) / len(self._history)

        if avg > self._config.high_threshold:
            self._alpha = min(self._config.max_alpha, self._alpha + self._config.increase_step)
        elif avg < self._config.low_threshold:
            self._alpha = max(self._config.min_alpha, self._alpha - self._config.decay_rate)

        return self._alpha
