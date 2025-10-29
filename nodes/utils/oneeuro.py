"""One Euro filter utilities for stabilizing parameter sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class OneEuroSettings:
    """Configuration parameters for the One Euro filter."""

    frequency: float
    min_cutoff: float
    beta: float
    derivative_cutoff: float = 1.0


class OneEuroFilter:
    """Vector-friendly One Euro filter implementation."""

    def __init__(
        self,
        frequency: float,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.settings = OneEuroSettings(
            frequency=max(frequency, 1e-2),
            min_cutoff=max(min_cutoff, 1e-4),
            beta=max(beta, 0.0),
            derivative_cutoff=max(derivative_cutoff, 1e-4),
        )
        self._previous_signal: Optional[np.ndarray] = None
        self._previous_derivative: Optional[np.ndarray] = None

    def _alpha(self, cutoff: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.settings.frequency
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        """Reset internal state."""
        self._previous_signal = None
        self._previous_derivative = None

    def filter(self, value: Iterable[float] | np.ndarray) -> np.ndarray:
        """Filter the incoming value."""
        signal = np.asarray(value, dtype=np.float64)

        if self._previous_signal is None:
            self._previous_signal = signal
            self._previous_derivative = np.zeros_like(signal)
            return signal

        derivative = (signal - self._previous_signal) * self.settings.frequency
        alpha_d = self._alpha(self.settings.derivative_cutoff)
        derivative_hat = alpha_d * derivative + (1.0 - alpha_d) * self._previous_derivative

        cutoff = self.settings.min_cutoff + self.settings.beta * np.abs(derivative_hat)
        alpha = self._alpha(cutoff)
        filtered = alpha * signal + (1.0 - alpha) * self._previous_signal

        self._previous_signal = filtered
        self._previous_derivative = derivative_hat
        return filtered

    __call__ = filter

