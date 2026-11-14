from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple causal-ish centered moving average for demo purposes."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """First derivative via central differences; edge uses one-sided."""
    x = np.asarray(x, dtype=float)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    dx = np.empty_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def bandpass_rms(
    x: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
) -> float:
    """Approximate band energy via FFT magnitude mask (demo-quality)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return float(np.sqrt(np.mean(x**2)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.fft.rfft(x - np.mean(x))
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    band = spec.copy()
    band[~mask] = 0.0
    recon = np.fft.irfft(band, n=n)
    return float(np.sqrt(np.mean(recon**2)))
