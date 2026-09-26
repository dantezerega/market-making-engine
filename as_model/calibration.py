"""Parameter estimation for the Avellaneda-Stoikov model, run directly
against this project's own LOB simulator output rather than assumed.

sigma: estimated from the realized mid-price increments of a background
       (no-MM) simulation run — dS ~ N(0, sigma^2 * dt) under the model's
       arithmetic-BM assumption, so sigma = std(diff(mid)) / sqrt(dt).

k, A:  estimated by regressing the empirical fill-intensity-vs-distance
       relationship observed in the simulator's own trade tape. For every
       trade in a background run we compute the distance between the
       trade price and the prevailing fundamental/mid price just before
       the trade. Binning those distances and counting trades per unit
       time per bin gives an empirical lambda(delta). Since
       lambda(delta) = A * exp(-k * delta), taking logs makes this a
       linear regression:  ln(lambda) = ln(A) - k * delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import optimize, stats


@dataclass
class CalibrationResult:
    sigma: float
    k: float
    A: float
    k_r_squared: float
    n_trades_used: int
    distance_bins: np.ndarray
    empirical_intensity: np.ndarray


def estimate_sigma(mid_price_path: Sequence[float], dt: float) -> float:
    """sigma = std(price increments) / sqrt(dt), under dS = sigma dW."""
    prices = np.asarray(mid_price_path, dtype=float)
    increments = np.diff(prices)
    increments = increments[np.isfinite(increments)]
    if len(increments) < 2:
        raise ValueError("Not enough price observations to estimate sigma.")
    return float(np.std(increments, ddof=1) / np.sqrt(dt))


def calibrate_k_A(
    trade_distances: Sequence[float],
    trade_window_seconds: float,
    n_bins: int = 20,
    min_count_per_bin: int = 3,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit lambda(delta) = A * exp(-k * delta) to empirical trade flow.

    trade_distances: |trade_price - reference_mid_at_trade_time| for every
        trade observed in a background (no-MM) run.
    trade_window_seconds: total elapsed simulated time the trades were
        drawn from (n_ticks * dt), used to turn counts into intensities
        (trades per unit time).

    Returns (k, A, r_squared, bin_centers, empirical_intensity).
    """
    distances = np.asarray(trade_distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances >= 0)]
    if len(distances) < min_count_per_bin * 2:
        raise ValueError("Not enough trades to calibrate k/A.")

    max_dist = np.percentile(distances, 95)  # trim extreme outlier tail
    edges = np.linspace(0.0, max(max_dist, 1e-6), n_bins + 1)
    counts, _ = np.histogram(distances, bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    mask = counts >= min_count_per_bin
    if mask.sum() < 3:
        # relax if too few bins qualify
        mask = counts > 0
    intensity = counts[mask] / trade_window_seconds
    centers = bin_centers[mask]

    log_intensity = np.log(intensity)
    slope, intercept, r_value, _, _ = stats.linregress(centers, log_intensity)
    k = float(-slope)
    A = float(np.exp(intercept))

    if k <= 0 or not np.isfinite(k):
        # Fall back to nonlinear least squares directly on lambda(delta) if
        # the log-linear regression produced a degenerate slope (e.g. too
        # few bins / noisy tail).
        def model(d, A_, k_):
            return A_ * np.exp(-k_ * d)

        popt, _ = optimize.curve_fit(
            model, centers, intensity, p0=[max(intensity.max(), 1e-3), 1.0], maxfev=10000
        )
        A, k = float(popt[0]), float(popt[1])
        r_value = np.nan

    return k, A, float(r_value ** 2) if np.isfinite(r_value) else float("nan"), bin_centers, np.where(mask, counts / trade_window_seconds, np.nan)
