"""Information Thermodynamics Analyzer.

Estimates Shannon-entropy-based information-processing metrics directly
from simulated time series: mutual information I(X;Y) via a
Kraskov-Stogbauer-Grassberger (KSG) k-nearest-neighbor estimator built on
`scipy.spatial.cKDTree` (with a histogram/binned fallback built on
`sklearn.metrics.mutual_info_score` for small or degenerate samples), and
transfer entropy T_{X->Y} = I(Y_t ; X_{t-1} | Y_{t-1}) via the Frenzel-Pompe
conditional-MI generalization of the same KSG estimator.

All estimates are reported in bits. Discrete/quantized inputs (e.g. integer
event counts from the micro agent engine) are dequantized with a small
jitter before estimation, since the continuous KSG estimator is undefined
when nearest-neighbor distances collapse to exactly zero.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.metrics import mutual_info_score

from config import HISTOGRAM_BINS, KSG_K_NEIGHBORS, TE_LAG

log = logging.getLogger("biocompute.info_thermodynamics")

_LOG2 = np.log(2.0)


def _dequantize(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small continuous jitter to break exact ties in quantized data.

    The KSG estimator is undefined when the k-th nearest-neighbor distance
    is exactly zero (which happens routinely for integer event-count time
    series, e.g. a cumulative activation counter that plateaus for many
    steps). Jitter magnitude is a small fraction of the data's own spread,
    small enough not to distort the estimate, large enough to resolve ties.
    """
    spread = np.std(values)
    scale = spread * 1e-4 if spread > 0 else 1e-9
    return values + rng.normal(0.0, scale, size=values.shape)


def _marginal_neighbor_counts(points: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Count neighbors of each point strictly within its own epsilon radius
    (Chebyshev/max norm), excluding the point itself.
    """
    tree = cKDTree(points)
    n = len(points)
    counts = np.empty(n)
    for i in range(n):
        radius = max(epsilon[i] - 1e-12, 0.0)
        idx = tree.query_ball_point(points[i], r=radius, p=np.inf)
        counts[i] = len(idx) - 1  # exclude self
    return counts


def mutual_information_ksg(x: np.ndarray, y: np.ndarray, k: int = KSG_K_NEIGHBORS,
                             rng: Optional[np.random.Generator] = None) -> float:
    """KSG-1 mutual information estimator I(X;Y), in bits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= k + 1:
        raise ValueError(f"Need more than k+1={k + 1} samples for KSG estimation, got {n}.")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0

    rng = rng or np.random.default_rng(0)
    x_j = _dequantize(x, rng).reshape(-1, 1)
    y_j = _dequantize(y, rng).reshape(-1, 1)

    joint = np.hstack([x_j, y_j])
    tree = cKDTree(joint)
    dist, _ = tree.query(joint, k=k + 1, p=np.inf)
    epsilon = dist[:, k]

    n_x = _marginal_neighbor_counts(x_j, epsilon)
    n_y = _marginal_neighbor_counts(y_j, epsilon)

    mi_nats = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(0.0, float(mi_nats / _LOG2))


def mutual_information_histogram(x: np.ndarray, y: np.ndarray, bins: int = HISTOGRAM_BINS) -> float:
    """Binned mutual information fallback, via sklearn's discrete MI estimator."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    x_edges = np.histogram_bin_edges(x, bins=bins)
    y_edges = np.histogram_bin_edges(y, bins=bins)
    x_binned = np.digitize(x, x_edges[1:-1])
    y_binned = np.digitize(y, y_edges[1:-1])
    mi_nats = mutual_info_score(x_binned, y_binned)
    return max(0.0, float(mi_nats / _LOG2))


def mutual_information(x: np.ndarray, y: np.ndarray, k: int = KSG_K_NEIGHBORS) -> float:
    """KSG estimator with an automatic histogram fallback for small/degenerate samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= k + 1:
        return mutual_information_histogram(x, y)
    try:
        value = mutual_information_ksg(x, y, k=k)
        if not np.isfinite(value):
            raise FloatingPointError("KSG estimator returned a non-finite value.")
        return value
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        log.warning("KSG mutual information estimation failed (%s); falling back to histogram.", exc)
        return mutual_information_histogram(x, y)


def _conditional_mutual_information_ksg(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                                          k: int = KSG_K_NEIGHBORS,
                                          rng: Optional[np.random.Generator] = None) -> float:
    """Frenzel-Pompe KSG estimator for the conditional mutual information I(A;B|C), in bits."""
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    n = len(a)
    if n <= k + 1:
        raise ValueError(f"Need more than k+1={k + 1} samples for conditional KSG estimation, got {n}.")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0

    rng = rng or np.random.default_rng(0)
    a_j = _dequantize(a, rng).reshape(-1, 1)
    b_j = _dequantize(b, rng).reshape(-1, 1)
    c_j = _dequantize(c, rng).reshape(-1, 1)

    joint = np.hstack([a_j, b_j, c_j])
    tree = cKDTree(joint)
    dist, _ = tree.query(joint, k=k + 1, p=np.inf)
    epsilon = dist[:, k]

    n_ac = _marginal_neighbor_counts(np.hstack([a_j, c_j]), epsilon)
    n_bc = _marginal_neighbor_counts(np.hstack([b_j, c_j]), epsilon)
    n_c = _marginal_neighbor_counts(c_j, epsilon)

    cmi_nats = digamma(k) - np.mean(digamma(n_ac + 1) + digamma(n_bc + 1) - digamma(n_c + 1))
    return max(0.0, float(cmi_nats / _LOG2))


def transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = TE_LAG,
                       k: int = KSG_K_NEIGHBORS) -> float:
    """Transfer entropy T_{X->Y} = I(Y_t ; X_{t-lag} | Y_{t-lag}), in bits.

    Quantifies the directed information flow from source X into target Y,
    beyond what Y's own past already predicts about its future.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) <= lag + k + 1:
        log.warning("Time series too short (%d samples) for lag=%d, k=%d transfer entropy; returning 0.",
                    len(x), lag, k)
        return 0.0

    y_future = y[lag:]
    x_past = x[:-lag]
    y_past = y[:-lag]

    try:
        value = _conditional_mutual_information_ksg(y_future, x_past, y_past, k=k)
        if not np.isfinite(value):
            raise FloatingPointError("Conditional KSG estimator returned a non-finite value.")
        return value
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        log.warning("Transfer entropy estimation failed (%s); returning 0.0.", exc)
        return 0.0


def noise_ratio(x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of output variance NOT explained by a linear fit to the input: 1 - r^2.

    A complementary, scale-free noise metric alongside the (nonlinear) KSG
    mutual information: 0 means Y is perfectly linearly predictable from X,
    1 means X carries no linear information about Y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 1.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 1.0
    return float(np.clip(1.0 - corr ** 2, 0.0, 1.0))


def compute_information_metrics(x: np.ndarray, y: np.ndarray, k: int = KSG_K_NEIGHBORS,
                                   lag: int = TE_LAG) -> Dict[str, float]:
    """Full information-thermodynamic readout for an (input, output) time-series pair."""
    return {
        "mutual_info": mutual_information(x, y, k=k),
        "transfer_entropy": transfer_entropy(x, y, lag=lag, k=k),
        "noise_ratio": noise_ratio(x, y),
    }
