"""Layer 4: Information Theory & Signal Processing Engine.

Quantifies informational throughput of the network (mutual information
between an input and output node's trajectories), noise-buffering
(Fano factor / SNR), and temporal delay/gain, comparing simulations with
LLPS condensate kinetics enabled vs. disabled.

The LLPS-on/off comparison, the MI-vs-noise sweep, and the multi-realization
Monte Carlo average each require multiple independent simulations; all
three fan out across the shared ``ProcessPoolExecutor`` (via
``utils.parallel_worker.run_parallel``) instead of running sequentially or
on a QThread, since QThread cannot free CPU-bound SciPy integration from
the GIL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.layer3_kinetics import SimulationResult, simulate_task
from utils.parallel_worker import ProgressCB, run_parallel

log = logging.getLogger("phasenet.layer4")


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    """Binned-histogram mutual information I(X;Y) in bits."""
    if len(x) < 2 or len(y) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    joint, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    joint_p = joint / joint.sum()
    x_p = joint_p.sum(axis=1)
    y_p = joint_p.sum(axis=0)

    mi = 0.0
    for i in range(joint_p.shape[0]):
        for j in range(joint_p.shape[1]):
            if joint_p[i, j] > 0 and x_p[i] > 0 and y_p[j] > 0:
                mi += joint_p[i, j] * np.log2(joint_p[i, j] / (x_p[i] * y_p[j]))
    return max(0.0, float(mi))


def fano_factor(y: np.ndarray) -> float:
    mean = np.mean(y)
    if mean <= 1e-12:
        return 0.0
    return float(np.var(y) / mean)


def snr(y: np.ndarray) -> float:
    mean = np.mean(y)
    if mean <= 1e-12:
        return 0.0
    return float((mean ** 2) / np.var(y)) if np.var(y) > 0 else float("inf")


def temporal_delay(times: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Lag (time units) at peak cross-correlation between input and output."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    corr = np.correlate(y_c, x_c, mode="full")
    lags = np.arange(-len(x) + 1, len(x))
    best_lag = lags[np.argmax(corr)]
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    return float(best_lag * dt)


def gain(x: np.ndarray, y: np.ndarray) -> float:
    x_amp = np.max(x) - np.min(x)
    if x_amp <= 1e-12:
        return 0.0
    y_amp = np.max(y) - np.min(y)
    return float(y_amp / x_amp)


@dataclass
class InfoMetrics:
    mutual_information: float
    fano_factor: float
    snr: float
    delay: float
    gain: float


def compute_information_metrics(result: SimulationResult, input_node: str, output_node: str,
                                  bins: int = 16) -> InfoMetrics:
    x = result.trajectories[input_node]
    y = result.trajectories[output_node]
    return InfoMetrics(
        mutual_information=mutual_information(x, y, bins=bins),
        fano_factor=fano_factor(y),
        snr=snr(y),
        delay=temporal_delay(result.times, x, y),
        gain=gain(x, y),
    )


def compare_llps_effect(G, input_nodes: List[str], output_nodes: List[str],
                          gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                          noise_amplitude: float = 0.1, signal_kind: str = "step", amplitude: float = 1.0,
                          bins: int = 16, progress_cb: ProgressCB = None,
                          max_workers: Optional[int] = None) -> Dict[str, Dict]:
    """Run with and without LLPS kinetics (in parallel) and compare information metrics."""
    common = dict(gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time,
                  noise_amplitude=noise_amplitude, signal_kind=signal_kind, amplitude=amplitude)
    tasks = [
        ((G, input_nodes, output_nodes), {**common, "llps_enabled": True, "seed": 0}),
        ((G, input_nodes, output_nodes), {**common, "llps_enabled": False, "seed": 0}),
    ]
    result_on, result_off = run_parallel(simulate_task, tasks, progress_cb=progress_cb,
                                          max_workers=max_workers, label="LLPS comparison")

    out_node = output_nodes[0]
    in_node = input_nodes[0]
    metrics_on = compute_information_metrics(result_on, in_node, out_node, bins=bins)
    metrics_off = compute_information_metrics(result_off, in_node, out_node, bins=bins)

    return {
        "with_llps": metrics_on.__dict__,
        "without_llps": metrics_off.__dict__,
        "result_with_llps": result_on,
        "result_without_llps": result_off,
    }


def mi_vs_noise_curve(G, input_nodes: List[str], output_nodes: List[str], noise_levels,
                        gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                        bins: int = 16, progress_cb: ProgressCB = None,
                        max_workers: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Sweep noise amplitude (in parallel, one process per level) for the Panel B plot."""
    tasks = [
        ((G, input_nodes, output_nodes), dict(gamma=gamma, kpart_threshold=kpart_threshold,
         sim_time=sim_time, noise_amplitude=float(level), signal_kind="gaussian", amplitude=1.0,
         llps_enabled=True, seed=i))
        for i, level in enumerate(noise_levels)
    ]
    results = run_parallel(simulate_task, tasks, progress_cb=progress_cb,
                            max_workers=max_workers, label="noise level")

    mi_values = [
        compute_information_metrics(r, input_nodes[0], output_nodes[0], bins=bins).mutual_information
        if r is not None else 0.0
        for r in results
    ]
    return {"noise_levels": np.array(noise_levels), "mutual_information": np.array(mi_values)}


def parallel_mi_realizations(G, input_nodes: List[str], output_nodes: List[str],
                               gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                               noise_amplitude: float = 0.2, n_realizations: int = 8,
                               signal_kind: str = "gaussian", amplitude: float = 1.0,
                               llps_enabled: bool = True, bins: int = 16,
                               progress_cb: ProgressCB = None, max_workers: Optional[int] = None) -> Dict:
    """Pool multiple independent stochastic noise realizations into one MI/Fano estimate.

    Each realization is a full simulation with its own RNG seed, run as a
    separate process-pool task; pooling their trajectories before computing
    mutual information reduces single-run estimator variance.
    """
    tasks = [
        ((G, input_nodes, output_nodes), dict(gamma=gamma, kpart_threshold=kpart_threshold,
         sim_time=sim_time, noise_amplitude=noise_amplitude, signal_kind=signal_kind,
         amplitude=amplitude, llps_enabled=llps_enabled, seed=seed))
        for seed in range(n_realizations)
    ]
    results: List[SimulationResult] = run_parallel(simulate_task, tasks, progress_cb=progress_cb,
                                                     max_workers=max_workers, label="realization")
    results = [r for r in results if r is not None]
    if not results:
        return {"mutual_information": 0.0, "fano_factor": 0.0, "n_realizations": 0, "realizations": []}

    all_x = np.concatenate([r.trajectories[input_nodes[0]] for r in results])
    all_y = np.concatenate([r.trajectories[output_nodes[0]] for r in results])

    return {
        "mutual_information": mutual_information(all_x, all_y, bins=bins),
        "fano_factor": float(np.mean([fano_factor(r.trajectories[output_nodes[0]]) for r in results])),
        "n_realizations": len(results),
        "realizations": results,
    }
