"""Layer 4: Information Theory & Signal Processing Engine.

Quantifies informational throughput of the network (mutual information
between an input and output node's trajectories), noise-buffering
(Fano factor / SNR), and temporal delay/gain, comparing simulations with
LLPS condensate kinetics enabled vs. disabled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from core.layer3_kinetics import KineticsSimulator, SimulationResult

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


def compare_llps_effect(simulator: KineticsSimulator, input_nodes, output_nodes,
                          signal_kind: str = "step", amplitude: float = 1.0,
                          bins: int = 16) -> Dict[str, Dict]:
    """Run with and without LLPS kinetics and compare information metrics."""
    result_on = simulator.simulate(input_nodes, output_nodes, signal_kind=signal_kind,
                                    amplitude=amplitude, llps_enabled=True)
    result_off = simulator.simulate(input_nodes, output_nodes, signal_kind=signal_kind,
                                     amplitude=amplitude, llps_enabled=False)

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


def mi_vs_noise_curve(simulator: KineticsSimulator, input_nodes, output_nodes,
                        noise_levels, bins: int = 16) -> Dict[str, np.ndarray]:
    """Sweep noise amplitude and record I(X;Y) for the Panel B workbench plot."""
    mi_values = []
    for noise in noise_levels:
        simulator.noise_amplitude = noise
        result = simulator.simulate(input_nodes, output_nodes, signal_kind="gaussian",
                                     amplitude=1.0, llps_enabled=True)
        metrics = compute_information_metrics(result, input_nodes[0], output_nodes[0], bins=bins)
        mi_values.append(metrics.mutual_information)
    return {"noise_levels": np.array(noise_levels), "mutual_information": np.array(mi_values)}
