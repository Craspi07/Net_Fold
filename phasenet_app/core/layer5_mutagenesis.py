"""Layer 5: In-Silico Mutagenesis & Sensitivity Dashboard.

Systematic node knockouts and valency (S_LLPS-lowering) mutagenesis to
observe information decay, plus a Sobol/Latin-Hypercube sensitivity
matrix over (K_part threshold, gamma).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.stats import qmc

from core.layer3_kinetics import KineticsSimulator
from core.layer4_infotheory import compute_information_metrics

log = logging.getLogger("phasenet.layer5")
ProgressCB = Optional[Callable[[int, str], None]]


def knockout_node(G: nx.Graph, node: str) -> nx.Graph:
    H = G.copy()
    if node in H:
        H.remove_node(node)
    return H


def deletion_scan(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                   gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                   progress_cb: ProgressCB = None) -> Dict[str, float]:
    """Knock out each non-input/output node and record I(X;Y) drop."""
    baseline_sim = KineticsSimulator(G, gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time)
    baseline_result = baseline_sim.simulate(input_nodes, output_nodes, llps_enabled=True)
    baseline_mi = compute_information_metrics(baseline_result, input_nodes[0], output_nodes[0]).mutual_information

    candidates = [n for n in G.nodes() if n not in input_nodes and n not in output_nodes]
    drop_map: Dict[str, float] = {}
    total = max(1, len(candidates))

    for i, node in enumerate(candidates):
        H = knockout_node(G, node)

        # Knocking out a node that disconnects input from output collapses
        # I(X;Y) to zero (full information decay) - no need to simulate.
        if input_nodes[0] not in H or output_nodes[0] not in H or not nx.has_path(
            H, input_nodes[0], output_nodes[0]
        ):
            drop_map[node] = baseline_mi
            if progress_cb:
                progress_cb(int(100 * (i + 1) / total), f"Knockout scan: {node}")
            continue

        sim = KineticsSimulator(H, gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time)
        result = sim.simulate(input_nodes, output_nodes, llps_enabled=True)
        mi = compute_information_metrics(result, input_nodes[0], output_nodes[0]).mutual_information
        drop_map[node] = float(baseline_mi - mi)

        if progress_cb:
            progress_cb(int(100 * (i + 1) / total), f"Knockout scan: {node}")

    return drop_map


def valency_mutagenesis(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                          delta_v: float = 0.5, gamma: float = 2.0, kpart_threshold: float = 0.5,
                          sim_time: float = 50.0, progress_cb: ProgressCB = None) -> Dict[str, float]:
    """Lower S_LLPS by delta_v (loss-of-disorder mutation) per node, holding k_cat fixed."""
    baseline_sim = KineticsSimulator(G, gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time)
    baseline_result = baseline_sim.simulate(input_nodes, output_nodes, llps_enabled=True)
    baseline_mi = compute_information_metrics(baseline_result, input_nodes[0], output_nodes[0]).mutual_information

    candidates = [n for n in G.nodes() if G.nodes[n].get("s_llps", 0.0) > kpart_threshold]
    drop_map: Dict[str, float] = {}
    total = max(1, len(candidates))

    for i, node in enumerate(candidates):
        H = G.copy()
        H.nodes[node]["s_llps"] = max(0.0, H.nodes[node].get("s_llps", 0.0) - delta_v)
        sim = KineticsSimulator(H, gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time)
        result = sim.simulate(input_nodes, output_nodes, llps_enabled=True)
        mi = compute_information_metrics(result, input_nodes[0], output_nodes[0]).mutual_information
        drop_map[node] = float(baseline_mi - mi)

        if progress_cb:
            progress_cb(int(100 * (i + 1) / total), f"Valency mutagenesis: {node}")

    return drop_map


def sensitivity_matrix(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                         kpart_range: Tuple[float, float] = (0.1, 0.9),
                         gamma_range: Tuple[float, float] = (0.5, 5.0),
                         n_samples: int = 32, sim_time: float = 30.0,
                         progress_cb: ProgressCB = None) -> Dict[str, np.ndarray]:
    """Latin Hypercube sample over (K_part threshold, gamma) -> I(X;Y) surface."""
    sampler = qmc.LatinHypercube(d=2, seed=7)
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample, [kpart_range[0], gamma_range[0]], [kpart_range[1], gamma_range[1]])

    thresholds = scaled[:, 0]
    gammas = scaled[:, 1]
    mi_values = np.zeros(n_samples)

    for i in range(n_samples):
        sim = KineticsSimulator(G, gamma=float(gammas[i]), kpart_threshold=float(thresholds[i]),
                                 sim_time=sim_time)
        result = sim.simulate(input_nodes, output_nodes, llps_enabled=True)
        mi_values[i] = compute_information_metrics(result, input_nodes[0], output_nodes[0]).mutual_information
        if progress_cb:
            progress_cb(int(100 * (i + 1) / n_samples), f"Sensitivity sample {i + 1}/{n_samples}")

    return {
        "kpart_thresholds": thresholds,
        "gammas": gammas,
        "mutual_information": mi_values,
    }
