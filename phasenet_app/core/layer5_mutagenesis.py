"""Layer 5: In-Silico Mutagenesis & Sensitivity Dashboard.

Systematic node knockouts and valency (S_LLPS-lowering) mutagenesis to
observe information decay, plus a Latin-Hypercube sensitivity matrix over
(K_part threshold, gamma). Every node/sample in these scans requires its
own full kinetic simulation + MI calculation, so each is dispatched as an
independent task to the shared ``ProcessPoolExecutor`` via
``utils.parallel_worker.run_parallel`` - these scans are embarrassingly
parallel and would otherwise dominate runtime if run sequentially.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.stats import qmc

from core.layer3_kinetics import simulate_task
from core.layer4_infotheory import compute_information_metrics
from utils.parallel_worker import ProgressCB, run_parallel

log = logging.getLogger("phasenet.layer5")


def knockout_node(G: nx.Graph, node: str) -> nx.Graph:
    H = G.copy()
    if node in H:
        H.remove_node(node)
    return H


def _mi_from_simulation(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                          gamma: float, kpart_threshold: float, sim_time: float) -> float:
    """Run one simulation and return I(X;Y); used as a single process-pool task unit."""
    if input_nodes[0] not in G or output_nodes[0] not in G or not nx.has_path(G, input_nodes[0], output_nodes[0]):
        return 0.0
    result = simulate_task(G, input_nodes, output_nodes, gamma=gamma, kpart_threshold=kpart_threshold,
                            sim_time=sim_time, llps_enabled=True, seed=0)
    return compute_information_metrics(result, input_nodes[0], output_nodes[0]).mutual_information


def deletion_scan(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                   gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                   progress_cb: ProgressCB = None, max_workers: Optional[int] = None) -> Dict[str, float]:
    """Knock out each non-input/output node (in parallel) and record I(X;Y) drop."""
    baseline_mi = _mi_from_simulation(G, input_nodes, output_nodes, gamma, kpart_threshold, sim_time)

    candidates = [n for n in G.nodes() if n not in input_nodes and n not in output_nodes]
    knocked_graphs = [knockout_node(G, node) for node in candidates]

    tasks = [
        ((H, input_nodes, output_nodes, gamma, kpart_threshold, sim_time), {})
        for H in knocked_graphs
    ]
    mi_values = run_parallel(_mi_from_simulation, tasks, progress_cb=progress_cb,
                              max_workers=max_workers, label="knockout")

    return {
        node: float(baseline_mi - (mi if mi is not None else 0.0))
        for node, mi in zip(candidates, mi_values)
    }


def _lower_valency(G: nx.Graph, node: str, delta_v: float) -> nx.Graph:
    H = G.copy()
    H.nodes[node]["s_llps"] = max(0.0, H.nodes[node].get("s_llps", 0.0) - delta_v)
    return H


def valency_mutagenesis(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                          delta_v: float = 0.5, gamma: float = 2.0, kpart_threshold: float = 0.5,
                          sim_time: float = 50.0, progress_cb: ProgressCB = None,
                          max_workers: Optional[int] = None) -> Dict[str, float]:
    """Lower S_LLPS by delta_v (loss-of-disorder mutation) per node, holding k_cat fixed.

    Tests whether functional loss on knockdown is spatial (LLPS-mediated)
    rather than catalytic, since the underlying reaction rate is untouched.
    """
    baseline_mi = _mi_from_simulation(G, input_nodes, output_nodes, gamma, kpart_threshold, sim_time)

    candidates = [n for n in G.nodes() if G.nodes[n].get("s_llps", 0.0) > kpart_threshold]
    mutated_graphs = [_lower_valency(G, node, delta_v) for node in candidates]

    tasks = [
        ((H, input_nodes, output_nodes, gamma, kpart_threshold, sim_time), {})
        for H in mutated_graphs
    ]
    mi_values = run_parallel(_mi_from_simulation, tasks, progress_cb=progress_cb,
                              max_workers=max_workers, label="valency mutant")

    return {
        node: float(baseline_mi - (mi if mi is not None else 0.0))
        for node, mi in zip(candidates, mi_values)
    }


def sensitivity_matrix(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                         kpart_range: Tuple[float, float] = (0.1, 0.9),
                         gamma_range: Tuple[float, float] = (0.5, 5.0),
                         n_samples: int = 32, sim_time: float = 30.0,
                         progress_cb: ProgressCB = None, max_workers: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Latin Hypercube sample over (K_part threshold, gamma) -> I(X;Y) surface, in parallel."""
    sampler = qmc.LatinHypercube(d=2, seed=7)
    sample = sampler.random(n=n_samples)
    scaled = qmc.scale(sample, [kpart_range[0], gamma_range[0]], [kpart_range[1], gamma_range[1]])

    thresholds = scaled[:, 0]
    gammas = scaled[:, 1]

    tasks = [
        ((G, input_nodes, output_nodes, float(gammas[i]), float(thresholds[i]), sim_time), {})
        for i in range(n_samples)
    ]
    mi_values = run_parallel(_mi_from_simulation, tasks, progress_cb=progress_cb,
                              max_workers=max_workers, label="sensitivity sample")
    mi_values = np.array([v if v is not None else 0.0 for v in mi_values])

    return {
        "kpart_thresholds": thresholds,
        "gammas": gammas,
        "mutual_information": mi_values,
    }
