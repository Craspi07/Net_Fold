"""Layer 1: Network Topology & Node Criticality.

Computes centrality metrics, global network efficiency vulnerability upon
single-node removal, and classifies nodes into topological roles.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import networkx as nx
import numpy as np

log = logging.getLogger("phasenet.layer1")
ProgressCB = Optional[Callable[[int, str], None]]


def global_efficiency(G: nx.Graph) -> float:
    """E_global = 1/(N(N-1)) * sum_{i!=j} 1/d_ij over all node pairs."""
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    total = 0.0
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i, dists in lengths.items():
        for j, d in dists.items():
            if i != j and d > 0:
                total += 1.0 / d
    return total / (n * (n - 1))


def compute_vulnerability(G: nx.Graph, max_nodes: int = 200, progress_cb: ProgressCB = None) -> Dict[str, float]:
    """Delta E_global for removal of each node.

    For large graphs (>max_nodes) this is sampled on the highest-degree
    nodes to keep runtime bounded, since it is O(N) global-efficiency
    recomputations, each O(N^2).
    """
    baseline = global_efficiency(G)
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = sorted(nodes, key=lambda n: G.degree(n), reverse=True)[:max_nodes]

    vulnerability: Dict[str, float] = {n: 0.0 for n in G.nodes()}
    total = len(nodes)
    for i, node in enumerate(nodes):
        H = G.copy()
        H.remove_node(node)
        e = global_efficiency(H)
        vulnerability[node] = baseline - e
        if progress_cb and total:
            progress_cb(int(100 * (i + 1) / total), f"Vulnerability: {node}")
    return vulnerability


def classify_roles(centrality: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Classify each node as Hub / Bottleneck / Periphery via percentile cuts."""
    nodes = list(centrality.keys())
    if not nodes:
        return {}
    degrees = np.array([centrality[n]["degree_centrality"] for n in nodes])
    betweenness = np.array([centrality[n]["betweenness_centrality"] for n in nodes])

    deg_hi = np.percentile(degrees, 75) if len(degrees) else 0
    bet_hi = np.percentile(betweenness, 75) if len(betweenness) else 0

    roles: Dict[str, str] = {}
    for n, deg, bet in zip(nodes, degrees, betweenness):
        if deg >= deg_hi and bet >= bet_hi:
            roles[n] = "Hub"
        elif bet >= bet_hi and deg < deg_hi:
            roles[n] = "Bottleneck"
        else:
            roles[n] = "Periphery"
    return roles


def compute_topology(G: nx.Graph, progress_cb: ProgressCB = None) -> Dict[str, Dict]:
    """Run all Layer 1 metrics and attach them as node attributes on G."""
    if G.number_of_nodes() == 0:
        return {}

    if progress_cb:
        progress_cb(5, "Computing degree centrality...")
    degree_centrality = nx.degree_centrality(G)

    if progress_cb:
        progress_cb(20, "Computing betweenness centrality...")
    weight_key = "weight" if nx.get_edge_attributes(G, "weight") else None
    betweenness = nx.betweenness_centrality(G, weight=weight_key, normalized=True)

    if progress_cb:
        progress_cb(45, "Computing PageRank...")
    try:
        pagerank = nx.pagerank(G, weight=weight_key)
    except nx.PowerIterationFailedConvergence:
        pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    if progress_cb:
        progress_cb(55, "Computing eigenvector centrality...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500, weight=weight_key)
    except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution):
        eigenvector = {n: 0.0 for n in G.nodes()}

    if progress_cb:
        progress_cb(65, "Computing global network vulnerability...")
    vulnerability = compute_vulnerability(
        G, progress_cb=lambda pct, msg: progress_cb(65 + int(pct * 0.3), msg) if progress_cb else None
    )

    centrality: Dict[str, Dict[str, float]] = {}
    for n in G.nodes():
        centrality[n] = {
            "degree_centrality": degree_centrality.get(n, 0.0),
            "betweenness_centrality": betweenness.get(n, 0.0),
            "pagerank": pagerank.get(n, 0.0),
            "eigenvector_centrality": eigenvector.get(n, 0.0),
            "vulnerability": vulnerability.get(n, 0.0),
        }

    roles = classify_roles(centrality)
    for n in G.nodes():
        centrality[n]["role"] = roles.get(n, "Periphery")
        G.nodes[n].update(centrality[n])

    if progress_cb:
        progress_cb(100, "Layer 1 topology analysis complete.")
    return centrality
