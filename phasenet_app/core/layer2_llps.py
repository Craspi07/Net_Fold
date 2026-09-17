"""Layer 2: Phase Separation (LLPS) Propensity Profiling.

Estimates disorder, charge segregation (kappa), sticker/spacer valency,
and a composite LLPS propensity score S_LLPS in [0, 1] per protein node,
using only sequence composition (no external disorder API dependency
required, though UniProt-reported disordered regions are used when present).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np

log = logging.getLogger("phasenet.layer2")
ProgressCB = Optional[Callable[[int, str], None]]

# Simplified Uversky/Dunker-style per-residue disorder propensity scale
# (higher = more disorder-promoting). Values are illustrative approximations
# of published disorder propensity rankings, used as a fast IUPred proxy.
DISORDER_PROPENSITY = {
    "A": 0.45, "R": 0.65, "N": 0.55, "D": 0.65, "C": 0.25,
    "Q": 0.60, "E": 0.70, "G": 0.55, "H": 0.45, "I": 0.15,
    "L": 0.20, "K": 0.65, "M": 0.30, "F": 0.15, "P": 0.75,
    "S": 0.55, "T": 0.45, "W": 0.20, "Y": 0.30, "V": 0.15,
}

POSITIVE = {"R", "K"}
NEGATIVE = {"D", "E"}
AROMATIC = {"F", "Y", "W"}
STICKER_RESIDUES = {"R", "Y", "F", "G", "D", "E"}

WINDOW = 15


def sequence_disorder_score(seq: str) -> float:
    """Mean sliding-window disorder propensity, a fast IUPred-like proxy."""
    if not seq:
        return 0.0
    scores = np.array([DISORDER_PROPENSITY.get(res, 0.4) for res in seq])
    if len(scores) < WINDOW:
        return float(np.mean(scores))
    kernel = np.ones(WINDOW) / WINDOW
    smoothed = np.convolve(scores, kernel, mode="valid")
    return float(np.mean(smoothed))


def charge_segregation_kappa(seq: str) -> float:
    """Approximate Das-Pappu kappa: charge patterning along the sequence.

    kappa in [0, 1]; higher = more blocky segregation of + and - charges,
    which favours LLPS via electrostatic complex coacervation.
    """
    if len(seq) < WINDOW:
        return 0.0
    charges = np.array([1.0 if r in POSITIVE else (-1.0 if r in NEGATIVE else 0.0) for r in seq])
    n_pos = np.sum(charges > 0)
    n_neg = np.sum(charges < 0)
    total_charged = n_pos + n_neg
    if total_charged < 2:
        return 0.0

    windows = []
    for i in range(0, len(charges) - WINDOW + 1, max(1, WINDOW // 2)):
        w = charges[i:i + WINDOW]
        local_asymmetry = abs(np.sum(w > 0) - np.sum(w < 0)) / WINDOW
        windows.append(local_asymmetry)
    if not windows:
        return 0.0
    global_fcr = total_charged / len(seq)
    kappa = float(np.mean(windows)) * min(1.0, global_fcr * 3)
    return float(np.clip(kappa, 0.0, 1.0))


def sticker_valency(seq: str) -> Dict[str, float]:
    """Count aromatic/charged sticker residues that drive multivalent contacts."""
    if not seq:
        return {"valency": 0.0, "aromatic_frac": 0.0, "charged_frac": 0.0}
    length = len(seq)
    sticker_count = sum(seq.count(r) for r in STICKER_RESIDUES)
    aromatic_count = sum(seq.count(r) for r in AROMATIC)
    charged_count = sum(seq.count(r) for r in POSITIVE | NEGATIVE)
    return {
        "valency": sticker_count / length,
        "aromatic_frac": aromatic_count / length,
        "charged_frac": charged_count / length,
    }


def composite_llps_score(disorder: float, kappa: float, valency_info: Dict[str, float]) -> float:
    """Weighted composite S_LLPS in [0, 1].

    Weighting follows the general consensus that disorder and multivalent
    sticker content dominate, with charge patterning as a secondary driver.
    """
    valency_term = min(1.0, valency_info["valency"] * 4.0)
    aromatic_term = min(1.0, valency_info["aromatic_frac"] * 8.0)
    score = (
        0.40 * disorder
        + 0.20 * kappa
        + 0.25 * valency_term
        + 0.15 * aromatic_term
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_llps(G: nx.Graph, progress_cb: ProgressCB = None) -> Dict[str, Dict]:
    """Annotate every node with LLPS feature vector + S_LLPS, in-place on G."""
    results: Dict[str, Dict] = {}
    nodes = list(G.nodes())
    total = max(1, len(nodes))

    for i, n in enumerate(nodes):
        seq = G.nodes[n].get("sequence", "") or ""
        disorder_regions = G.nodes[n].get("disorder_regions", [])

        disorder = sequence_disorder_score(seq)
        if disorder_regions and seq:
            annotated_frac = min(
                1.0,
                sum(max(0, r["end"] - r["start"]) for r in disorder_regions) / max(1, len(seq)),
            )
            disorder = float(np.clip(0.5 * disorder + 0.5 * annotated_frac, 0.0, 1.0))

        kappa = charge_segregation_kappa(seq)
        valency_info = sticker_valency(seq)
        s_llps = composite_llps_score(disorder, kappa, valency_info)

        feature = {
            "disorder_score": disorder,
            "kappa": kappa,
            "valency": valency_info["valency"],
            "aromatic_frac": valency_info["aromatic_frac"],
            "charged_frac": valency_info["charged_frac"],
            "s_llps": s_llps,
        }
        results[n] = feature
        G.nodes[n].update(feature)

        if progress_cb:
            progress_cb(int(100 * (i + 1) / total), f"LLPS profiling: {n}")

    return results


def identify_critical_hubs(G: nx.Graph, betweenness_pctl: float = 75, llps_pctl: float = 75) -> List[str]:
    """Phase Separation Critical Hubs: high C_B AND high S_LLPS."""
    nodes = list(G.nodes())
    if not nodes:
        return []
    betweenness = np.array([G.nodes[n].get("betweenness_centrality", 0.0) for n in nodes])
    s_llps = np.array([G.nodes[n].get("s_llps", 0.0) for n in nodes])
    if not np.any(betweenness) or not np.any(s_llps):
        return []
    bet_thresh = np.percentile(betweenness, betweenness_pctl)
    llps_thresh = np.percentile(s_llps, llps_pctl)
    critical = [n for n, b, s in zip(nodes, betweenness, s_llps) if b >= bet_thresh and s >= llps_thresh]
    for n in nodes:
        G.nodes[n]["ps_critical_hub"] = n in critical
    return critical
