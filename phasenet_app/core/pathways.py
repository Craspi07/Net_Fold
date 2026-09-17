"""Pre-curated pathway templates for immediate benchmarking without
building an arbitrary network from scratch.

Each entry gives seed identifiers plus a parameter set (discovery and
simulation defaults) tuned for that pathway's known biology, so the
sandbox can be exercised as a "what-if" tool against a validated context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PathwayTemplate:
    name: str
    description: str
    seed_ids: List[str]
    organism: int = 9606
    min_score: float = 0.4
    expand_depth: int = 1
    gamma: float = 2.0
    kpart_threshold: float = 0.5
    suggested_input: str = ""
    suggested_output: str = ""
    ptm_weight: float = 0.0
    multivalency_weight: float = 0.0
    notes: str = ""


PATHWAYS: Dict[str, PathwayTemplate] = {
    "NF-κB / NEMO Signalosome": PathwayTemplate(
        name="NF-κB / NEMO Signalosome",
        description=(
            "TNF-receptor-proximal IKK complex assembly. NEMO (IKBKG) "
            "polyubiquitin-chain-driven clustering is a well-studied "
            "phase-separation-adjacent signalosome."
        ),
        seed_ids=["IKBKG", "CHUK", "IKBKB", "TRAF6", "TAB1", "TAB2", "MAP3K7", "RELA", "NFKB1", "TNFAIP3"],
        min_score=0.5,
        expand_depth=1,
        gamma=3.0,
        kpart_threshold=0.45,
        suggested_input="TRAF6",
        suggested_output="RELA",
        ptm_weight=0.4,
        multivalency_weight=0.3,
        notes="NEMO oligomerization via K63-linked ubiquitin chains is treated as a PTM-driven multivalency effect.",
    ),
    "Ras/MAPK Clustering": PathwayTemplate(
        name="Ras/MAPK Clustering",
        description=(
            "KRAS nanoclustering at the plasma membrane nucleates the "
            "RAF/MEK/ERK cascade; membrane-proximal clustering behaves "
            "like a 2D condensate that gates signal amplitude and duration."
        ),
        seed_ids=["KRAS", "HRAS", "NRAS", "SOS1", "GRB2", "SHC1", "RAF1", "BRAF", "MAP2K1", "MAPK1", "MAPK3"],
        min_score=0.4,
        expand_depth=1,
        gamma=2.5,
        kpart_threshold=0.5,
        suggested_input="SOS1",
        suggested_output="MAPK1",
        ptm_weight=0.5,
        multivalency_weight=0.2,
        notes="GRB2/SOS1 SH3-domain multivalency and RAS phosphorylation are the dominant context modifiers here.",
    ),
    "Stress Granules": PathwayTemplate(
        name="Stress Granules",
        description=(
            "Canonical cytoplasmic RNA-protein condensates assembled "
            "under stress via G3BP1/2-driven, RNA-dependent LLPS."
        ),
        seed_ids=["G3BP1", "G3BP2", "CAPRIN1", "USP10", "TIA1", "TIAL1", "ATXN2", "EIF4G1", "PABPC1", "NUFIP2"],
        min_score=0.4,
        expand_depth=2,
        gamma=4.0,
        kpart_threshold=0.35,
        suggested_input="G3BP1",
        suggested_output="EIF4G1",
        ptm_weight=0.2,
        multivalency_weight=0.6,
        notes="G3BP1 dimerization plus multivalent RNA binding is the primary driver; PTM contribution is secondary.",
    ),
}


def get_pathway_names() -> List[str]:
    return list(PATHWAYS.keys())


def get_pathway(name: str) -> PathwayTemplate:
    if name not in PATHWAYS:
        raise KeyError(f"Unknown pathway template: {name}")
    return PATHWAYS[name]
