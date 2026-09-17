# PhaseNet-Sim

A modular PyQt6 desktop application integrating graph network theory,
biomolecular phase separation (LLPS) prediction, spatial/kinetic modeling,
and information-theoretic analysis across a 5-layer computational pipeline.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Pipeline

1. **Network Topology & Node Criticality** (`core/layer1_topology.py`) —
   betweenness/degree/PageRank/eigenvector centrality and global network
   vulnerability (`ΔE_global`) upon node removal; classifies nodes as
   Hub / Bottleneck / Periphery.
2. **LLPS Propensity Profiling** (`core/layer2_llps.py`) — sequence
   disorder estimation, charge segregation (`κ`), sticker/spacer valency,
   and a composite `S_LLPS ∈ [0, 1]` score; flags Phase-Separation
   Critical Hubs (high betweenness AND high `S_LLPS`).
3. **Spatial & Condensate-Modulated Kinetics** (`core/layer3_kinetics.py`) —
   ODE (`scipy.integrate.solve_ivp`) signal propagation with a Condensate
   State Engine (`K_part = exp(-ΔG / kT)`, adjustable amplification/
   sequestration factor `γ`); a Gillespie SSA alternative is also provided.
4. **Information Theory & Signal Processing** (`core/layer4_infotheory.py`) —
   mutual information `I(X;Y)`, Fano factor / SNR, temporal delay and gain,
   comparing LLPS-enabled vs. LLPS-disabled kinetics.
5. **In-Silico Mutagenesis & Sensitivity** (`core/layer5_mutagenesis.py`) —
   systematic node knockouts, valency-loss mutagenesis, and a Latin
   Hypercube sensitivity sweep over `(K_part threshold, γ)`.

## Data sources

- UniProt REST API (`https://rest.uniprot.org`) for sequences, gene names,
  and disordered-region annotations.
- STRING-DB API (`https://string-db.org/api/`) for confidence-weighted
  functional/physical interaction edges and network expansion.

Both are fetched asynchronously (`aiohttp`) off the GUI thread and cached
locally in SQLite (`./cache/phasenet_cache.db`) to avoid redundant calls.

## Project layout

```text
phasenet_app/
├── main.py
├── core/
│   ├── network_builder.py
│   ├── layer1_topology.py
│   ├── layer2_llps.py
│   ├── layer3_kinetics.py
│   ├── layer4_infotheory.py
│   └── layer5_mutagenesis.py
├── ui/
│   ├── main_window.py
│   ├── graph_view.py
│   └── plot_widgets.py
└── utils/
    ├── cache.py
    └── worker.py
```
