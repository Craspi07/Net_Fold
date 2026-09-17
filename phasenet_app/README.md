# PhaseNet-Sim

A modular PyQt6 desktop application acting as an exploratory "what-if"
sandbox, integrating graph network theory, biomolecular phase separation
(LLPS) prediction, spatial/kinetic modeling, and information-theoretic
analysis across a 5-layer computational pipeline. Three pre-curated
pathways (NF-κB/NEMO, Ras/MAPK, Stress Granules) give immediate,
pre-parameterized benchmarks without building an arbitrary network first.

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
2. **Context-Aware LLPS Propensity Profiling** (`core/layer2_llps.py`) —
   sequence disorder estimation, charge segregation (`κ`), sticker/spacer
   valency, and a composite `S_LLPS ∈ [0, 1]` score; flags Phase-Separation
   Critical Hubs (high betweenness AND high `S_LLPS`). A user-adjustable
   PTM weight and multivalency weight dynamically modulate `S_LLPS` on top
   of the sequence-intrinsic base score (`S_LLPS_base`), acknowledging that
   LLPS propensity is context-dependent rather than fixed by sequence alone.
3. **Spatial & Condensate-Modulated Kinetics** (`core/layer3_kinetics.py`) —
   ODE (`scipy.integrate.solve_ivp`) signal propagation with a Condensate
   State Engine (`K_part = exp(-ΔG / kT)`, adjustable amplification/
   sequestration factor `γ`); a Gillespie SSA alternative is also provided.
   The hot ODE-coupling and Gillespie loops are `numba`-jitted, and
   `simulate_task()` is a plain, picklable entry point so a single
   simulation is the unit of work dispatched across processes.
4. **Information Theory & Signal Processing** (`core/layer4_infotheory.py`) —
   mutual information `I(X;Y)`, Fano factor / SNR, temporal delay and gain,
   comparing LLPS-enabled vs. LLPS-disabled kinetics. The comparison run,
   the MI-vs-noise sweep, and multi-realization Monte Carlo pooling each
   fan their independent simulations out across a `ProcessPoolExecutor`.
5. **In-Silico Mutagenesis & Sensitivity** (`core/layer5_mutagenesis.py`) —
   systematic node knockouts, valency-loss mutagenesis (tests whether
   functional loss is spatial/LLPS-mediated vs. enzymatic, since `k_cat`
   is left untouched), and a Latin Hypercube sensitivity sweep over
   `(K_part threshold, γ)`. Every knockout/mutant/sample is an independent
   process-pool task.

## Pre-curated pathways (`core/pathways.py`)

- **NF-κB / NEMO Signalosome** — TNF-receptor-proximal IKK complex assembly.
- **Ras/MAPK Clustering** — KRAS nanoclustering nucleating RAF/MEK/ERK signaling.
- **Stress Granules** — G3BP1/2-driven, RNA-dependent cytoplasmic condensates.

Each template ships seed identifiers plus pre-tuned discovery/simulation
parameters (score threshold, γ, `K_part` threshold, suggested input/output
nodes, and PTM/multivalency weights) for immediate benchmarking.

## Parallelism

Layers 3-5 involve many independent, CPU-bound kinetic simulations (noise
realizations, per-node knockouts, LHS samples) that would block the GIL
under `QThread` alone. `utils/parallel_worker.py` keeps a single persistent
`ProcessPoolExecutor`; core-layer orchestrator functions fan their task
batches out across it via `run_parallel()`, while a `FunctionWorker` QThread
supervises the blocking wait so the Qt event loop is never touched. The
per-simulation hot loops are additionally accelerated with `numba.njit`
(gracefully falling back to pure Python if `numba` isn't installed).

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
│   ├── pathways.py
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
    └── parallel_worker.py
```
