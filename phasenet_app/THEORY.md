# PhaseNet-Sim — Theory & Mathematical Reference

This document derives and explains every formula the pipeline actually
computes, layer by layer, with pointers to the exact function that
implements it. It is a reference for *why* a number in the app is what it
is, not a tutorial on graph theory or LLPS biology in general.

---

## Layer 1 — Network Topology & Node Criticality

*Implemented in [`core/layer1_topology.py`](core/layer1_topology.py).*

### Degree centrality

$$C_D(i) = \frac{\deg(i)}{N - 1}$$

Fraction of all other nodes that `i` is directly connected to.
`networkx.degree_centrality` (`compute_topology`).

### Betweenness centrality

$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and
$\sigma_{st}(i)$ the number of those passing through $i$. Weighted by STRING
edge confidence when present (`networkx.betweenness_centrality`, normalized).

### PageRank

$$PR(i) = \frac{1-d}{N} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)}$$

with damping factor $d$ (NetworkX default 0.85), $M(i)$ the in-neighbors of
$i$, $L(j)$ the out-degree of $j$. Falls back to a uniform distribution if
power iteration fails to converge.

### Eigenvector centrality

$$\mathbf{x} = \frac{1}{\lambda} A \mathbf{x}$$

$\mathbf{x}$ is the eigenvector associated with the largest eigenvalue
$\lambda$ of the adjacency matrix $A$ — a node scores highly if it is
connected to other high-scoring nodes.

### Global network efficiency

$$E_{global}(G) = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$$

$d_{ij}$ = shortest-path distance between $i$ and $j$ (`global_efficiency`).
Disconnected pairs ($d_{ij} = \infty$) contribute 0.

### Vulnerability (single-node knockout)

$$\Delta E_{global}(i) = E_{global}(G) - E_{global}(G \setminus \{i\})$$

`compute_vulnerability` removes each node in turn, recomputes $E_{global}$
on the reduced graph, and reports the drop. For graphs above `max_nodes`
(default 200) this is restricted to the highest-degree nodes, since it is
$O(N)$ recomputations each costing $O(N^2)$.

### Topological role classification

Let $\hat{d}_i, \hat{b}_i$ be node $i$'s degree/betweenness centrality, and
$d_{75}, b_{75}$ the 75th percentile of each across the graph
(`classify_roles`):

$$
\text{role}(i) = \begin{cases}
\text{Hub} & \hat{d}_i \ge d_{75} \ \text{and}\ \hat{b}_i \ge b_{75} \\
\text{Bottleneck} & \hat{b}_i \ge b_{75} \ \text{and}\ \hat{d}_i < d_{75} \\
\text{Periphery} & \text{otherwise}
\end{cases}
$$

---

## Layer 2 — Context-Aware LLPS Propensity Profiling

*Implemented in [`core/layer2_llps.py`](core/layer2_llps.py).*

### Disorder score (IUPred proxy)

Each residue $r$ is assigned a fixed disorder-propensity weight
$w(r) \in [0,1]$ (`DISORDER_PROPENSITY`, a coarse approximation of the
Uversky/Dunker disorder scale). The per-sequence score is a sliding-window
average (`sequence_disorder_score`), window size $W = 15$:

$$D(\text{seq}) = \frac{1}{L-W+1}\sum_{k=0}^{L-W} \left(\frac{1}{W}\sum_{i=k}^{k+W-1} w(r_i)\right)$$

If UniProt-reported disordered regions exist, the score is blended 50/50
with the fraction of the sequence they annotate as disordered:

$$D' = \text{clip}\big(0.5\,D + 0.5\,f_{\text{annotated}},\ 0,\ 1\big)$$

### Charge segregation ($\kappa$, Das–Pappu-style)

`charge_segregation_kappa` assigns $+1$ to Arg/Lys, $-1$ to Asp/Glu, $0$
otherwise, then slides a window of size $W$ across the sequence (stride
$W/2$) and measures local charge imbalance:

$$\text{asym}_k = \frac{|n^+_k - n^-_k|}{W}, \qquad
\kappa = \text{clip}\!\left(\overline{\text{asym}} \cdot \min(1,\ 3\,\text{FCR}),\ 0,\ 1\right)$$

where $\text{FCR}$ (fraction of charged residues) is
$(n^+_{\text{total}} + n^-_{\text{total}})/L$. Higher $\kappa$ means charges
are blockier (segregated) rather than evenly mixed, favoring electrostatic
complex coacervation.

### Sticker/spacer valency

`sticker_valency` counts aromatic (F, Y, W) and charged (R, K, D, E)
sticker residues as a fraction of sequence length:

$$v_{\text{sticker}} = \frac{\#\{r \in \{R,Y,F,G,D,E\}\}}{L}, \qquad
f_{\text{aromatic}} = \frac{\#\{r \in \{F,Y,W\}\}}{L}$$

### Composite LLPS score

$$S_{LLPS} = \text{clip}\Big(0.40\,D + 0.20\,\kappa
+ 0.25\,\min(1,\ 4\,v_{\text{sticker}}) + 0.15\,\min(1,\ 8\,f_{\text{aromatic}}),\ 0,\ 1\Big)$$

(`composite_llps_score`) — disorder and multivalent sticker content
dominate, consistent with the general LLPS literature consensus; charge
patterning is a secondary term.

### PTM / multivalency context modulation

LLPS propensity is not fixed by sequence alone. `apply_context_modifiers`
perturbs $\kappa$ and $v_{\text{sticker}}$ before recomputing the composite
score, given user weights $w_{PTM}, w_{MV} \in [0,1]$:

$$\kappa' = \text{clip}\big(\kappa + w_{PTM}\cdot 0.4\,(1-\kappa),\ 0,\ 1\big)$$

$$v'_{\text{sticker}} = \text{clip}(v_{\text{sticker}} + w_{MV}\cdot 0.06,\ 0,\ 1)$$

$\kappa'$ approaches an asymptote as $w_{PTM}\to 1$ (phosphorylation
sharpens charge blockiness but cannot exceed maximal segregation);
$v'_{\text{sticker}}$ is a direct additive valency boost (extra binding
partners). The app reports both `s_llps_base` (unmodified) and `s_llps`
(context-modulated) so the effect of the sliders is always visible.

### Phase-Separation Critical Hubs

`identify_critical_hubs`: nodes at or above the 75th percentile of *both*
$C_B$ and $S_{LLPS}$ simultaneously — the topological positions where
condensate formation would do the most damage to information flow if
disrupted.

---

## Layer 3 — Spatial & Condensate-Modulated Kinetic Simulator

*Implemented in [`core/layer3_kinetics.py`](core/layer3_kinetics.py).*

### Signal propagation ODE

For every node $i$, concentration $C_i(t)$ evolves as

$$\frac{dC_i}{dt} = -\delta\, C_i \;+\; k_{sol}\, r_i \sum_{j} A_{ij}\,(C_j - C_i) \;+\; \text{Input}_i(t)$$

- $\delta$ = `degradation` (default 0.05)
- $A_{ij}$ = STRING confidence-weighted adjacency
- $k_{sol}$ = base soluble-phase rate constant (default 0.5)
- $r_i$ = the condensate rate multiplier (below)
- $\text{Input}_i(t)$ = forcing term, nonzero only at designated input
  nodes, shape set by `signal_kind`:
  - **step**: $A$ for $t \ge 1$, else 0
  - **pulse**: $A$ for $1 \le t \le 3$, else 0
  - **gaussian**: step $+\ \mathcal{N}(0,\ 0.15A)$ noise

Integrated with `scipy.integrate.solve_ivp` (RK45). The diffusive coupling
term is numba-`njit`-compiled (`_ode_coupling`) since it is the $O(N^2)$
inner loop evaluated at every solver substep.

### Condensate State Engine

Free energy of partitioning into the condensate, relative to a user-set
$S_{LLPS}$ threshold $\theta$ (`kpart_threshold`):

$$\Delta G(S_{LLPS}) = -(S_{LLPS} - \theta)\cdot\lambda, \qquad \lambda = 6\ \text{(DELTA\_G\_SCALE)}$$

Partition coefficient (Boltzmann-form, $k_BT=1$):

$$K_{part} = e^{-\Delta G / k_BT} = e^{(S_{LLPS}-\theta)\lambda}$$

Condensate-phase fraction for node $i$ (0 if $S_{LLPS} \le \theta$):

$$f_{cond}(i) = \frac{K_{part}}{1+K_{part}}$$

This is a logistic switch in $S_{LLPS}$ centered at $\theta$: nodes well
below threshold are essentially fully soluble ($f_{cond}\to 0$), nodes well
above are essentially fully condensed ($f_{cond}\to 1$), and $\lambda$
controls how sharp that transition is.

### Kinetic rate modulation

$$r_i = 1 + (\gamma - 1)\,f_{cond}(i)$$

so $r_i \to 1$ (unmodified) when $f_{cond}=0$, and $r_i \to \gamma$ when
$f_{cond}=1$ — $\gamma>1$ models condensate-driven amplification
(concentration/proximity effects), $\gamma<1$ models sequestration
(kinetic trapping / reduced accessibility). When "LLPS disabled" is
selected for comparison, $r_i \equiv 1$ for every node.

### Gillespie SSA (discrete stochastic alternative)

`gillespie_ssa` / `_gillespie_core` implement a standard exact
stochastic-simulation algorithm over three event classes per node $i$:
production (rate 1 at input nodes), decay (rate $\delta\,n_i$), and
graph-coupled transfer (rate $k_{sol}\,r_i\sum_j A_{ij}n_j$). At each step,
total rate $R = \sum_i(\text{production}_i+\text{decay}_i+\text{transfer}_i)$
determines the waiting time $\tau \sim \text{Exp}(R)$ and the next event is
chosen by roulette-wheel sampling proportional to each rate. Fully
numba-jitted for speed.

---

## Layer 4 — Information Theory & Signal Processing Engine

*Implemented in [`core/layer4_infotheory.py`](core/layer4_infotheory.py).*

### Mutual information (binned/histogram estimator)

$$I(X;Y) = \sum_{i,j} p(x_i,y_j)\, \log_2\!\frac{p(x_i,y_j)}{p(x_i)\,p(y_j)}$$

computed via `np.histogram2d` (default 16 bins per axis) on the input/output
trajectories, in bits.

### Fano factor

$$F = \frac{\mathrm{Var}(Y)}{\langle Y \rangle}$$

A measure of noise relative to signal level; $F=1$ is Poisson-like,
$F<1$ indicates sub-Poissonian (buffered) noise.

### Signal-to-noise ratio

$$\mathrm{SNR} = \frac{\langle Y \rangle^2}{\mathrm{Var}(Y)}$$

### Temporal delay

Lag at the peak of the cross-correlation between input and output:

$$\tau^{\star} = \arg\max_{\tau}\ \sum_t (y_t - \bar y)(x_{t-\tau} - \bar x)$$

(`np.correlate`, `mode="full"`), converted from sample lag to simulation
time using the trajectory's time step.

### Gain

$$G = \frac{\max(Y)-\min(Y)}{\max(X)-\min(X)}$$

### LLPS-on/off comparison, noise sweep, multi-realization pooling

`compare_llps_effect` runs the *same* simulation twice (γ active vs.
$\gamma\equiv1$) and reports all five metrics above for each, so a single
number ($\Delta I$, $\Delta F$, ...) quantifies what the condensate state
engine is actually buying (or costing) the network. `mi_vs_noise_curve`
repeats this across a sweep of `noise_amplitude` values. `parallel_mi_realizations`
runs $R$ independent noisy trials with different RNG seeds and *pools*
their trajectories before estimating $I(X;Y)$ — a larger effective sample
reduces the single-run estimator's variance. All three fan their
independent `simulate_task` calls out across a `ProcessPoolExecutor`
(see `utils/parallel_worker.py`).

---

## Layer 5 — In-Silico Mutagenesis & Sensitivity Dashboard

*Implemented in [`core/layer5_mutagenesis.py`](core/layer5_mutagenesis.py).*

### Deletion (knockout) scan

For each candidate node $v$ (excluding the chosen input/output nodes):

$$\Delta I(v) = I_{\text{baseline}}(X;Y) - I_{G\setminus\{v\}}(X;Y)$$

If removing $v$ disconnects the input from the output entirely, $\Delta I(v)$
is taken to equal $I_{\text{baseline}}$ directly (full information decay)
without needing to simulate — `nx.has_path` is checked first.

### Valency (loss-of-disorder) mutagenesis

For each node with $S_{LLPS} > \theta$, construct a mutant graph with

$$S_{LLPS}'(v) = \max(0,\ S_{LLPS}(v) - \Delta v)$$

and measure the same $\Delta I(v)$. Because $k_{sol}$ (the catalytic-analog
rate) is left completely untouched, any information drop here is
attributable to loss of condensate-mediated spatial organization rather
than to a change in intrinsic reaction rate — this isolates the "is the
phenotype spatial or enzymatic?" question.

### Sensitivity matrix (Latin Hypercube sampling)

`scipy.stats.qmc.LatinHypercube` draws $n$ well-stratified samples over the
2D parameter box $(\theta, \gamma) \in [\theta_{min},\theta_{max}] \times
[\gamma_{min},\gamma_{max}]$ (default $[0.1,0.9]\times[0.5,5.0]$). LHS is
used instead of a uniform grid or pure-random sampling because it
guarantees each parameter's marginal is evenly covered even with a small
sample budget — important since each sample is a full kinetic simulation.
$I(X;Y)$ is computed for every sample point, giving a scattered estimate of
the $I(X;Y)$ response surface over $(\theta,\gamma)$-space (visualized as
the Tab 4 heatmap). Every knockout / mutant / LHS sample is dispatched as
an independent task to the process pool.
