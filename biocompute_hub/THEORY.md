# BioCompute-Hub — Theory & Mathematical Reference

This document derives and explains every formula the two simulation
engines and the information-thermodynamics analyzer actually compute, with
pointers to the exact function that implements each. It is a reference for
*why* a number the app reports is what it is, not a general biophysics
primer.

---

## Microscopic Agent Engine

*Implemented in [`simulation/micro_agent_engine.py`](simulation/micro_agent_engine.py).*

### Overdamped Langevin dynamics

Each particle's position $x$ evolves under the overdamped Langevin
(Brownian dynamics) equation:

$$x_{t+dt} = x_t - \frac{1}{\gamma}\nabla U(x_t)\,dt + \sqrt{2D\,dt}\,\xi,
\qquad \xi \sim \mathcal{N}(0, I)$$

Discretized with **Euler-Maruyama** integration (`_step_positions`). The
friction coefficient $\gamma$ and diffusion coefficient $D$ are linked by
the **Einstein relation**, $\gamma = k_BT/D$ (`config.GAMMA_SCAFFOLD`,
`GAMMA_LIGAND`), so species with higher $D$ (Input ligands, `D_LIGAND=5.0`)
also move under proportionally lower effective friction than the slower
Scaffold/Substrate species (`D_SCAFFOLD=1.0`).

Since $-\nabla U = F$ (force is minus the potential gradient), the
implementation form is:

$$x_{t+dt} = x_t + \frac{F(x_t)}{\gamma}\,dt + \sqrt{2D\,dt}\,\xi$$

### Lennard-Jones-style soft-core attraction

The pairwise potential driving clustering (a discrete analogue of phase
separation) is the standard 12-6 Lennard-Jones form:

$$U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

$$F(r) = -\frac{dU}{dr} = \frac{24\varepsilon}{r}\left[2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

with $\varepsilon=$ `LJ_EPSILON`, $\sigma=$ `LJ_SIGMA`. $F(r)$ is repulsive at
very short range (the $r^{-12}$ term dominates) and attractive for
$r > 2^{1/6}\sigma$ out to a cutoff `LJ_CUTOFF` $= 3\sigma$, beyond which the
interaction is truncated to zero. `_lj_forces` computes this pairwise for
**all** particle pairs under periodic boundary conditions (minimum-image
convention), fully vectorized as an $(N,N,2)$ tensor rather than a
per-particle Python loop.

**Stability safeguards** (why the solver doesn't explode): the squared
distance $r^2$ is floored at $(\sigma/2)^2$ before computing $1/r^2$
(prevents a division blow-up if two particles nearly coincide), and the
resulting force magnitude is clipped to $\pm$`LJ_FORCE_CAP` before being
applied — a standard soft-core stabilization for explicit integrators,
since the true LJ force diverges as $r\to 0$ while `dt` is finite.

### Gillespie-style catalytic signaling event

At every step, for each Input particle, the minimum periodic distance to
any Scaffold particle is computed. If that distance is below
`CONVERSION_RADIUS` ($R_c$), the particle is "near a scaffold cluster," and
a Bernoulli trial with

$$p = k_{cat}\cdot dt$$

decides whether catalysis fires this step (`_catalysis_step`). On success,
the *nearest currently-Inactive* Substrate particle to that Input particle
is converted to Active. This is the discrete-time, single-step-probability
form of a Gillespie first-order reaction: over many steps the expected
number of conversions per near-scaffold Input particle per unit time
converges to the continuous rate $k_{cat}$.

**Readouts** (`run`): `Input_Signal(t)` = count of Input particles
currently within $R_c$ of some Scaffold particle; `Output_Signal(t)` =
cumulative (monotonically non-decreasing) count of Active substrate
particles. These are the two time series handed to the information
analyzer.

---

## Mesoscopic Phase-Field Engine

*Implemented in [`simulation/meso_field_engine.py`](simulation/meso_field_engine.py), via `FiPy`.*

### Cahn-Hilliard equation

The condensate order parameter $\phi \in [-1,1]$ ($\phi>0$ = condensate
phase, $\phi<0$ = dilute phase) obeys conserved (mass-preserving)
gradient-flow dynamics on the Ginzburg-Landau double-well free energy
$f(\phi)=\tfrac14(1-\phi^2)^2$:

$$\frac{\partial \phi}{\partial t} = \nabla\!\cdot\!\Big(M\,\nabla\mu\Big),
\qquad \mu = \frac{\delta F}{\delta\phi} = \phi^3-\phi-\kappa\nabla^2\phi$$

$M$ = `MOBILITY_M`, $\kappa$ = `KAPPA` (gradient energy coefficient — the
interfacial-width/surface-tension parameter). Starting from small random
noise (`PHI_NOISE_STD`) rather than a uniform field is what triggers
**spinodal decomposition**: the double well is linearly unstable around
$\phi=0$, so infinitesimal fluctuations grow, driving phase separation
into $\phi\to\pm1$ domains that subsequently coarsen.

**Numerical scheme.** The equation is 4th-order in space, which FiPy's
finite-volume method cannot discretize directly; it is split into the
standard coupled pair of 2nd-order equations, `phi` and the chemical
potential `psi`:

$$\partial_t\phi = \nabla\!\cdot\!(M\nabla\psi), \qquad \psi = \phi^3-\phi-\kappa\nabla^2\phi$$

The nonlinear bulk term $\phi^3-\phi$ is handled by a **semi-implicit
Taylor linearization** around the current iterate (the pattern used in
FiPy's own Cahn-Hilliard example): writing
$d^2f/d\phi^2 = 3\phi^2-1$,

$$\psi \approx (3\phi_{old}^2-1)(\phi_{new}-\phi_{old}) + (\phi_{old}^3-\phi_{old}) - \kappa\nabla^2\phi_{new}$$

which FiPy solves as a coupled linear system each **sweep**
(`eq.sweep(dt=dt)`), re-linearizing and repeating until the residual drops
below `SWEEP_RESIDUAL_TOL` or `MAX_SWEEPS` is reached — this is what keeps
the implicit solve both nonlinear-accurate and numerically stable at
`MESO_DT=0.01`, where a naive fully-explicit 4th-order scheme would need a
much smaller timestep to avoid diverging.

### Coupled reaction-diffusion for the signaling molecule

$$\frac{\partial C}{\partial t} = D_C\nabla^2 C - k(\phi)\,C + S(t)$$

with a **phase-dependent degradation rate** (`_step_reaction_diffusion`):

$$k(\phi) = \begin{cases} k_{cond} & \phi > 0 \ \text{(inside the condensate)} \\ k_{sol} & \phi \le 0 \end{cases}$$

($k_{sol}=$ `K_REACT_SOL`, $k_{cond}=$ `K_REACT_COND`, with
$k_{cond} \gg k_{sol}$ — condensates locally concentrate degradation
machinery/enzymes). The source term is a **localized, time-varying
stochastic drive**: a fixed Gaussian spatial profile centered on the grid
(width `SOURCE_RADIUS`) scaled by a positive stochastic amplitude drawn
fresh each step,

$$S(\mathbf{x}, t) = \max\!\big(0,\ A_0 + \mathcal{N}(0,\sigma_S)\big)\cdot
\exp\!\left(-\frac{|\mathbf{x}-\mathbf{x}_0|^2}{2\,R_{src}^2}\right)$$

$A_0=$ `SOURCE_AMPLITUDE`, $\sigma_S=$ `SOURCE_NOISE_STD`. This equation is
one-way coupled to the phase field ($\phi$ affects $C$'s degradation, $C$
does not feed back into $\phi$), so it is solved once per step after the
Cahn-Hilliard sweep, using the just-updated $\phi$ to set $k(\phi)$.

**Readouts**: `Input_Source_S(t)` = the scalar stochastic amplitude $A_0+\text{noise}$
at that step; `Output_Reacted_C(t)` = $\sum_{\text{cells}} C$, the total
signaling-molecule mass currently in the field.

**Divergence guard**: after every step, `run` checks `np.all(np.isfinite(...))`
on both fields and raises `FloatingPointError` rather than silently
returning a corrupted result if the solve ever produces NaN/Inf.

---

## Information Thermodynamics Analyzer

*Implemented in [`analysis/info_thermodynamics.py`](analysis/info_thermodynamics.py).*

### KSG mutual information estimator

For continuous time series $X,Y$ (no binning/discretization), the
**Kraskov-Stögbauer-Grassberger (KSG-1)** estimator gives, in nats:

$$\hat I(X;Y) = \psi(k) - \big\langle \psi(n_x+1) + \psi(n_y+1)\big\rangle + \psi(N)$$

where $\psi$ is the digamma function, $k$ = `KSG_K_NEIGHBORS`, and for each
sample point $i$: $\varepsilon_i$ is the Chebyshev (max-norm) distance to
its $k$-th nearest neighbor in the **joint** $(x,y)$ space; $n_x(i)$
($n_y(i)$) is the number of other points whose $x$ ($y$) coordinate alone
lies within $\varepsilon_i$. Implemented with `scipy.spatial.cKDTree` for
both the joint-space $k$-NN query and the marginal-space radius counts
(`mutual_information_ksg`). The app reports the result in **bits**
(divided by $\ln 2$).

**Dequantization.** Simulated time series are frequently quantized (the
micro engine's `Input_Signal` is a small integer count; `Output_Signal` is
a plateauing cumulative integer). KSG is undefined when the $k$-th nearest
neighbor distance is exactly zero (a tie), which happens constantly with
raw integer data. `_dequantize` adds i.i.d. Gaussian jitter with a standard
deviation of $10^{-4}\times$ the series' own spread before estimation —
small enough not to distort the estimate, large enough to resolve ties.

**Fallback.** If KSG is infeasible (fewer than $k+2$ samples) or returns a
non-finite value, `mutual_information` falls back to a binned estimator via
`sklearn.metrics.mutual_info_score` on digitized samples
(`mutual_information_histogram`).

### Transfer entropy (conditional mutual information)

$$T_{X\to Y} = I(Y_t\,;\,X_{t-\ell} \mid Y_{t-\ell})$$

$\ell=$ `TE_LAG` — the information the source's *past* provides about the
target's *present*, beyond what the target's own past already predicts.
This is a **conditional** mutual information, estimated with the
Frenzel-Pompe generalization of KSG (`_conditional_mutual_information_ksg`):
for the triple $(A,B,C)=(Y_t, X_{t-\ell}, Y_{t-\ell})$, $\varepsilon_i$ is now
the $k$-th nearest-neighbor distance in the **3D joint** $(A,B,C)$ space,
and

$$\hat I(A;B\mid C) = \psi(k) - \big\langle \psi(n_{AC}+1) + \psi(n_{BC}+1) - \psi(n_C+1)\big\rangle$$

with $n_{AC}, n_{BC}$ counted in the 2D $(A,C)$ and $(B,C)$ marginal
subspaces and $n_C$ in the 1D $C$ marginal. `transfer_entropy` builds the
lagged embedding ($Y_t = y[\ell:]$, $X_{t-\ell}=x[:-\ell]$,
$Y_{t-\ell}=y[:-\ell]$) and calls this estimator; it returns 0.0 (rather
than raising) if the series is too short or the estimate is non-finite.

*Validated directionality*: on synthetic data where $y_t = 0.8\,x_{t-1} +
\text{noise}$, this implementation correctly recovers a large
$T_{X\to Y}$ and a near-zero $T_{Y\to X}$ — confirming the estimator
detects the true causal direction rather than merely correlation.

### Noise ratio

$$\text{noise\_ratio} = 1 - \rho_{XY}^2, \qquad \rho_{XY} = \text{corr}(X,Y)$$

The fraction of $Y$'s variance **not** linearly explained by $X$ — a
scale-free, complementary readout alongside the (nonlinear-capable) KSG
mutual information. 0 = perfectly linearly predictable; 1 = no linear
relationship (a constant series returns 1.0 by convention, since a
constant carries no information regardless of correlation being
undefined).
