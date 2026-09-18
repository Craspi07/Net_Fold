"""Biophysical constants and hyperparameters for BioCompute-Hub.

All values are dimensionless / coarse-grained rescalings chosen for
numerical stability under explicit (Euler-Maruyama / finite-volume) time
stepping, not literal physical units - standard practice for this class
of agent-based and phase-field biophysical simulation.
"""
from __future__ import annotations

# --------------------------------------------------------------- Global
BOX_SIZE = 100.0          # L: simulation box side length (square, periodic)
DT = 0.01                 # base integration timestep
N_STEPS_DEFAULT = 2000    # default number of integration steps for a run
RANDOM_SEED = 42          # default RNG seed for reproducibility

# --------------------------------------------------------- Thermal / diffusive
KB_T = 1.0                                  # thermal energy scale
D_SCAFFOLD = 1.0                            # scaffold / substrate diffusion coefficient
D_LIGAND = 5.0                              # input ligand diffusion coefficient (faster, smaller)
GAMMA_SCAFFOLD = KB_T / D_SCAFFOLD          # Langevin friction coefficient, scaffold-scale species
GAMMA_LIGAND = KB_T / D_LIGAND              # Langevin friction coefficient, ligand-scale species

# --------------------------------------------------------------- Reaction kinetics
K_ON = 0.5                 # binding/association rate (informs the catalysis-trigger encounter radius)
K_CAT = 0.1                # catalytic conversion rate, Inactive -> Active

# ============================================================ Micro agent engine
N_PARTICLES = 200          # total particle count
N_SCAFFOLD = 60            # scaffold (phase-separating, LJ-attractive) particles
N_INPUT = 40                # input/ligand (signal-carrying) particles
# remainder (N_PARTICLES - N_SCAFFOLD - N_INPUT) are substrate particles,
# initialized Inactive and convertible to Active by nearby catalysis.

LJ_EPSILON = 1.0           # Lennard-Jones potential well depth
LJ_SIGMA = 2.0              # Lennard-Jones particle interaction length scale
LJ_CUTOFF = 3.0 * LJ_SIGMA  # interaction cutoff radius
LJ_FORCE_CAP = 75.0         # max force magnitude - prevents Euler-Maruyama blow-up at short range

CONVERSION_RADIUS = 3.0     # R_c: Input-to-Scaffold encounter radius that gates catalysis

# ============================================================ Meso phase-field engine
GRID_NX = 48                # grid cells along x
GRID_NY = 48                # grid cells along y
GRID_DX = BOX_SIZE / GRID_NX  # cell width (square cells: dy = dx)
MESO_DT = 0.01               # phase-field integration timestep
MESO_N_STEPS_DEFAULT = 250   # default number of meso steps (PDE solves are costlier per-step
                              # than micro-engine steps; tuned to keep a full run in ~1 minute)

MOBILITY_M = 1.0             # Cahn-Hilliard mobility M
KAPPA = 2.0                  # gradient energy coefficient kappa
PHI_NOISE_STD = 0.3          # initial noise amplitude for phi (drives spinodal decomposition)
MAX_SWEEPS = 3               # max nonlinear sweeps per implicit CH step (semi-implicit
                              # linearization converges well within this at MESO_DT=0.01)
SWEEP_RESIDUAL_TOL = 1e-3    # residual tolerance to stop sweeping early

D_C = 5.0                    # signaling-molecule (C) diffusion coefficient
K_REACT_SOL = 0.05           # degradation rate of C outside the condensate (phi <= 0)
K_REACT_COND = 0.5           # degradation rate of C inside the condensate (phi > 0); must exceed K_REACT_SOL
SOURCE_AMPLITUDE = 1.0       # baseline stochastic source amplitude
SOURCE_NOISE_STD = 0.2       # stddev of the Gaussian perturbation to the source amplitude
SOURCE_RADIUS = 5.0          # spatial width (std) of the localized Gaussian source profile

# ============================================================ Info thermodynamics
KSG_K_NEIGHBORS = 4          # k (nearest-neighbor count) for the KSG estimator
HISTOGRAM_BINS = 16          # bin count for the histogram-based MI fallback
TE_LAG = 1                   # time lag (in samples) used for transfer entropy
