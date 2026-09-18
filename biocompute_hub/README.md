# BioCompute-Hub

A computational biophysics and information-thermodynamics simulation
suite bridging physical macromolecular phase separation (LLPS) and
biological signal processing, at two levels of description:

- **Microscopic** (`simulation/micro_agent_engine.py`) - discrete agent-based
  2D overdamped Langevin dynamics (Euler-Maruyama), with a Lennard-Jones-style
  attractive potential driving Scaffold-particle clustering, and a
  Gillespie-style catalytic event (Input-near-Scaffold triggers Inactive
  -> Active substrate conversion with probability `p = k_cat * dt`).
- **Mesoscopic** (`simulation/meso_field_engine.py`) - continuum Cahn-Hilliard
  phase-field dynamics (`FiPy`, periodic 2D grid) coupled to a
  reaction-diffusion equation for a signaling molecule whose degradation
  rate is elevated inside the condensate phase, driven by a localized,
  time-varying stochastic source.

Both engines' input/output time series are analyzed for information content
with `analysis/info_thermodynamics.py`: a Kraskov-Stogbauer-Grassberger
(KSG) k-nearest-neighbor estimator (`scipy.spatial.cKDTree`) for mutual
information and transfer entropy, with a histogram-based fallback
(`sklearn.metrics.mutual_info_score`) for small or degenerate samples.

## Install

A virtual environment is recommended: this project's dependencies (notably
`fipy`) are heavy and best kept isolated from your system Python and from
other projects (`phasenet_app` in this repo pulls in a different,
non-overlapping dependency set via PyQt6).

```bash
cd biocompute_hub

# 1. create the venv
python3 -m venv .venv

# 2. activate it
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (cmd)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# 3. install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python main.py --mode micro    # agent-based Langevin dynamics
python main.py --mode meso     # Cahn-Hilliard phase-field PDE
```

Deactivate the environment anytime with `deactivate`.

Both save a three-panel dashboard to `output.png` (override with
`--output`): final spatial state, input/output signal time series, and the
computed mutual information / transfer entropy / noise ratio.

Useful flags: `--n-steps N`, `--seed S`, `--ksg-k K`, `--te-lag L`.

## Numerical stability

All constants in `config.py` are pre-tuned for stability under explicit
time-stepping: Langevin forces are Lennard-Jones-capped (`LJ_FORCE_CAP`) to
prevent Euler-Maruyama blow-up at short range, and the Cahn-Hilliard solve
uses a semi-implicit nonlinear linearization swept to a residual tolerance
each step. The meso engine raises `FloatingPointError` (caught and reported
by `main.py`, exit code 1) rather than silently returning garbage if a run
ever produces non-finite field values.
