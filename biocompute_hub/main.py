"""BioCompute-Hub CLI orchestrator and Matplotlib dashboard.

Usage:
    python main.py --mode micro [--n-steps N] [--seed S] [--output PATH]
    python main.py --mode meso  [--n-steps N] [--seed S] [--output PATH]

Runs the selected simulation engine, routes its input/output time series
through the information-thermodynamics analyzer, and saves a three-panel
dashboard (final spatial state, input/output time series, information
metrics) to PNG.
"""
from __future__ import annotations

import argparse
import logging
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe: this tool only ever saves to a file
import matplotlib.pyplot as plt
import numpy as np

from analysis.info_thermodynamics import compute_information_metrics
from config import (
    KSG_K_NEIGHBORS, MESO_N_STEPS_DEFAULT, N_STEPS_DEFAULT, RANDOM_SEED, TE_LAG,
)
from simulation.micro_agent_engine import (
    INPUT, SCAFFOLD, MicroSimulationResult, run_micro_simulation,
)
from simulation.meso_field_engine import MesoSimulationResult, run_meso_simulation

log = logging.getLogger("biocompute.main")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BioCompute-Hub: microscopic/mesoscopic phase-separation "
                    "and information-thermodynamics simulation suite.",
    )
    parser.add_argument("--mode", choices=["micro", "meso"], required=True,
                         help="Simulation engine to run: 'micro' (Langevin agent-based) "
                              "or 'meso' (Cahn-Hilliard phase-field).")
    parser.add_argument("--n-steps", type=int, default=None,
                         help="Number of integration steps (default: per-mode config default).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--output", type=str, default="output.png", help="Dashboard PNG output path.")
    parser.add_argument("--ksg-k", type=int, default=KSG_K_NEIGHBORS,
                         help="Number of nearest neighbors for the KSG estimator.")
    parser.add_argument("--te-lag", type=int, default=TE_LAG, help="Lag (samples) for transfer entropy.")
    return parser.parse_args(argv)


def _metrics_bar_panel(ax: plt.Axes, metrics: dict) -> None:
    labels = ["Mutual Info\n(bits)", "Transfer Entropy\n(bits)", "Noise Ratio\n(1 - r²)"]
    values = [metrics["mutual_info"], metrics["transfer_entropy"], metrics["noise_ratio"]]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.set_title("Information Thermodynamics")
    ax.set_ylabel("Value")


def build_micro_dashboard(result: MicroSimulationResult, metrics: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax0 = axes[0]
    positions = result.final_positions
    species = result.species
    active_mask = result.final_active_mask
    substrate_idx = result.substrate_indices

    scaffold_pts = positions[species == SCAFFOLD]
    input_pts = positions[species == INPUT]
    substrate_pts = positions[substrate_idx]
    inactive_pts = substrate_pts[~active_mask]
    active_pts = substrate_pts[active_mask]

    ax0.scatter(*scaffold_pts.T, c="#4c72b0", s=40, label="Scaffold", edgecolors="k", linewidths=0.3)
    ax0.scatter(*input_pts.T, c="#55a868", s=25, label="Input", edgecolors="k", linewidths=0.3)
    ax0.scatter(*inactive_pts.T, c="#b0b0b0", s=15, label="Substrate (Inactive)", alpha=0.7)
    if len(active_pts):
        ax0.scatter(*active_pts.T, c="#c44e52", s=25, label="Substrate (Active)", edgecolors="k", linewidths=0.3)
    ax0.set_title("Final Spatial State (Micro)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.set_aspect("equal")

    ax1 = axes[1]
    t = np.arange(len(result.time_series))
    ax1.plot(t, result.time_series[:, 0], label="Input Signal (count near scaffold)", color="#55a868")
    ax1.plot(t, result.time_series[:, 1], label="Output Signal (cumulative active)", color="#c44e52")
    ax1.set_title("Signal Time Series")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)

    _metrics_bar_panel(axes[2], metrics)

    fig.suptitle("BioCompute-Hub - Microscopic Agent-Based Simulation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def build_meso_dashboard(result: MesoSimulationResult, metrics: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax0 = axes[0]
    im = ax0.imshow(result.phi_field, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                     extent=[0, result.nx, 0, result.ny])
    fig.colorbar(im, ax=ax0, label="phi (condensate order parameter)")
    ax0.set_title("Final Condensate Phase Field (Meso)")
    ax0.set_xlabel("x (cells)")
    ax0.set_ylabel("y (cells)")

    ax1 = axes[1]
    t = np.arange(len(result.time_series))
    ax1.plot(t, result.time_series[:, 0], label="Input Source S(t)", color="#55a868", alpha=0.8)
    ax1.plot(t, result.time_series[:, 1], label="Output ΣC(t) (total reacted)", color="#c44e52")
    ax1.set_title("Signal Time Series")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Amplitude")
    ax1.legend(fontsize=8)

    _metrics_bar_panel(axes[2], metrics)

    fig.suptitle("BioCompute-Hub - Mesoscopic Phase-Field Simulation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)

    if args.mode == "micro":
        n_steps = args.n_steps or N_STEPS_DEFAULT
        log.info("Running microscopic Langevin agent-based simulation (%d steps, seed=%d)...",
                  n_steps, args.seed)
        result = run_micro_simulation(n_steps=n_steps, seed=args.seed)
        x, y = result.time_series[:, 0], result.time_series[:, 1]
        metrics = compute_information_metrics(x, y, k=args.ksg_k, lag=args.te_lag)
        fig = build_micro_dashboard(result, metrics)
    else:
        n_steps = args.n_steps or MESO_N_STEPS_DEFAULT
        log.info("Running mesoscopic Cahn-Hilliard phase-field simulation (%d steps, seed=%d)...",
                  n_steps, args.seed)
        try:
            result = run_meso_simulation(n_steps=n_steps, seed=args.seed)
        except FloatingPointError as exc:
            log.error("Meso field simulation diverged: %s", exc)
            return 1
        x, y = result.time_series[:, 0], result.time_series[:, 1]
        metrics = compute_information_metrics(x, y, k=args.ksg_k, lag=args.te_lag)
        fig = build_meso_dashboard(result, metrics)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Dashboard saved to %s", args.output)
    log.info("Information metrics: mutual_info=%.4f bits, transfer_entropy=%.4f bits, noise_ratio=%.4f",
              metrics["mutual_info"], metrics["transfer_entropy"], metrics["noise_ratio"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
