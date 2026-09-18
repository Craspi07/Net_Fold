"""Microscopic Agent-Based Dynamics: overdamped Langevin / Brownian dynamics.

Simulates stochastic multivalent assembly (Lennard-Jones-style attractive
clustering of Scaffold particles, forming a condensate) coupled with a
discrete Gillespie-style catalytic signaling event: a diffusing Input
(ligand) particle that comes within a critical radius of a Scaffold
particle triggers conversion of an Inactive substrate molecule to Active,
with per-timestep probability p = k_cat * dt.

All pairwise interactions are fully vectorized over NumPy arrays - the only
Python-level loop is the outer, inherently-sequential time-stepping loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from config import (
    BOX_SIZE, CONVERSION_RADIUS, DT, GAMMA_LIGAND, GAMMA_SCAFFOLD, K_CAT,
    LJ_CUTOFF, LJ_EPSILON, LJ_FORCE_CAP, LJ_SIGMA, N_INPUT, N_PARTICLES,
    N_SCAFFOLD, N_STEPS_DEFAULT, RANDOM_SEED, D_LIGAND, D_SCAFFOLD,
)

# Species labels
SCAFFOLD = 0
INPUT = 1
SUBSTRATE = 2  # activation state tracked separately via `self.active`


def _periodic_pairwise_displacement(positions: np.ndarray, box_size: float) -> np.ndarray:
    """Minimum-image pairwise displacement r_i - r_j under periodic boundary conditions.

    Returns an (N, N, 2) array; diff[i, j] = positions[i] - positions[j], wrapped
    into (-L/2, L/2] along each axis so interactions respect the periodic box.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    diff -= box_size * np.round(diff / box_size)
    return diff


def _cross_periodic_distance(a: np.ndarray, b: np.ndarray, box_size: float) -> np.ndarray:
    """Minimum-image pairwise Euclidean distance between two point sets a (M,2), b (K,2) -> (M,K)."""
    diff = a[:, None, :] - b[None, :, :]
    diff -= box_size * np.round(diff / box_size)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


@dataclass
class MicroSimulationResult:
    trajectory: np.ndarray            # (n_steps+1, N, 2) positions over time
    species: np.ndarray                # (N,) int labels: SCAFFOLD / INPUT / SUBSTRATE
    active_history: np.ndarray         # (n_steps+1, n_substrate) bool, substrate activation state
    time_series: np.ndarray            # (n_steps+1, 2): columns [Input_Signal, Output_Signal]
    substrate_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def final_positions(self) -> np.ndarray:
        return self.trajectory[-1]

    @property
    def final_active_mask(self) -> np.ndarray:
        return self.active_history[-1]


class LangevinSimulator:
    """2D overdamped Langevin dynamics with Euler-Maruyama integration.

    x_{t+dt} = x_t - (1/gamma) * grad_U(x_t) * dt + sqrt(2*D*dt) * xi,
    xi ~ N(0, 1) i.i.d. per particle per dimension. grad_U is derived from a
    Lennard-Jones-style soft-core potential applied pairwise between all
    particles (force-capped for stability), which drives Scaffold particles
    to nucleate an attractive cluster (a discrete analogue of phase
    separation) that recruits nearby Substrate particles.
    """

    def __init__(self, n_particles: int = N_PARTICLES, n_scaffold: int = N_SCAFFOLD,
                 n_input: int = N_INPUT, box_size: float = BOX_SIZE, dt: float = DT,
                 seed: Optional[int] = RANDOM_SEED):
        if n_scaffold + n_input >= n_particles:
            raise ValueError("n_scaffold + n_input must be < n_particles (remainder is substrate).")

        self.n_particles = n_particles
        self.box_size = box_size
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        self.species = np.empty(n_particles, dtype=int)
        self.species[:n_scaffold] = SCAFFOLD
        self.species[n_scaffold:n_scaffold + n_input] = INPUT
        self.species[n_scaffold + n_input:] = SUBSTRATE

        self.diffusion = np.where(self.species == INPUT, D_LIGAND, D_SCAFFOLD)
        self.gamma = np.where(self.species == INPUT, GAMMA_LIGAND, GAMMA_SCAFFOLD)

        self.positions = self.rng.uniform(0.0, box_size, size=(n_particles, 2))

        self.substrate_indices = np.where(self.species == SUBSTRATE)[0]
        self.input_indices = np.where(self.species == INPUT)[0]
        self.scaffold_indices = np.where(self.species == SCAFFOLD)[0]
        self.active = np.zeros(len(self.substrate_indices), dtype=bool)

    def _lj_forces(self) -> np.ndarray:
        """Vectorized, force-capped Lennard-Jones forces on every particle."""
        diff = _periodic_pairwise_displacement(self.positions, self.box_size)  # (N, N, 2)
        r2 = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(r2, np.inf)  # exclude self-interaction
        r2 = np.maximum(r2, (0.5 * LJ_SIGMA) ** 2)  # soft-core floor: prevent divide-by-near-zero

        sr2 = (LJ_SIGMA ** 2) / r2
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2
        r = np.sqrt(r2)

        force_mag = (24.0 * LJ_EPSILON / r) * (2.0 * sr12 - sr6)
        force_mag = np.clip(force_mag, -LJ_FORCE_CAP, LJ_FORCE_CAP)
        within_cutoff = r < LJ_CUTOFF
        force_mag = np.where(within_cutoff, force_mag, 0.0)

        # Force on i from j points along diff[i, j] = pos_i - pos_j.
        force_vectors = (force_mag / r)[:, :, None] * diff  # (N, N, 2)
        return np.sum(force_vectors, axis=1)  # (N, 2)

    def _step_positions(self) -> None:
        forces = self._lj_forces()
        drift = (forces / self.gamma[:, None]) * self.dt
        noise_std = np.sqrt(2.0 * self.diffusion * self.dt)
        xi = self.rng.standard_normal(size=self.positions.shape)
        diffusion_term = noise_std[:, None] * xi

        self.positions = (self.positions + drift + diffusion_term) % self.box_size

    def _catalysis_step(self) -> Tuple[int, int]:
        """Gillespie-style trigger: Input within R_c of a Scaffold converts one
        Inactive substrate (nearest to that input) to Active with probability
        p = k_cat * dt. Returns (input_signal_count, newly_activated_count).
        """
        if len(self.input_indices) == 0 or len(self.scaffold_indices) == 0:
            return 0, 0

        input_pos = self.positions[self.input_indices]
        scaffold_pos = self.positions[self.scaffold_indices]
        dist = _cross_periodic_distance(input_pos, scaffold_pos, self.box_size)  # (n_input, n_scaffold)
        min_dist = dist.min(axis=1)
        near_mask = min_dist < CONVERSION_RADIUS
        input_signal = int(np.sum(near_mask))
        if input_signal == 0:
            return 0, 0

        near_input_positions = input_pos[near_mask]
        p = K_CAT * self.dt
        trial = self.rng.random(size=near_input_positions.shape[0]) < p
        newly_activated = 0

        for pos in near_input_positions[trial]:
            inactive_mask = ~self.active
            if not np.any(inactive_mask):
                break
            substrate_pos = self.positions[self.substrate_indices[inactive_mask]]
            local_dist = _cross_periodic_distance(pos[None, :], substrate_pos, self.box_size)[0]
            nearest_local = np.argmin(local_dist)
            target_idx = np.where(inactive_mask)[0][nearest_local]
            self.active[target_idx] = True
            newly_activated += 1

        return input_signal, newly_activated

    def step(self) -> Tuple[int, int]:
        """Advance the system by one timestep: forces + Euler-Maruyama + catalysis."""
        self._step_positions()
        return self._catalysis_step()

    def run(self, n_steps: int = N_STEPS_DEFAULT) -> MicroSimulationResult:
        """Run the full simulation and return spatial history + signal time series."""
        n_substrate = len(self.substrate_indices)
        trajectory = np.empty((n_steps + 1, self.n_particles, 2))
        active_history = np.empty((n_steps + 1, n_substrate), dtype=bool)
        time_series = np.zeros((n_steps + 1, 2))

        trajectory[0] = self.positions
        active_history[0] = self.active
        cumulative_active = int(np.sum(self.active))
        time_series[0] = (0.0, cumulative_active)

        for t in range(1, n_steps + 1):
            input_signal, newly_activated = self.step()
            cumulative_active += newly_activated
            trajectory[t] = self.positions
            active_history[t] = self.active
            time_series[t] = (input_signal, cumulative_active)

        return MicroSimulationResult(
            trajectory=trajectory,
            species=self.species,
            active_history=active_history,
            time_series=time_series,
            substrate_indices=self.substrate_indices,
        )


def run_micro_simulation(n_steps: int = N_STEPS_DEFAULT,
                          seed: Optional[int] = RANDOM_SEED) -> MicroSimulationResult:
    """Convenience entry point used by main.py's CLI orchestrator."""
    simulator = LangevinSimulator(seed=seed)
    return simulator.run(n_steps=n_steps)
