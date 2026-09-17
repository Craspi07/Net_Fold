"""Layer 3: Spatial & Condensate-Modulated Kinetic Simulator.

An ODE-based signal propagation model over the protein interaction graph,
with a Condensate State Engine that partitions high-S_LLPS nodes between
a soluble and a condensate compartment via K_part = exp(-dG / kB*T), and
rescales local kinetics by an adjustable amplification/sequestration
factor gamma inside condensates. A lightweight Gillespie SSA is also
provided as a discrete-stochastic alternative.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

log = logging.getLogger("phasenet.layer3")
ProgressCB = Optional[Callable[[int, str], None]]

KB_T = 1.0  # normalized thermal energy unit
DELTA_G_SCALE = 6.0  # sharpness of the soluble<->condensate transition


def input_signal(t: float, kind: str, amplitude: float, rng: Optional[np.random.Generator] = None) -> float:
    if kind == "step":
        return amplitude if t >= 1.0 else 0.0
    if kind == "pulse":
        return amplitude if 1.0 <= t <= 3.0 else 0.0
    if kind == "gaussian":
        base = amplitude if t >= 1.0 else 0.0
        noise = (rng or np.random.default_rng()).normal(0, amplitude * 0.15)
        return max(0.0, base + noise)
    return 0.0


@dataclass
class SimulationResult:
    times: np.ndarray
    trajectories: Dict[str, np.ndarray]
    condensate_fraction: Dict[str, float]
    node_order: List[str]


class KineticsSimulator:
    """Discrete-time ODE simulator of signal propagation with LLPS kinetics."""

    def __init__(self, G: nx.Graph, gamma: float = 2.0, kpart_threshold: float = 0.5,
                 sim_time: float = 50.0, noise_amplitude: float = 0.1,
                 k_sol: float = 0.5, degradation: float = 0.05, n_points: int = 500):
        self.G = G
        self.gamma = gamma
        self.kpart_threshold = kpart_threshold
        self.sim_time = sim_time
        self.noise_amplitude = noise_amplitude
        self.k_sol = k_sol
        self.degradation = degradation
        self.n_points = n_points
        self.nodes = list(G.nodes())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self._rng = np.random.default_rng(42)

    def delta_g(self, s_llps: float) -> float:
        """Free-energy of partitioning: more negative -> favors condensate."""
        return -(s_llps - self.kpart_threshold) * DELTA_G_SCALE

    def k_part(self, s_llps: float) -> float:
        return float(np.exp(-self.delta_g(s_llps) / KB_T))

    def condensate_fraction(self, node: str) -> float:
        s_llps = self.G.nodes[node].get("s_llps", 0.0)
        if s_llps <= self.kpart_threshold:
            return 0.0
        kp = self.k_part(s_llps)
        return float(kp / (1.0 + kp))

    def _rate_multiplier(self, node: str, llps_enabled: bool) -> float:
        if not llps_enabled:
            return 1.0
        frac = self.condensate_fraction(node)
        return 1.0 + (self.gamma - 1.0) * frac

    def simulate(self, input_nodes: List[str], output_nodes: List[str], signal_kind: str = "step",
                 amplitude: float = 1.0, llps_enabled: bool = True,
                 progress_cb: ProgressCB = None) -> SimulationResult:
        if not self.nodes:
            raise ValueError("Graph has no nodes to simulate.")

        n = len(self.nodes)
        adjacency = nx.to_numpy_array(self.G, nodelist=self.nodes, weight="weight")
        rate_mult = np.array([self._rate_multiplier(node, llps_enabled) for node in self.nodes])
        input_idx = [self.index[node] for node in input_nodes if node in self.index]

        if progress_cb:
            progress_cb(10, "Integrating ODE signal propagation...")

        def rhs(t: float, C: np.ndarray) -> np.ndarray:
            dC = -self.degradation * C
            for i in range(n):
                if adjacency[i].any():
                    coupling = self.k_sol * rate_mult[i] * adjacency[i] * (C - C[i])
                    dC[i] += np.sum(coupling)
            for idx in input_idx:
                dC[idx] += input_signal(t, signal_kind, amplitude)
            return dC

        t_eval = np.linspace(0, self.sim_time, self.n_points)
        y0 = np.zeros(n)

        sol = solve_ivp(rhs, [0, self.sim_time], y0, t_eval=t_eval, method="RK45",
                         rtol=1e-4, atol=1e-6)

        if not sol.success:
            log.warning("ODE integration did not converge cleanly: %s", sol.message)

        trajectories: Dict[str, np.ndarray] = {}
        for node in self.nodes:
            series = sol.y[self.index[node]]
            if signal_kind == "gaussian" and node in output_nodes + input_nodes:
                series = series + self._rng.normal(0, amplitude * self.noise_amplitude * 0.5, size=series.shape)
                series = np.clip(series, 0, None)
            trajectories[node] = series

        cond_frac = {node: self.condensate_fraction(node) for node in self.nodes}

        if progress_cb:
            progress_cb(100, "Kinetic simulation complete.")

        return SimulationResult(times=sol.t, trajectories=trajectories,
                                 condensate_fraction=cond_frac, node_order=self.nodes)

    def gillespie_ssa(self, input_nodes: List[str], output_nodes: List[str],
                       max_time: float = 50.0, max_steps: int = 20000) -> SimulationResult:
        """Discrete stochastic alternative: simple SSA over production/decay/transfer events."""
        n = len(self.nodes)
        adjacency = nx.to_numpy_array(self.G, nodelist=self.nodes, weight="weight")
        rate_mult = np.array([self._rate_multiplier(node, True) for node in self.nodes])
        input_idx = {self.index[node] for node in input_nodes if node in self.index}

        counts = np.zeros(n)
        t = 0.0
        times = [0.0]
        history = [counts.copy()]

        for _ in range(max_steps):
            if t >= max_time:
                break
            production = np.array([1.0 if i in input_idx else 0.0 for i in range(n)])
            decay = self.degradation * counts
            transfer = np.zeros(n)
            for i in range(n):
                if adjacency[i].any():
                    transfer[i] = self.k_sol * rate_mult[i] * np.sum(adjacency[i] * counts)

            rates = production + decay + transfer
            total_rate = rates.sum()
            if total_rate <= 0:
                break
            tau = self._rng.exponential(1.0 / total_rate)
            t += tau
            choice = self._rng.choice(n * 3, p=np.concatenate([production, decay, transfer]) / total_rate)
            idx = choice % n
            if choice < n:
                counts[idx] += 1
            elif choice < 2 * n:
                counts[idx - n] = max(0, counts[idx - n] - 1)
            else:
                counts[idx - 2 * n] += 1
            times.append(t)
            history.append(counts.copy())

        times_arr = np.array(times)
        history_arr = np.array(history)
        trajectories = {node: history_arr[:, self.index[node]] for node in self.nodes}
        cond_frac = {node: self.condensate_fraction(node) for node in self.nodes}
        return SimulationResult(times=times_arr, trajectories=trajectories,
                                 condensate_fraction=cond_frac, node_order=self.nodes)
