"""Layer 3: Spatial & Condensate-Modulated Kinetic Simulator.

An ODE-based signal propagation model over the protein interaction graph,
with a Condensate State Engine that partitions high-S_LLPS nodes between
a soluble and a condensate compartment via K_part = exp(-dG / kB*T), and
rescales local kinetics by an adjustable amplification/sequestration
factor gamma inside condensates. A lightweight Gillespie SSA is also
provided as a discrete-stochastic alternative.

The hot numeric loops (ODE coupling term, Gillespie SSA core) are
numba-jitted so a single simulation runs in roughly the time of a compiled
loop rather than a pure-Python one; ``simulate_task`` is a plain
module-level, picklable entry point so this module doubles as the unit of
work dispatched to a ``ProcessPoolExecutor`` by Layers 4 and 5, which need
many independent simulations (noise realizations, knockouts, sensitivity
samples) run in parallel, off the GIL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation without numba
    NUMBA_AVAILABLE = False

    def njit(*jit_args, **jit_kwargs):
        def decorator(func):
            return func
        if len(jit_args) == 1 and callable(jit_args[0]) and not jit_kwargs:
            return jit_args[0]
        return decorator

log = logging.getLogger("phasenet.layer3")
ProgressCB = Optional[Callable[[int, str], None]]

KB_T = 1.0  # normalized thermal energy unit
DELTA_G_SCALE = 6.0  # sharpness of the soluble<->condensate transition


@njit(cache=True)
def _ode_coupling(C: np.ndarray, adjacency: np.ndarray, rate_mult: np.ndarray, k_sol: float) -> np.ndarray:
    """Numba-jitted diffusive coupling term: sum_j k_sol * rate_mult[i] * A_ij * (C_j - C_i)."""
    n = C.shape[0]
    dC = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            w = adjacency[i, j]
            if w != 0.0:
                acc += w * (C[j] - C[i])
        dC[i] = k_sol * rate_mult[i] * acc
    return dC


@njit(cache=True)
def _gillespie_core(adjacency: np.ndarray, rate_mult: np.ndarray, degradation: float, k_sol: float,
                     input_mask: np.ndarray, max_time: float, max_steps: int, seed: int):
    """Numba-jitted Gillespie SSA: production at inputs, first-order decay, graph-coupled transfer."""
    np.random.seed(seed)
    n = adjacency.shape[0]
    counts = np.zeros(n)
    t = 0.0
    max_records = max_steps + 1
    times = np.zeros(max_records)
    history = np.zeros((max_records, n))
    history[0, :] = counts
    record = 1

    for _ in range(max_steps):
        if t >= max_time:
            break
        production = np.zeros(n)
        for i in range(n):
            if input_mask[i]:
                production[i] = 1.0
        decay = degradation * counts
        transfer = np.zeros(n)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                w = adjacency[i, j]
                if w != 0.0:
                    acc += w * counts[j]
            transfer[i] = k_sol * rate_mult[i] * acc

        total_rate = 0.0
        for i in range(n):
            total_rate += production[i] + decay[i] + transfer[i]
        if total_rate <= 0.0:
            break

        tau = -np.log(np.random.random()) / total_rate
        t += tau

        r = np.random.random() * total_rate
        cum = 0.0
        chosen = -1
        event_type = -1
        for i in range(n):
            cum += production[i]
            if r <= cum:
                chosen = i
                event_type = 0
                break
        if chosen == -1:
            for i in range(n):
                cum += decay[i]
                if r <= cum:
                    chosen = i
                    event_type = 1
                    break
        if chosen == -1:
            for i in range(n):
                cum += transfer[i]
                if r <= cum:
                    chosen = i
                    event_type = 2
                    break
        if chosen == -1:
            chosen = n - 1
            event_type = 2

        if event_type == 0:
            counts[chosen] += 1.0
        elif event_type == 1:
            if counts[chosen] > 0.0:
                counts[chosen] -= 1.0
        else:
            counts[chosen] += 1.0

        times[record] = t
        history[record, :] = counts
        record += 1
        if record >= max_records:
            break

    return times[:record], history[:record]


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
                 k_sol: float = 0.5, degradation: float = 0.05, n_points: int = 500,
                 seed: Optional[int] = None):
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
        self._rng = np.random.default_rng(seed)

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
            dC = -self.degradation * C + _ode_coupling(C, adjacency, rate_mult, self.k_sol)
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
        """Discrete stochastic alternative, delegating the hot loop to numba."""
        adjacency = nx.to_numpy_array(self.G, nodelist=self.nodes, weight="weight")
        rate_mult = np.array([self._rate_multiplier(node, True) for node in self.nodes])
        input_mask = np.array([node in input_nodes for node in self.nodes])
        seed = int(self._rng.integers(0, 2**31 - 1))

        times, history = _gillespie_core(adjacency, rate_mult, self.degradation, self.k_sol,
                                          input_mask, max_time, max_steps, seed)

        trajectories = {node: history[:, self.index[node]] for node in self.nodes}
        cond_frac = {node: self.condensate_fraction(node) for node in self.nodes}
        return SimulationResult(times=times, trajectories=trajectories,
                                 condensate_fraction=cond_frac, node_order=self.nodes)


def simulate_task(G: nx.Graph, input_nodes: List[str], output_nodes: List[str],
                   gamma: float = 2.0, kpart_threshold: float = 0.5, sim_time: float = 50.0,
                   noise_amplitude: float = 0.1, k_sol: float = 0.5, degradation: float = 0.05,
                   n_points: int = 500, signal_kind: str = "step", amplitude: float = 1.0,
                   llps_enabled: bool = True, seed: Optional[int] = None) -> SimulationResult:
    """Plain, picklable entry point: build a simulator from raw parameters and run it.

    This is the unit of work submitted to a ``ProcessPoolExecutor`` - it
    takes only picklable arguments (a NetworkX graph and scalars) and
    returns a picklable ``SimulationResult``, so Layers 4 and 5 can fan
    many of these out across separate OS processes instead of blocking the
    GIL in QThreads.
    """
    simulator = KineticsSimulator(G, gamma=gamma, kpart_threshold=kpart_threshold, sim_time=sim_time,
                                   noise_amplitude=noise_amplitude, k_sol=k_sol, degradation=degradation,
                                   n_points=n_points, seed=seed)
    return simulator.simulate(input_nodes, output_nodes, signal_kind=signal_kind,
                               amplitude=amplitude, llps_enabled=llps_enabled)
