"""Mesoscopic Phase-Field Engine: Cahn-Hilliard + reaction-diffusion (FiPy).

Solves, on a periodic 2D finite-volume grid:

    d(phi)/dt = div( M * grad( phi^3 - phi - kappa * lap(phi) ) )     (Cahn-Hilliard)
    dC/dt     = D_C * lap(C) - k(phi) * C + S(t)                      (reaction-diffusion)

phi in [-1, 1] is the condensate order parameter (phi > 0 inside the
condensate phase); C is a signaling molecule whose degradation rate k(phi)
is switched to a higher value inside the condensate, and which is driven by
a localized, time-varying stochastic source S(t).

The 4th-order Cahn-Hilliard equation is split into the standard coupled
pair (phi, psi) - psi being the chemical potential - each 2nd order, and
solved with a semi-implicit linearization of the nonlinear bulk term
(psi = phi^3 - phi - kappa * lap(phi)), swept to a residual tolerance at
each timestep for nonlinear convergence. This is the standard FiPy
approach for the Cahn-Hilliard equation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from fipy import CellVariable, DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy.meshes import PeriodicGrid2D

from config import (
    D_C, GRID_DX, GRID_NX, GRID_NY, K_REACT_COND, K_REACT_SOL, KAPPA,
    MAX_SWEEPS, MESO_DT, MESO_N_STEPS_DEFAULT, MOBILITY_M, PHI_NOISE_STD,
    RANDOM_SEED, SOURCE_AMPLITUDE, SOURCE_NOISE_STD, SOURCE_RADIUS,
    SWEEP_RESIDUAL_TOL,
)


@dataclass
class MesoSimulationResult:
    phi_field: np.ndarray       # (ny, nx) final condensate order-parameter field
    c_field: np.ndarray          # (ny, nx) final signaling-molecule concentration field
    time_series: np.ndarray      # (n_steps+1, 2): columns [Input_Source_S, Output_Reacted_C]
    nx: int
    ny: int


class PhaseFieldSimulator:
    """Coupled Cahn-Hilliard / reaction-diffusion solver on a periodic grid."""

    def __init__(self, nx: int = GRID_NX, ny: int = GRID_NY, dx: float = GRID_DX,
                 dt: float = MESO_DT, seed: Optional[int] = RANDOM_SEED):
        self.nx, self.ny, self.dx, self.dt = nx, ny, dx, dt
        self.rng = np.random.default_rng(seed)

        self.mesh = PeriodicGrid2D(nx=nx, ny=ny, dx=dx, dy=dx)

        self.phi = CellVariable(name="phi", mesh=self.mesh, hasOld=True)
        self.psi = CellVariable(name="psi", mesh=self.mesh, hasOld=True)
        self.C = CellVariable(name="C", mesh=self.mesh, hasOld=True, value=0.0)
        self.k_field = CellVariable(name="k_field", mesh=self.mesh, value=K_REACT_SOL)
        self.source_var = CellVariable(name="source", mesh=self.mesh, value=0.0)

        self.phi.setValue(self.rng.normal(0.0, PHI_NOISE_STD, size=self.mesh.numberOfCells))

        dfdphi = self.phi ** 3 - self.phi
        d2fdphi2 = 3.0 * self.phi ** 2 - 1.0
        eq_phi_transient = TransientTerm(var=self.phi) == DiffusionTerm(coeff=MOBILITY_M, var=self.psi)
        eq_psi_implicit = (
            ImplicitSourceTerm(coeff=1.0, var=self.psi)
            == ImplicitSourceTerm(coeff=d2fdphi2, var=self.phi) - d2fdphi2 * self.phi + dfdphi
            - DiffusionTerm(coeff=KAPPA, var=self.phi)
        )
        self.eq_ch = eq_phi_transient & eq_psi_implicit

        self.eq_rd = (
            TransientTerm(var=self.C)
            == DiffusionTerm(coeff=D_C, var=self.C)
            - ImplicitSourceTerm(coeff=self.k_field, var=self.C)
            + self.source_var
        )

        centers = self.mesh.cellCenters.value  # (2, n_cells)
        cx = nx * dx / 2.0
        cy = ny * dx / 2.0
        r2 = (centers[0] - cx) ** 2 + (centers[1] - cy) ** 2
        self._source_profile = np.exp(-r2 / (2.0 * SOURCE_RADIUS ** 2))

    def _step_cahn_hilliard(self) -> None:
        self.phi.updateOld()
        self.psi.updateOld()
        residual = 1e5
        sweeps = 0
        while residual > SWEEP_RESIDUAL_TOL and sweeps < MAX_SWEEPS:
            residual = self.eq_ch.sweep(dt=self.dt)
            sweeps += 1

    def _step_reaction_diffusion(self) -> float:
        self.C.updateOld()
        self.k_field.setValue(np.where(self.phi.value > 0.0, K_REACT_COND, K_REACT_SOL))

        source_amplitude = max(0.0, SOURCE_AMPLITUDE + self.rng.normal(0.0, SOURCE_NOISE_STD))
        self.source_var.setValue(source_amplitude * self._source_profile)

        self.eq_rd.solve(var=self.C, dt=self.dt)
        return source_amplitude

    def step(self) -> tuple:
        """Advance both fields by one timestep. Returns (S_t, total_C)."""
        self._step_cahn_hilliard()
        s_t = self._step_reaction_diffusion()
        output_c = float(np.sum(self.C.value))
        return s_t, output_c

    def run(self, n_steps: int = MESO_N_STEPS_DEFAULT) -> MesoSimulationResult:
        time_series = np.zeros((n_steps + 1, 2))
        time_series[0] = (0.0, float(np.sum(self.C.value)))

        for t in range(1, n_steps + 1):
            s_t, output_c = self.step()
            time_series[t] = (s_t, output_c)
            if not np.all(np.isfinite(self.phi.value)) or not np.all(np.isfinite(self.C.value)):
                raise FloatingPointError(
                    f"Meso field solver diverged at step {t}/{n_steps} "
                    "(non-finite phi/C values) - reduce dt or timestep count."
                )

        phi_field = self.phi.value.reshape((self.ny, self.nx))
        c_field = self.C.value.reshape((self.ny, self.nx))
        return MesoSimulationResult(phi_field=phi_field, c_field=c_field,
                                     time_series=time_series, nx=self.nx, ny=self.ny)


def run_meso_simulation(n_steps: int = MESO_N_STEPS_DEFAULT,
                         seed: Optional[int] = RANDOM_SEED) -> MesoSimulationResult:
    """Convenience entry point used by main.py's CLI orchestrator."""
    simulator = PhaseFieldSimulator(seed=seed)
    return simulator.run(n_steps=n_steps)
