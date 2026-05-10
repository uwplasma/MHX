"""Exact resistive-decay validation benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from mhx.diagnostics import (
    fit_exponential_growth,
    magnetic_energy,
    mode_amplitude,
    trajectory_mode_amplitude,
)
from mhx.equations.reduced_mhd import reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.plotting import plot_decay_amplitude, plot_decay_energy, plot_decay_relative_error
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState, ReducedMHDTrajectory
from mhx.time_integrators import evolve_rk4

RESISTIVE_DECAY_SCHEMA = "mhx.validation.resistive_decay.v1"


@dataclass(frozen=True)
class ResistiveDecayResult:
    """Numerical and exact histories for one Fourier-mode diffusion benchmark."""

    trajectory: ReducedMHDTrajectory
    exact_psi: Array
    numerical_amplitude: Array
    exact_amplitude: Array
    numerical_energy: Array
    exact_energy: Array
    relative_amplitude_error: Array
    relative_energy_error: Array
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def resistive_decay_rate(
    *,
    resistivity: float,
    mode: tuple[int, int],
    lengths: tuple[float, float],
) -> float:
    r"""Return the exact scalar decay rate ``η|k|²`` for one periodic mode."""
    kx = 2.0 * np.pi * mode[0] / lengths[0]
    ky = 2.0 * np.pi * mode[1] / lengths[1]
    return float(resistivity * (kx**2 + ky**2))


def run_resistive_decay_validation(
    *,
    shape: tuple[int, int] = (32, 32),
    mode: tuple[int, int] = (1, 0),
    resistivity: float = 5.0e-2,
    t1: float = 1.0,
    dt: float = 1.0e-2,
    save_every: int = 1,
    max_relative_amplitude_error: float = 1.0e-6,
    max_relative_energy_error: float = 2.0e-6,
    max_decay_rate_relative_error: float = 1.0e-6,
) -> ResistiveDecayResult:
    """Run a literature-anchored exact Fourier-mode resistive-diffusion gate."""
    jax_enable_x64 = configure_jax(enable_x64=True)
    grid = CartesianGrid(shape=shape, lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    psi0 = grid.cosinusoid(mode=mode)
    state0 = ReducedMHDState(psi=psi0, omega=jnp.zeros_like(psi0))
    params = ReducedMHDParams(resistivity=resistivity, viscosity=0.0)
    steps = max(1, round(t1 / dt))

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return reduced_mhd_rhs(state, params, lengths=grid.lengths)

    trajectory = evolve_rk4(state0, rhs, dt=dt, steps=steps, save_every=save_every)
    rate = resistive_decay_rate(resistivity=resistivity, mode=mode, lengths=grid.lengths)
    times = trajectory.times
    exact_multiplier = jnp.exp(-rate * times)
    exact_psi = exact_multiplier[-1] * psi0
    numerical_amplitude = trajectory_mode_amplitude(trajectory, mode=mode)
    exact_amplitude = mode_amplitude(state0, mode=mode) * exact_multiplier
    numerical_energy = jnp.asarray(
        [
            magnetic_energy(
                ReducedMHDState(
                    psi=trajectory.states.psi[index],
                    omega=trajectory.states.omega[index],
                ),
                lengths=grid.lengths,
            )
            for index in range(trajectory.times.shape[0])
        ]
    )
    exact_energy = magnetic_energy(state0, lengths=grid.lengths) * jnp.exp(-2.0 * rate * times)
    tiny = jnp.finfo(numerical_amplitude.dtype).tiny
    relative_amplitude_error = jnp.abs(numerical_amplitude - exact_amplitude) / jnp.maximum(
        jnp.abs(exact_amplitude), tiny
    )
    relative_energy_error = jnp.abs(numerical_energy - exact_energy) / jnp.maximum(
        jnp.abs(exact_energy), tiny
    )
    fitted_decay_rate = -float(fit_exponential_growth(times, numerical_amplitude))
    decay_rate_relative_error = abs(fitted_decay_rate - rate) / max(abs(rate), 1.0e-300)
    field_l2_relative_error = float(
        jnp.linalg.norm(trajectory.states.psi[-1] - exact_psi)
        / jnp.maximum(jnp.linalg.norm(exact_psi), tiny)
    )
    energy_monotone = bool(jnp.all(jnp.diff(numerical_energy) <= 1.0e-14))
    checks = {
        "amplitude_matches_exp_minus_eta_k2_t": bool(
            float(jnp.max(relative_amplitude_error)) <= max_relative_amplitude_error
        ),
        "energy_matches_exp_minus_2_eta_k2_t": bool(
            float(jnp.max(relative_energy_error)) <= max_relative_energy_error
        ),
        "fitted_decay_rate_matches_eta_k2": bool(
            decay_rate_relative_error <= max_decay_rate_relative_error
        ),
        "magnetic_energy_monotone_nonincreasing": energy_monotone,
        "final_field_l2_error_small": field_l2_relative_error <= max_relative_amplitude_error,
    }
    diagnostics = {
        "schema": RESISTIVE_DECAY_SCHEMA,
        "shape": list(shape),
        "mode": list(mode),
        "resistivity": resistivity,
        "jax_enable_x64": jax_enable_x64,
        "t1": float(t1),
        "dt": float(dt),
        "save_every": int(save_every),
        "n_steps": float(steps),
        "decay_rate_eta_k_squared": rate,
        "fitted_decay_rate": fitted_decay_rate,
        "decay_rate_relative_error": decay_rate_relative_error,
        "max_relative_amplitude_error": float(jnp.max(relative_amplitude_error)),
        "max_relative_energy_error": float(jnp.max(relative_energy_error)),
        "final_field_l2_relative_error": field_l2_relative_error,
        "reference": "single Fourier mode of resistive reduced-MHD induction equation",
    }
    validation = {
        "schema": "mhx.validation.resistive_decay.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_relative_amplitude_error": max_relative_amplitude_error,
            "max_relative_energy_error": max_relative_energy_error,
            "max_decay_rate_relative_error": max_decay_rate_relative_error,
        },
        "diagnostics": diagnostics,
    }
    return ResistiveDecayResult(
        trajectory=trajectory,
        exact_psi=exact_psi,
        numerical_amplitude=numerical_amplitude,
        exact_amplitude=exact_amplitude,
        numerical_energy=numerical_energy,
        exact_energy=exact_energy,
        relative_amplitude_error=relative_amplitude_error,
        relative_energy_error=relative_energy_error,
        diagnostics=diagnostics,
        validation=validation,
    )


def write_resistive_decay_validation(
    outdir: str | Path,
    **kwargs: Any,
) -> tuple[Path, dict[str, Any]]:
    """Run the exact-decay validation and write JSON, NPZ, plots, and manifest."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_resistive_decay_validation(**kwargs)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "decay_history.npz"
    manifest_path = output_dir / "manifest.json"
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(result.validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        history_path,
        schema=RESISTIVE_DECAY_SCHEMA,
        time=np.asarray(result.trajectory.times),
        numerical_amplitude=np.asarray(result.numerical_amplitude),
        exact_amplitude=np.asarray(result.exact_amplitude),
        numerical_energy=np.asarray(result.numerical_energy),
        exact_energy=np.asarray(result.exact_energy),
        relative_amplitude_error=np.asarray(result.relative_amplitude_error),
        relative_energy_error=np.asarray(result.relative_energy_error),
    )

    figure_dir = output_dir / "figures"
    amplitude_path = plot_decay_amplitude(
        result.trajectory.times,
        result.numerical_amplitude,
        result.exact_amplitude,
        path=figure_dir / "decay_amplitude.png",
    )
    error_path = plot_decay_relative_error(
        result.trajectory.times,
        result.relative_amplitude_error,
        result.relative_energy_error,
        path=figure_dir / "decay_relative_error.png",
    )
    energy_path = plot_decay_energy(
        result.trajectory.times,
        result.numerical_energy,
        result.exact_energy,
        path=figure_dir / "decay_energy.png",
    )
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs={
            "diagnostics": diagnostics_path.name,
            "validation": validation_path.name,
            "history": history_path.name,
            "decay_amplitude": str(amplitude_path.relative_to(output_dir)),
            "decay_relative_error": str(error_path.relative_to(output_dir)),
            "decay_energy": str(energy_path.relative_to(output_dir)),
        },
        claim_level="validation",
        claim_scope="Exact resistive-decay manufactured-solution validation.",
    )
    return manifest_path, result.validation
