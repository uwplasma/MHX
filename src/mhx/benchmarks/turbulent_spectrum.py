"""Forced 2-D reduced-MHD turbulence and its magnetic cascade spectrum."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.benchmarks.turbulence import (
    TURBULENCE_DOMAIN,
    TurbulenceResult,
    _broadband_scalar_field,
    _finite_result,
    _run_turbulence_trajectory,
    _validate_turbulence_inputs,
    turbulent_initial_state,
)
from mhx.config import MeshConfig
from mhx.diagnostics.spectra import fit_power_law, magnetic_power_spectrum_2d
from mhx.equations.arakawa_reduced_mhd import arakawa_reduced_mhd_rhs
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics.spectral import laplacian
from mhx.state import ReducedMHDParams, ReducedMHDState

TURBULENT_SPECTRUM_SCHEMA = "mhx.validation.turbulent_spectrum.v1"
# The inertial range, k = 4 to 40, stays clear of the 2/3 dealiasing edge.
FIT_K_RANGE = (4, 40)
SLOPE_RANGE = (-1.7, -1.4)


@dataclass(frozen=True)
class TurbulentSpectrumResult:
    """Saved arrays and gates for turbulent spectrum validation."""

    turbulence: TurbulenceResult
    k_1d: np.ndarray
    power_1d: np.ndarray
    slope: float
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def _apply_2_3_dealiasing(state: ReducedMHDState) -> ReducedMHDState:
    """Apply the 2/3 dealiasing rule in Fourier space to prevent aliasing instability."""

    def dealias_array(field: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.ones_like(field, dtype=bool)
        for axis in range(field.ndim):
            n = field.shape[axis]
            k = jnp.fft.fftfreq(n) * n
            axis_mask = jnp.abs(k) <= (n / 3.0)
            shape = [1] * field.ndim
            shape[axis] = n
            mask = mask & jnp.reshape(axis_mask, shape)
        return jnp.real(jnp.fft.ifftn(jnp.where(mask, jnp.fft.fftn(field), 0.0)))

    return ReducedMHDState(psi=dealias_array(state.psi), omega=dealias_array(state.omega))


def run_turbulent_spectrum_validation(
    *,
    shape: tuple[int, int] = (256, 256),
    resistivity: float = 5.0e-5,
    viscosity: float = 5.0e-5,
    turbulent_flux_amplitude: float = 0.5,
    turbulent_flow_amplitude: float = 0.5,
    forcing_amplitude: float = 0.05,
    dt: float = 1.0e-3,
    t_end: float = 50.0,
    save_every: int = 1000,
    seed: int = 11,
) -> TurbulentSpectrumResult:
    """Run a 2-D reduced-MHD turbulence test to measure the cascade spectrum.

    The flow is driven by a static large-scale vorticity forcing at ``k = 1-3``
    and integrated with Arakawa brackets and 2/3 dealiasing. The gate checks
    that the saved histories are finite and that the inertial-range slope lies
    between ``-1.7`` and ``-1.4``.
    """
    _validate_turbulence_inputs(
        shape=shape,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
    )

    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=shape, lower=(0.0, 0.0), upper=TURBULENCE_DOMAIN)
    )
    turbulent_state = turbulent_initial_state(
        grid,
        seed=seed,
        flux_amplitude=turbulent_flux_amplitude,
        flow_amplitude=turbulent_flow_amplitude,
        kmin=1,
        kmax=4,
    )
    initial_state = ReducedMHDState(psi=turbulent_state.psi, omega=turbulent_state.omega)
    initial_state = _apply_2_3_dealiasing(initial_state)

    forcing_stream = _broadband_scalar_field(grid, seed=seed + 2027, kmin=1, kmax=3)
    forcing_omega = forcing_amplitude * laplacian(jnp.asarray(forcing_stream), lengths=grid.lengths)
    params = ReducedMHDParams(resistivity=resistivity, viscosity=viscosity)

    def forcing(state: ReducedMHDState) -> ReducedMHDState:
        dealiased_state = _apply_2_3_dealiasing(state)
        base = arakawa_reduced_mhd_rhs(dealiased_state, params, lengths=grid.lengths)
        rhs_with_forcing = ReducedMHDState(psi=base.psi, omega=base.omega + forcing_omega)
        return _apply_2_3_dealiasing(rhs_with_forcing)

    result = _run_turbulence_trajectory(
        initial_state,
        params,
        grid=grid,
        dt=dt,
        t_end=t_end,
        save_every=save_every,
        rhs=forcing,
        reconnection_proxy=False,
    )

    k_1d, power_1d = magnetic_power_spectrum_2d(result.psi[-1], domain_size=TURBULENCE_DOMAIN[0])
    k_fit_min, k_fit_max = FIT_K_RANGE
    slope, _ = fit_power_law(k_1d, power_1d, kmin=k_fit_min, kmax=k_fit_max)

    checks = {
        "finite_histories": _finite_result(result),
        "spectral_slope_in_range": bool(SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]),
    }
    diagnostics = {
        "schema": TURBULENT_SPECTRUM_SCHEMA,
        "shape": list(shape),
        "domain": [0.0, TURBULENCE_DOMAIN[0], 0.0, TURBULENCE_DOMAIN[1]],
        "dt": dt,
        "t_end": t_end,
        "initial_total_energy": float(result.total_energy[0]),
        "final_total_energy": float(result.total_energy[-1]),
        "max_vorticity": float(np.max(result.vorticity_linf)),
        "max_current": float(np.max(result.current_linf)),
        "spectral_slope": float(slope),
        "fit_k_range": [k_fit_min, k_fit_max],
    }
    validation = {
        "schema": "mhx.validation.turbulent_spectrum.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"min_slope": SLOPE_RANGE[0], "max_slope": SLOPE_RANGE[1]},
        "diagnostics": diagnostics,
    }

    return TurbulentSpectrumResult(
        turbulence=result,
        k_1d=k_1d,
        power_1d=power_1d,
        slope=float(slope),
        diagnostics=diagnostics,
        validation=validation,
    )


def write_turbulent_spectrum_validation(
    outdir: str | Path, **kwargs: Any
) -> tuple[Path, dict[str, Any]]:
    """Run the spectrum test and write its arrays, figure, validation and manifest.

    Keyword arguments are forwarded to :func:`run_turbulent_spectrum_validation`.
    ``turbulent_spectrum.npz`` keeps every saved frame of ``psi``, ``omega`` and
    the current density, plus the energies and the final 1-D spectrum.

    Returns:
        The path to ``manifest.json`` and the validation dictionary.
    """
    result = run_turbulent_spectrum_validation(**kwargs)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output_dir / "diagnostics.json"
    validation_path = output_dir / "validation.json"
    history_path = output_dir / "turbulent_spectrum.npz"

    diagnostics_path.write_text(json.dumps(result.diagnostics, indent=2, sort_keys=True))
    validation_path.write_text(json.dumps(result.validation, indent=2, sort_keys=True))

    t_result = result.turbulence
    payload: dict[str, Any] = {
        "schema": TURBULENT_SPECTRUM_SCHEMA,
        "time": t_result.time,
        "psi": t_result.psi,
        "omega": t_result.omega,
        "current_density": t_result.current_density,
        "magnetic_energy": t_result.magnetic_energy,
        "kinetic_energy": t_result.kinetic_energy,
        "total_energy": t_result.total_energy,
        "k_1d": result.k_1d,
        "power_1d": result.power_1d,
    }
    np.savez_compressed(history_path, **payload)

    summary_path = _write_turbulent_spectrum_summary(
        result, figure_dir / "turbulent_spectrum_summary.png"
    )

    outputs: dict[str, str] = {
        "diagnostics": diagnostics_path.name,
        "validation": validation_path.name,
        "history": history_path.name,
        "summary": str(summary_path.relative_to(output_dir)),
    }

    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        config=result.diagnostics,
        outputs=outputs,
        claim_level="validation",
        claim_scope="2D MHD turbulent cascade spectrum test with Arakawa brackets.",
    )
    return manifest_path, result.validation


def _write_turbulent_spectrum_summary(result: TurbulentSpectrumResult, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

    k_1d = result.k_1d
    power_1d = result.power_1d

    # Show k <= 100 only; the spectrum falls off in the dissipation range above it.
    shown = k_1d <= 100
    ax.loglog(
        k_1d[shown],
        power_1d[shown],
        label=f"Power Spectrum (t={result.turbulence.time[-1]:.1f})",
        color="teal",
        linewidth=2,
    )

    k_ref = np.linspace(10, 50, 100)
    ref_idx = int(np.argmin(np.abs(k_1d - 10)))
    scale = power_1d[ref_idx] * (10**1.5)
    ax.loglog(k_ref, scale * k_ref**-1.50, "r--", label=r"$k^{-1.50}$ (Iroshnikov-Kraichnan)")

    scale_kolmo = power_1d[ref_idx] * (10**1.66)
    ax.loglog(k_ref, scale_kolmo * k_ref**-1.66, "b--", label=r"$k^{-1.66}$ (Kolmogorov)")

    slope = result.diagnostics.get("spectral_slope", 0.0)
    passed = result.validation.get("passed", False)
    ax.set_title(f"Turbulent Magnetic Power Spectrum\nMeasured Slope: {slope:.3f} | Pass: {passed}")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$S_B(k)$")
    ax.legend(frameon=False)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
