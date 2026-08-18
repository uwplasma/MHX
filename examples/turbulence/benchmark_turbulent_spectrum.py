"""2D MHD turbulent cascade spectrum test."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from mhx.config import MeshConfig
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.numerics.spectral import laplacian
from mhx.equations.arakawa_reduced_mhd import arakawa_reduced_mhd_rhs

# Reuse the canonical turbulence machinery from the benchmarks package rather
# than duplicating it inside ``examples/``.
from mhx.benchmarks.turbulence import (
    TURBULENCE_DOMAIN,
    TurbulenceResult,
    _broadband_scalar_field,
    _finite_result,
    _run_turbulence_trajectory,
    _validate_turbulence_inputs,
    turbulent_initial_state,
)

TURBULENT_SPECTRUM_SCHEMA = "mhx.validation.turbulent_spectrum.v1"


@dataclass(frozen=True)
class TurbulentSpectrumResult:
    """Saved arrays and gates for turbulent spectrum validation."""

    turbulence: TurbulenceResult
    k_1d: np.ndarray
    power_1d: np.ndarray
    slope: float
    diagnostics: dict[str, Any]
    validation: dict[str, Any]


def _compute_magnetic_power_spectrum(
    psi_field: np.ndarray, domain_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1D isotropic power spectrum matching continuous energy density."""
    psi_hat = np.fft.fftn(psi_field)
    n = psi_field.shape[0]

    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

    k_squared = kx_grid**2 + ky_grid**2
    physical_derivative_factor = (2.0 * np.pi / domain_size) ** 2
    power_B_2d = k_squared * physical_derivative_factor * np.abs(psi_hat) ** 2

    dx_dy = (domain_size / n) ** 2
    scaling_factor = dx_dy / (n**2)
    power_B_2d *= scaling_factor

    k_radius = np.round(np.sqrt(k_squared)).astype(int)
    k_max = n // 2
    k_1d = np.arange(1, k_max)
    power_1d = np.zeros(k_max - 1)

    dk = 2.0 * np.pi / domain_size
    for r in k_1d:
        power_1d[r - 1] = np.sum(power_B_2d[k_radius == r]) / dk

    return k_1d, power_1d


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
    """Run a 2-D reduced-MHD turbulence test to measure the cascade spectrum."""
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

    k_1d, power_1d = _compute_magnetic_power_spectrum(
        result.psi[-1], domain_size=TURBULENCE_DOMAIN[0]
    )

    k_max_fit = shape[0] / 3.0
    mask = (k_1d >= 5) & (k_1d <= k_max_fit)
    k_fit = k_1d[mask]
    power_fit = power_1d[mask]
    slope, _ = np.polyfit(np.log10(k_fit), np.log10(power_fit), 1)

    passed_slope = bool(-1.7 <= slope <= -1.4)

    checks = {
        "finite_histories": _finite_result(result),
        "spectral_slope_in_range": passed_slope,
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
    }
    validation = {
        "schema": "mhx.validation.turbulent_spectrum.gates.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {"min_slope": -1.7, "max_slope": -1.4},
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
    """Write turbulent spectrum artifacts and validation."""
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

    summary_path = _write_turbulent_spectrum_summary(result, figure_dir / "turbulent_spectrum_summary.png")

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

    ax.loglog(k_1d, power_1d, label=f"Power Spectrum (t={result.turbulence.time[-1]:.1f})",
              color="teal", linewidth=2)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="2D MHD Turbulent Cascade Spectrum Test")
    parser.add_argument("--nx", type=int, default=256, help="Grid points in X")
    parser.add_argument("--ny", type=int, default=256, help="Grid points in Y")
    parser.add_argument("--eta", type=float, default=5.0e-5, help="Resistivity")
    parser.add_argument("--nu", type=float, default=5.0e-5, help="Viscosity")
    parser.add_argument("--dt", type=float, default=1.0e-3, help="Time step")
    parser.add_argument("--t-end", dest="t_end", type=float, default=50.0, help="Simulation end time")
    parser.add_argument("--save-every", type=int, default=1000, help="Save interval")
    parser.add_argument("--outdir", type=str, default="outputs/turbulent_spectrum_output", help="Output directory")
    args = parser.parse_args()

    print("Running 2D MHD Turbulent Cascade Spectrum Test...")
    print(f"Grid: {args.nx}x{args.ny} | dt: {args.dt} | t_end: {args.t_end}")
    print(f"Output directory: {Path(args.outdir).resolve()}")

    manifest_path, validation = write_turbulent_spectrum_validation(
        outdir=args.outdir,
        shape=(args.nx, args.ny),
        resistivity=args.eta,
        viscosity=args.nu,
        dt=args.dt,
        t_end=args.t_end,
        save_every=args.save_every,
    )

    passed = validation.get("passed", False)
    print(f"wrote {manifest_path}")
    print(f"passed={passed}")
    if not passed:
        print("Failed checks:")
        for check, passed_check in validation.get("checks", {}).items():
            if not passed_check:
                print(f"  - {check}")


if __name__ == "__main__":
    main()
