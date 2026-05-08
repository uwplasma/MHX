"""Validation benchmark catalog artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mhx.io import write_manifest

BENCHMARK_CATALOG_SCHEMA = "mhx.benchmark_catalog.v1"


@dataclass(frozen=True)
class BenchmarkCatalogEntry:
    """One reproducible validation or benchmark workflow entry."""

    name: str
    command: str
    schema: str
    purpose: str
    expected_outputs: tuple[str, ...]
    literature_anchor: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible entry metadata."""
        return {
            "name": self.name,
            "command": self.command,
            "schema": self.schema,
            "purpose": self.purpose,
            "expected_outputs": list(self.expected_outputs),
            "literature_anchor": self.literature_anchor,
        }


def validation_catalog_entries() -> tuple[BenchmarkCatalogEntry, ...]:
    """Return the active reviewer-facing FAST validation catalog."""
    return (
        BenchmarkCatalogEntry(
            name="linear_tearing_fast",
            command=(
                "mhx benchmark run --config examples/linear_tearing.toml "
                "--outdir outputs/benchmarks/linear_tearing_fast --gif"
            ),
            schema="mhx.reduced_mhd.trajectory.v1",
            purpose="End-to-end reduced-MHD run, figures, GIF, report, and validation.",
            expected_outputs=(
                "manifest.json",
                "diagnostics.json",
                "trajectory.npz",
                "figures/energy_history.png",
                "figures/flux_final.png",
                "figures/mode_amplitude.png",
                "figures/flux_movie.gif",
                "report.json",
                "report.md",
                "validation.json",
            ),
            literature_anchor="Reduced-MHD pseudo-spectral tearing smoke workflow.",
        ),
        BenchmarkCatalogEntry(
            name="resistive_decay",
            command="mhx benchmark decay --outdir outputs/benchmarks/resistive_decay",
            schema="mhx.validation.resistive_decay.v1",
            purpose="Exact Fourier-mode resistive diffusion law and energy decay.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "decay_history.npz",
                "figures/decay_amplitude.png",
                "figures/decay_energy.png",
                "figures/decay_relative_error.png",
            ),
            literature_anchor="Linear resistive induction equation used by tearing theory.",
        ),
        BenchmarkCatalogEntry(
            name="reconnection_scaling",
            command="mhx benchmark scaling --outdir outputs/benchmarks/reconnection_scaling",
            schema="mhx.validation.reconnection_scaling.v1",
            purpose="Analytic FKR, plasmoid, and ideal-tearing scaling exponents.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "scaling_history.npz",
                "figures/fkr_scaling.png",
                "figures/plasmoid_scaling.png",
                "figures/ideal_tearing_scaling.png",
            ),
            literature_anchor="FKR, Loureiro-Schekochihin-Cowley, Pucci-Velli.",
        ),
        BenchmarkCatalogEntry(
            name="fkr_window",
            command="mhx benchmark fkr-window --outdir outputs/benchmarks/fkr_window",
            schema="mhx.validation.fkr_window.v1",
            purpose="Constant-psi FKR regime-window inequalities.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "fkr_window.npz",
                "figures/fkr_constant_psi_window.png",
            ),
            literature_anchor="FKR constant-psi and Coppi-regime separation.",
        ),
        BenchmarkCatalogEntry(
            name="harris_delta_prime",
            command=(
                "mhx benchmark harris-delta-prime "
                "--outdir outputs/benchmarks/harris_delta_prime"
            ),
            schema="mhx.validation.harris_delta_prime.v1",
            purpose=(
                "Numerical Harris-sheet outer-region Delta-prime solve against "
                "the analytic FKR matching formula."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "harris_delta_prime.npz",
                "figures/harris_delta_prime.png",
            ),
            literature_anchor=(
                "Harris-sheet outer tearing equation and FKR constant-psi matching."
            ),
        ),
        BenchmarkCatalogEntry(
            name="fkr_growth_rate",
            command="mhx benchmark fkr-growth --outdir outputs/benchmarks/fkr_growth_rate",
            schema="mhx.validation.fkr_growth_rate.v1",
            purpose=(
                "Asymptotic FKR growth-rate gate assembled from numerical "
                "Harris Delta-prime matching."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "fkr_growth_rate.npz",
                "figures/fkr_growth_rate.png",
            ),
            literature_anchor=(
                "FKR constant-psi growth gamma tau_a ~ "
                "S_a^(-3/5)(ka)^(2/5)(Delta'a)^(4/5)."
            ),
        ),
        BenchmarkCatalogEntry(
            name="linear_tearing_eigenvalue",
            command=(
                "mhx benchmark linear-tearing-eigenvalue "
                "--outdir outputs/benchmarks/linear_tearing_eigenvalue"
            ),
            schema="mhx.validation.linear_tearing_eigenvalue.v1",
            purpose=(
                "Direct finite-difference Harris-sheet reduced-MHD tearing "
                "eigenvalue gate against the published k=0.5, S=1000 growth rate."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "linear_tearing_eigenvalue.npz",
                "figures/linear_tearing_eigenvalue.png",
            ),
            literature_anchor=(
                "MacTaggart 2019 and MacTaggart-Stewart 2017 reduced-MHD "
                "Harris-sheet eigenproblem, gamma approximately 0.0131."
            ),
        ),
        BenchmarkCatalogEntry(
            name="linear_tearing_dispersion",
            command=(
                "mhx benchmark linear-tearing-dispersion "
                "--outdir outputs/benchmarks/linear_tearing_dispersion"
            ),
            schema="mhx.validation.linear_tearing_dispersion.v1",
            purpose=(
                "Small Harris-sheet finite-domain tearing dispersion gate with "
                "unstable-band, stable-control, residual, and reference-point checks."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "linear_tearing_dispersion.npz",
                "figures/linear_tearing_dispersion.png",
            ),
            literature_anchor=(
                "FKR unstable interval and MacTaggart reduced-MHD Harris-sheet "
                "reference point at S=1000, k=0.5."
            ),
        ),
        BenchmarkCatalogEntry(
            name="linear_tearing_layer",
            command=(
                "mhx benchmark linear-tearing-layer "
                "--outdir outputs/benchmarks/linear_tearing_layer"
            ),
            schema="mhx.validation.linear_tearing_layer.v1",
            purpose=(
                "FAST Harris-sheet eigenfunction-shape gate checking flow-layer "
                "narrowing with S, outer-flux width stability, and eigen residuals."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "linear_tearing_layer.npz",
                "figures/linear_tearing_layer.png",
            ),
            literature_anchor=(
                "Classical tearing localization near the resonant surface; broad "
                "FAST trend gate, not an asymptotic exponent claim."
            ),
        ),
        BenchmarkCatalogEntry(
            name="linear_tearing_timedomain",
            command=(
                "mhx benchmark linear-tearing-timedomain "
                "--outdir outputs/benchmarks/linear_tearing_timedomain"
            ),
            schema="mhx.validation.linear_tearing_timedomain.v1",
            purpose=(
                "Time-domain RK4 replay of the direct Harris-sheet tearing "
                "eigenmode with growth-rate fit and mode-alignment checks."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "linear_tearing_timedomain.npz",
                "figures/linear_tearing_timedomain.png",
            ),
            literature_anchor=(
                "Same MacTaggart reduced-MHD Harris eigenproblem as the direct "
                "eigenvalue gate, validating time-domain growth recovery."
            ),
        ),
        BenchmarkCatalogEntry(
            name="linearized_rhs",
            command="mhx benchmark linearized-rhs --outdir outputs/benchmarks/linearized_rhs",
            schema="mhx.validation.linearized_rhs.v1",
            purpose="JAX JVP versus centered finite-difference linearized RHS.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "linearized_rhs.npz",
                "figures/linearized_rhs_errors.png",
            ),
            literature_anchor="Matrix-free linearization prerequisite for tearing modes.",
        ),
        BenchmarkCatalogEntry(
            name="reduced_mhd_eigenmode",
            command=(
                "mhx benchmark reduced-mhd-eigenmode "
                "--outdir outputs/benchmarks/reduced_mhd_eigenmode"
            ),
            schema="mhx.validation.reduced_mhd_linear_eigenmode.v1",
            purpose="Zero-state reduced-MHD psi/omega Fourier diffusion eigenmodes.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "reduced_mhd_linear_eigenmode.npz",
                "figures/reduced_mhd_linear_eigenmode_errors.png",
            ),
            literature_anchor="Linear reduced-MHD diffusion-block eigenvalues.",
        ),
        BenchmarkCatalogEntry(
            name="cosine_equilibrium_linearization",
            command=(
                "mhx benchmark cosine-equilibrium-linearization "
                "--outdir outputs/benchmarks/cosine_equilibrium_linearization"
            ),
            schema="mhx.validation.cosine_equilibrium_linearization.v1",
            purpose=(
                "Analytic nonzero-current-sheet JVP couplings around psi0=A cos(y)."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "cosine_equilibrium_linearization.npz",
                "figures/cosine_equilibrium_linearization_errors.png",
            ),
            literature_anchor=(
                "Reduced-MHD Poisson-bracket linearization for tearing operators."
            ),
        ),
        BenchmarkCatalogEntry(
            name="periodic_current_sheet_eigenvalue",
            command=(
                "mhx benchmark current-sheet-eigenvalue "
                "--outdir outputs/benchmarks/periodic_current_sheet_eigenvalue"
            ),
            schema="mhx.validation.periodic_current_sheet_eigenvalue.v1",
            purpose=(
                "Tiny dense spectrum of the nonzero periodic current-sheet JVP "
                "with gauge-mode and no-spurious-growth gates."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "periodic_current_sheet_eigenvalue.npz",
                "figures/periodic_current_sheet_spectrum.png",
            ),
            literature_anchor=(
                "Current-sheet reduced-MHD linear operator prerequisite for "
                "calibrated FKR/Coppi tearing benchmarks."
            ),
        ),
        BenchmarkCatalogEntry(
            name="periodic_current_sheet_timedomain",
            command=(
                "mhx benchmark current-sheet-timedomain "
                "--outdir outputs/benchmarks/periodic_current_sheet_timedomain"
            ),
            schema="mhx.validation.periodic_current_sheet_timedomain.v1",
            purpose=(
                "RK4 time-domain replay of a real decaying eigenmode of the "
                "periodic current-sheet JVP operator."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "periodic_current_sheet_timedomain.npz",
                "figures/periodic_current_sheet_timedomain.png",
            ),
            literature_anchor=(
                "Reduced-MHD linear operator consistency: dq/dt=Lq must replay "
                "q(t)=exp(lambda t)q(0) for a dense JVP eigenmode."
            ),
        ),
        BenchmarkCatalogEntry(
            name="periodic_current_sheet_nonlinear_bridge",
            command=(
                "mhx benchmark current-sheet-nonlinear-bridge "
                "--outdir outputs/benchmarks/periodic_current_sheet_nonlinear_bridge"
            ),
            schema="mhx.validation.periodic_current_sheet_nonlinear_bridge.v1",
            purpose=(
                "Nonlinear RK4 trajectory-map differentiability check: centered "
                "finite differences converge to the JAX JVP tangent."
            ),
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "periodic_current_sheet_nonlinear_bridge.npz",
                "figures/periodic_current_sheet_nonlinear_bridge.png",
            ),
            literature_anchor=(
                "Differentiable reduced-MHD solver validation needed before "
                "adjoints, inverse design, and neural-ODE surrogate datasets."
            ),
        ),
        BenchmarkCatalogEntry(
            name="diffusion_eigenvalue",
            command=(
                "mhx benchmark diffusion-eigenvalue "
                "--outdir outputs/benchmarks/diffusion_eigenvalue"
            ),
            schema="mhx.validation.diffusion_eigenvalue.v1",
            purpose="Matrix-free Rayleigh quotient and residual on a Fourier eigenpair.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "diffusion_eigenvalue.npz",
                "figures/diffusion_eigenvalue_errors.png",
            ),
            literature_anchor="Periodic spectral Laplacian eigenpairs.",
        ),
        BenchmarkCatalogEntry(
            name="power_iteration",
            command="mhx benchmark power-iteration --outdir outputs/benchmarks/power_iteration",
            schema="mhx.validation.power_iteration.v1",
            purpose="Dominant-eigenpair loop on a known diagonal matrix-free operator.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "power_iteration_history.npz",
                "figures/power_iteration_history.png",
            ),
            literature_anchor="Classical power iteration for matrix-free operators.",
        ),
        BenchmarkCatalogEntry(
            name="arnoldi",
            command="mhx benchmark arnoldi --outdir outputs/benchmarks/arnoldi",
            schema="mhx.validation.arnoldi.v1",
            purpose="Krylov Ritz spectrum on a non-normal fixture.",
            expected_outputs=(
                "diagnostics.json",
                "validation.json",
                "arnoldi_spectrum.npz",
                "figures/arnoldi_ritz_values.png",
            ),
            literature_anchor="Arnoldi/Krylov eigensolver scaffold for tearing modes.",
        ),
        BenchmarkCatalogEntry(
            name="timing",
            command="mhx benchmark timing --outdir outputs/benchmarks/timing",
            schema="mhx.benchmark.timing.v1",
            purpose="FAST wall-clock and Python-allocation performance artifacts.",
            expected_outputs=(
                "timing.json",
                "timing.md",
                "figures/timing_summary.png",
                "manifest.json",
            ),
            literature_anchor="Reproducible performance tracking, not a physics gate.",
        ),
    )


def write_benchmark_catalog(outdir: str | Path) -> tuple[Path, Path]:
    """Write validation catalog JSON, Markdown, and manifest artifacts."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = validation_catalog_entries()
    catalog = {
        "schema": BENCHMARK_CATALOG_SCHEMA,
        "entries": [entry.to_dict() for entry in entries],
    }
    json_path = output_dir / "benchmark_catalog.json"
    markdown_path = output_dir / "benchmark_catalog.md"
    manifest_path = output_dir / "manifest.json"
    json_path.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_catalog_markdown(entries), encoding="utf-8")
    write_manifest(
        manifest_path,
        config={"schema": BENCHMARK_CATALOG_SCHEMA},
        outputs={
            "catalog_json": json_path.name,
            "catalog_markdown": markdown_path.name,
        },
    )
    return json_path, markdown_path


def _catalog_markdown(entries: tuple[BenchmarkCatalogEntry, ...]) -> str:
    rows = "\n".join(
        f"| `{entry.name}` | `{entry.schema}` | `{entry.command}` | {entry.purpose} |"
        for entry in entries
    )
    return (
        "# MHX validation benchmark catalog\n\n"
        "This file is generated by `mhx benchmark catalog`. It is a compact "
        "reviewer-facing index of active FAST validation gates, schemas, commands, "
        "and artifact expectations.\n\n"
        "| Name | Schema | Command | Purpose |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n"
    )
