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
