"""Standardized diagnostics shared by MHX workflows."""

from mhx.diagnostics.reduced_mhd import (
    DiagnosticContext,
    DiagnosticSpec,
    DiagnosticsRegistry,
    compute_reduced_mhd_diagnostics,
    default_diagnostics_registry,
    fit_exponential_growth,
    kinetic_energy,
    magnetic_divergence_linf,
    magnetic_energy,
    mode_amplitude,
    select_fit_window,
    total_energy,
    trajectory_energies,
    trajectory_mode_amplitude,
)

__all__ = [
    "DiagnosticContext",
    "DiagnosticSpec",
    "DiagnosticsRegistry",
    "compute_reduced_mhd_diagnostics",
    "default_diagnostics_registry",
    "fit_exponential_growth",
    "kinetic_energy",
    "magnetic_divergence_linf",
    "magnetic_energy",
    "mode_amplitude",
    "select_fit_window",
    "total_energy",
    "trajectory_energies",
    "trajectory_mode_amplitude",
]
