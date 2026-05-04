"""Standardized diagnostics shared by MHX workflows."""

from mhx.diagnostics.reduced_mhd import (
    fit_exponential_growth,
    kinetic_energy,
    magnetic_energy,
    mode_amplitude,
    select_fit_window,
    total_energy,
    trajectory_energies,
    trajectory_mode_amplitude,
)

__all__ = [
    "fit_exponential_growth",
    "kinetic_energy",
    "magnetic_energy",
    "mode_amplitude",
    "select_fit_window",
    "total_energy",
    "trajectory_energies",
    "trajectory_mode_amplitude",
]
