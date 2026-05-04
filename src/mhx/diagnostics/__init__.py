"""Standardized diagnostics shared by MHX workflows."""

from mhx.diagnostics.reduced_mhd import (
    kinetic_energy,
    magnetic_energy,
    total_energy,
    trajectory_energies,
)

__all__ = ["kinetic_energy", "magnetic_energy", "total_energy", "trajectory_energies"]

