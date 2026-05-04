"""Benchmark problem builders."""

from mhx.benchmarks.report import validate_run, write_run_report
from mhx.benchmarks.tearing import linear_tearing_initial_state, run_linear_tearing_smoke
from mhx.benchmarks.theory import (
    FKRConstantPsiEstimate,
    PlasmoidScalingEstimate,
    fkr_constant_psi_estimate,
    harris_sheet_delta_prime,
    ideal_tearing_aspect_ratio,
    loureiro_plasmoid_estimate,
)

__all__ = [
    "FKRConstantPsiEstimate",
    "PlasmoidScalingEstimate",
    "fkr_constant_psi_estimate",
    "harris_sheet_delta_prime",
    "ideal_tearing_aspect_ratio",
    "linear_tearing_initial_state",
    "loureiro_plasmoid_estimate",
    "run_linear_tearing_smoke",
    "validate_run",
    "write_run_report",
]
