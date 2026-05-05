"""Benchmark problem builders."""

from mhx.benchmarks.decay import (
    RESISTIVE_DECAY_SCHEMA,
    ResistiveDecayResult,
    resistive_decay_rate,
    run_resistive_decay_validation,
    write_resistive_decay_validation,
)
from mhx.benchmarks.report import validate_run, write_run_report
from mhx.benchmarks.scaling import (
    RECONNECTION_SCALING_SCHEMA,
    ReconnectionScalingResult,
    loglog_slope,
    run_reconnection_scaling_validation,
    write_reconnection_scaling_validation,
)
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
    "RECONNECTION_SCALING_SCHEMA",
    "RESISTIVE_DECAY_SCHEMA",
    "ReconnectionScalingResult",
    "ResistiveDecayResult",
    "fkr_constant_psi_estimate",
    "harris_sheet_delta_prime",
    "ideal_tearing_aspect_ratio",
    "loglog_slope",
    "linear_tearing_initial_state",
    "loureiro_plasmoid_estimate",
    "resistive_decay_rate",
    "run_reconnection_scaling_validation",
    "run_resistive_decay_validation",
    "run_linear_tearing_smoke",
    "validate_run",
    "write_reconnection_scaling_validation",
    "write_resistive_decay_validation",
    "write_run_report",
]
