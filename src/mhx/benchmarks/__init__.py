"""Benchmark problem builders."""

from mhx.benchmarks.decay import (
    RESISTIVE_DECAY_SCHEMA,
    ResistiveDecayResult,
    resistive_decay_rate,
    run_resistive_decay_validation,
    write_resistive_decay_validation,
)
from mhx.benchmarks.eigenvalue import (
    ARNOLDI_SCHEMA,
    DIFFUSION_EIGENVALUE_SCHEMA,
    POWER_ITERATION_SCHEMA,
    ArnoldiValidationResult,
    DiffusionEigenvalueResult,
    PowerIterationValidationResult,
    run_arnoldi_validation,
    run_diffusion_eigenvalue_validation,
    run_power_iteration_validation,
    write_arnoldi_validation,
    write_diffusion_eigenvalue_validation,
    write_power_iteration_validation,
)
from mhx.benchmarks.fkr import (
    FKR_WINDOW_SCHEMA,
    FKRWindowResult,
    run_fkr_window_validation,
    write_fkr_window_validation,
)
from mhx.benchmarks.linearized import (
    LINEARIZED_RHS_SCHEMA,
    LinearizedRHSResult,
    run_linearized_rhs_validation,
    write_linearized_rhs_validation,
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
from mhx.benchmarks.timing import (
    TIMING_BENCHMARK_SCHEMA,
    TimingBenchmarkResult,
    TimingCaseResult,
    run_timing_benchmark,
    write_timing_benchmark,
)

__all__ = [
    "ARNOLDI_SCHEMA",
    "ArnoldiValidationResult",
    "FKRConstantPsiEstimate",
    "FKRWindowResult",
    "DIFFUSION_EIGENVALUE_SCHEMA",
    "DiffusionEigenvalueResult",
    "FKR_WINDOW_SCHEMA",
    "LINEARIZED_RHS_SCHEMA",
    "POWER_ITERATION_SCHEMA",
    "LinearizedRHSResult",
    "PlasmoidScalingEstimate",
    "PowerIterationValidationResult",
    "RECONNECTION_SCALING_SCHEMA",
    "RESISTIVE_DECAY_SCHEMA",
    "TIMING_BENCHMARK_SCHEMA",
    "ReconnectionScalingResult",
    "ResistiveDecayResult",
    "TimingBenchmarkResult",
    "TimingCaseResult",
    "fkr_constant_psi_estimate",
    "harris_sheet_delta_prime",
    "ideal_tearing_aspect_ratio",
    "loglog_slope",
    "linear_tearing_initial_state",
    "loureiro_plasmoid_estimate",
    "resistive_decay_rate",
    "run_arnoldi_validation",
    "run_diffusion_eigenvalue_validation",
    "run_fkr_window_validation",
    "run_linearized_rhs_validation",
    "run_power_iteration_validation",
    "run_reconnection_scaling_validation",
    "run_resistive_decay_validation",
    "run_linear_tearing_smoke",
    "run_timing_benchmark",
    "validate_run",
    "write_arnoldi_validation",
    "write_diffusion_eigenvalue_validation",
    "write_fkr_window_validation",
    "write_linearized_rhs_validation",
    "write_power_iteration_validation",
    "write_reconnection_scaling_validation",
    "write_resistive_decay_validation",
    "write_run_report",
    "write_timing_benchmark",
]
