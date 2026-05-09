"""Generate documentation figures for numerical-validation pages."""

from __future__ import annotations

import shutil
from pathlib import Path

from mhx.benchmarks import (
    write_arnoldi_validation,
    write_cosine_equilibrium_linearization_validation,
    write_diffusion_eigenvalue_validation,
    write_fkr_growth_rate_validation,
    write_fkr_window_validation,
    write_harris_delta_prime_validation,
    write_linear_tearing_dispersion_validation,
    write_linear_tearing_eigenvalue_validation,
    write_linear_tearing_layer_validation,
    write_linear_tearing_timedomain_validation,
    write_linearized_rhs_validation,
    write_nonlinear_duration_audit,
    write_nonlinear_energy_budget_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
    write_power_iteration_validation,
    write_reconnection_scaling_validation,
    write_reduced_mhd_linear_eigenmode_validation,
    write_resistive_decay_validation,
    write_timing_benchmark,
)


def main() -> None:
    """Regenerate deterministic validation figures used by the docs."""
    run_dir = Path("outputs/docs_validation/resistive_decay")
    scaling_run_dir = Path("outputs/docs_validation/reconnection_scaling")
    fkr_window_run_dir = Path("outputs/docs_validation/fkr_window")
    fkr_growth_run_dir = Path("outputs/docs_validation/fkr_growth_rate")
    harris_delta_prime_run_dir = Path("outputs/docs_validation/harris_delta_prime")
    linear_tearing_dispersion_run_dir = Path("outputs/docs_validation/linear_tearing_dispersion")
    linear_tearing_eigenvalue_run_dir = Path("outputs/docs_validation/linear_tearing_eigenvalue")
    linear_tearing_layer_run_dir = Path("outputs/docs_validation/linear_tearing_layer")
    linear_tearing_timedomain_run_dir = Path("outputs/docs_validation/linear_tearing_timedomain")
    linearized_run_dir = Path("outputs/docs_validation/linearized_rhs")
    reduced_mhd_eigenmode_run_dir = Path("outputs/docs_validation/reduced_mhd_eigenmode")
    cosine_linearization_run_dir = Path(
        "outputs/docs_validation/cosine_equilibrium_linearization"
    )
    periodic_current_sheet_run_dir = Path(
        "outputs/docs_validation/periodic_current_sheet_eigenvalue"
    )
    periodic_current_sheet_timedomain_run_dir = Path(
        "outputs/docs_validation/periodic_current_sheet_timedomain"
    )
    periodic_current_sheet_nonlinear_bridge_run_dir = Path(
        "outputs/docs_validation/periodic_current_sheet_nonlinear_bridge"
    )
    nonlinear_energy_budget_run_dir = Path("outputs/docs_validation/nonlinear_energy_budget")
    nonlinear_duration_audit_run_dir = Path("outputs/docs_validation/nonlinear_duration_audit")
    diffusion_eigen_run_dir = Path("outputs/docs_validation/diffusion_eigenvalue")
    power_iteration_run_dir = Path("outputs/docs_validation/power_iteration")
    arnoldi_run_dir = Path("outputs/docs_validation/arnoldi")
    timing_run_dir = Path("outputs/docs_validation/timing")
    decay_docs_dir = Path("docs/_static/validation/exact_decay")
    scaling_docs_dir = Path("docs/_static/validation/reconnection_scaling")
    fkr_window_docs_dir = Path("docs/_static/validation/fkr_window")
    fkr_growth_docs_dir = Path("docs/_static/validation/fkr_growth_rate")
    harris_delta_prime_docs_dir = Path("docs/_static/validation/harris_delta_prime")
    linear_tearing_dispersion_docs_dir = Path("docs/_static/validation/linear_tearing_dispersion")
    linear_tearing_eigenvalue_docs_dir = Path("docs/_static/validation/linear_tearing_eigenvalue")
    linear_tearing_layer_docs_dir = Path("docs/_static/validation/linear_tearing_layer")
    linear_tearing_timedomain_docs_dir = Path("docs/_static/validation/linear_tearing_timedomain")
    linearized_docs_dir = Path("docs/_static/validation/linearized_rhs")
    reduced_mhd_eigenmode_docs_dir = Path("docs/_static/validation/reduced_mhd_eigenmode")
    cosine_linearization_docs_dir = Path(
        "docs/_static/validation/cosine_equilibrium_linearization"
    )
    periodic_current_sheet_docs_dir = Path(
        "docs/_static/validation/periodic_current_sheet_eigenvalue"
    )
    periodic_current_sheet_timedomain_docs_dir = Path(
        "docs/_static/validation/periodic_current_sheet_timedomain"
    )
    periodic_current_sheet_nonlinear_bridge_docs_dir = Path(
        "docs/_static/validation/periodic_current_sheet_nonlinear_bridge"
    )
    nonlinear_energy_budget_docs_dir = Path("docs/_static/validation/nonlinear_energy_budget")
    nonlinear_duration_audit_docs_dir = Path("docs/_static/validation/nonlinear_duration_audit")
    diffusion_eigen_docs_dir = Path("docs/_static/validation/diffusion_eigenvalue")
    power_iteration_docs_dir = Path("docs/_static/validation/power_iteration")
    arnoldi_docs_dir = Path("docs/_static/validation/arnoldi")
    timing_docs_dir = Path("docs/_static/performance")
    write_resistive_decay_validation(run_dir)
    write_reconnection_scaling_validation(scaling_run_dir)
    write_fkr_window_validation(fkr_window_run_dir)
    write_fkr_growth_rate_validation(fkr_growth_run_dir)
    write_harris_delta_prime_validation(harris_delta_prime_run_dir)
    write_linear_tearing_dispersion_validation(linear_tearing_dispersion_run_dir)
    write_linear_tearing_eigenvalue_validation(linear_tearing_eigenvalue_run_dir)
    write_linear_tearing_layer_validation(linear_tearing_layer_run_dir)
    write_linear_tearing_timedomain_validation(linear_tearing_timedomain_run_dir)
    write_linearized_rhs_validation(linearized_run_dir)
    write_reduced_mhd_linear_eigenmode_validation(reduced_mhd_eigenmode_run_dir)
    write_cosine_equilibrium_linearization_validation(cosine_linearization_run_dir)
    write_periodic_current_sheet_eigenvalue_validation(periodic_current_sheet_run_dir)
    write_periodic_current_sheet_timedomain_validation(
        periodic_current_sheet_timedomain_run_dir
    )
    write_periodic_current_sheet_nonlinear_bridge_validation(
        periodic_current_sheet_nonlinear_bridge_run_dir
    )
    write_nonlinear_energy_budget_validation(nonlinear_energy_budget_run_dir)
    write_nonlinear_duration_audit(nonlinear_duration_audit_run_dir)
    write_diffusion_eigenvalue_validation(diffusion_eigen_run_dir)
    write_power_iteration_validation(power_iteration_run_dir)
    write_arnoldi_validation(arnoldi_run_dir)
    write_timing_benchmark(timing_run_dir, repeats=1, warmups=0)
    decay_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decay_amplitude.png", "decay_energy.png", "decay_relative_error.png"):
        shutil.copy2(run_dir / "figures" / name, decay_docs_dir / name)
    scaling_docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png"):
        shutil.copy2(scaling_run_dir / "figures" / name, scaling_docs_dir / name)
    fkr_window_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        fkr_window_run_dir / "figures" / "fkr_constant_psi_window.png",
        fkr_window_docs_dir / "fkr_constant_psi_window.png",
    )
    fkr_growth_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        fkr_growth_run_dir / "figures" / "fkr_growth_rate.png",
        fkr_growth_docs_dir / "fkr_growth_rate.png",
    )
    harris_delta_prime_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        harris_delta_prime_run_dir / "figures" / "harris_delta_prime.png",
        harris_delta_prime_docs_dir / "harris_delta_prime.png",
    )
    linear_tearing_dispersion_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linear_tearing_dispersion_run_dir
        / "figures"
        / "linear_tearing_dispersion.png",
        linear_tearing_dispersion_docs_dir / "linear_tearing_dispersion.png",
    )
    linear_tearing_eigenvalue_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linear_tearing_eigenvalue_run_dir
        / "figures"
        / "linear_tearing_eigenvalue.png",
        linear_tearing_eigenvalue_docs_dir / "linear_tearing_eigenvalue.png",
    )
    linear_tearing_layer_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linear_tearing_layer_run_dir / "figures" / "linear_tearing_layer.png",
        linear_tearing_layer_docs_dir / "linear_tearing_layer.png",
    )
    linear_tearing_timedomain_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linear_tearing_timedomain_run_dir
        / "figures"
        / "linear_tearing_timedomain.png",
        linear_tearing_timedomain_docs_dir / "linear_tearing_timedomain.png",
    )
    linearized_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        linearized_run_dir / "figures" / "linearized_rhs_errors.png",
        linearized_docs_dir / "linearized_rhs_errors.png",
    )
    reduced_mhd_eigenmode_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        reduced_mhd_eigenmode_run_dir
        / "figures"
        / "reduced_mhd_linear_eigenmode_errors.png",
        reduced_mhd_eigenmode_docs_dir / "reduced_mhd_linear_eigenmode_errors.png",
    )
    cosine_linearization_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        cosine_linearization_run_dir
        / "figures"
        / "cosine_equilibrium_linearization_errors.png",
        cosine_linearization_docs_dir / "cosine_equilibrium_linearization_errors.png",
    )
    periodic_current_sheet_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        periodic_current_sheet_run_dir
        / "figures"
        / "periodic_current_sheet_spectrum.png",
        periodic_current_sheet_docs_dir / "periodic_current_sheet_spectrum.png",
    )
    periodic_current_sheet_timedomain_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        periodic_current_sheet_timedomain_run_dir
        / "figures"
        / "periodic_current_sheet_timedomain.png",
        periodic_current_sheet_timedomain_docs_dir / "periodic_current_sheet_timedomain.png",
    )
    periodic_current_sheet_nonlinear_bridge_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        periodic_current_sheet_nonlinear_bridge_run_dir
        / "figures"
        / "periodic_current_sheet_nonlinear_bridge.png",
        periodic_current_sheet_nonlinear_bridge_docs_dir
        / "periodic_current_sheet_nonlinear_bridge.png",
    )
    nonlinear_energy_budget_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        nonlinear_energy_budget_run_dir / "figures" / "nonlinear_energy_budget.png",
        nonlinear_energy_budget_docs_dir / "nonlinear_energy_budget.png",
    )
    nonlinear_duration_audit_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        nonlinear_duration_audit_run_dir / "figures" / "nonlinear_duration_audit.png",
        nonlinear_duration_audit_docs_dir / "nonlinear_duration_audit.png",
    )
    diffusion_eigen_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        diffusion_eigen_run_dir / "figures" / "diffusion_eigenvalue_errors.png",
        diffusion_eigen_docs_dir / "diffusion_eigenvalue_errors.png",
    )
    power_iteration_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        power_iteration_run_dir / "figures" / "power_iteration_history.png",
        power_iteration_docs_dir / "power_iteration_history.png",
    )
    arnoldi_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        arnoldi_run_dir / "figures" / "arnoldi_ritz_values.png",
        arnoldi_docs_dir / "arnoldi_ritz_values.png",
    )
    timing_docs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        timing_run_dir / "figures" / "timing_summary.png",
        timing_docs_dir / "timing_summary.png",
    )


if __name__ == "__main__":
    main()
