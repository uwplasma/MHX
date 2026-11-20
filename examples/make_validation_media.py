"""Regenerate the deterministic validation figures used by the documentation.

Calls the ``mhx.benchmarks`` validation writers, each of which writes its figures
under ``RUN_ROOT/<name>/figures``, then copies the figures the docs embed into
``DOCS_ROOT``. The full set takes a long time (tens of minutes on a CPU,
dominated by the double-Harris long run and convergence study); to refresh one
figure, comment out the others below. Run from the repository root:

    python examples/make_validation_media.py
"""

import shutil
import time
from pathlib import Path

from mhx.benchmarks import (
    double_harris_seeded_long_run_presets,
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
    write_nonlinear_energy_budget_validation,
    write_periodic_current_sheet_eigenvalue_validation,
    write_periodic_current_sheet_nonlinear_bridge_validation,
    write_periodic_current_sheet_timedomain_validation,
    write_periodic_double_harris_convergence_validation,
    write_periodic_double_harris_nonlinear_growth_validation,
    write_periodic_double_harris_seeded_long_run_validation,
    write_power_iteration_validation,
    write_reconnection_scaling_validation,
    write_reduced_mhd_linear_eigenmode_validation,
    write_resistive_decay_validation,
)
from mhx.runtime import configure_jax

# --- Input parameters ------------------------------------------------------
RUN_ROOT = Path("outputs/docs_validation")  # where the writers put their full outputs
DOCS_ROOT = Path("docs/_static/validation")  # figures embedded in the docs
LONG_RUN_SHAPE = (64, 64)  # grid of the double-Harris seeded long run (movies)
CONVERGENCE_RESOLUTIONS = (16, 24)  # resolutions of the double-Harris convergence study
CONVERGENCE_DT_VALUES = (2.0e-2, 1.0e-2)  # time steps of the convergence study
CONVERGENCE_T_END = 8.0  # end time of the convergence study
CONVERGENCE_FIT_WINDOW = (0.0, 4.0)  # growth-rate fit window of the convergence study


def publish(run_name, docs_dir, *names):
    """Copy ``figures/<name>`` of one run into the docs folder."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(RUN_ROOT / run_name / "figures" / name, docs_dir / name)
        print(f"  copied {docs_dir / name}", flush=True)


# --- Run -------------------------------------------------------------------
configure_jax(enable_x64=True)
print(f"Regenerating validation figures: runs in {RUN_ROOT}, docs in {DOCS_ROOT}", flush=True)
start = time.perf_counter()

write_resistive_decay_validation(RUN_ROOT / "resistive_decay")
publish("resistive_decay", DOCS_ROOT / "exact_decay",
        "decay_amplitude.png", "decay_energy.png", "decay_relative_error.png")
print("resistive decay done", flush=True)

write_reconnection_scaling_validation(RUN_ROOT / "reconnection_scaling")
publish("reconnection_scaling", DOCS_ROOT / "reconnection_scaling",
        "fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png")
print("reconnection scaling done", flush=True)

write_fkr_window_validation(RUN_ROOT / "fkr_window")
publish("fkr_window", DOCS_ROOT / "fkr_window", "fkr_constant_psi_window.png")

write_fkr_growth_rate_validation(RUN_ROOT / "fkr_growth_rate")
publish("fkr_growth_rate", DOCS_ROOT / "fkr_growth_rate", "fkr_growth_rate.png")

write_harris_delta_prime_validation(RUN_ROOT / "harris_delta_prime")
publish("harris_delta_prime", DOCS_ROOT / "harris_delta_prime", "harris_delta_prime.png")

write_linear_tearing_dispersion_validation(RUN_ROOT / "linear_tearing_dispersion")
publish("linear_tearing_dispersion", DOCS_ROOT / "linear_tearing_dispersion",
        "linear_tearing_dispersion.png")

write_linear_tearing_eigenvalue_validation(RUN_ROOT / "linear_tearing_eigenvalue")
publish("linear_tearing_eigenvalue", DOCS_ROOT / "linear_tearing_eigenvalue",
        "linear_tearing_eigenvalue.png")

write_linear_tearing_layer_validation(RUN_ROOT / "linear_tearing_layer")
publish("linear_tearing_layer", DOCS_ROOT / "linear_tearing_layer", "linear_tearing_layer.png")

write_linear_tearing_timedomain_validation(RUN_ROOT / "linear_tearing_timedomain")
publish("linear_tearing_timedomain", DOCS_ROOT / "linear_tearing_timedomain",
        "linear_tearing_timedomain.png")
print("linear tearing validations done", flush=True)

write_linearized_rhs_validation(RUN_ROOT / "linearized_rhs")
publish("linearized_rhs", DOCS_ROOT / "linearized_rhs", "linearized_rhs_errors.png")

write_reduced_mhd_linear_eigenmode_validation(RUN_ROOT / "reduced_mhd_eigenmode")
publish("reduced_mhd_eigenmode", DOCS_ROOT / "reduced_mhd_eigenmode",
        "reduced_mhd_linear_eigenmode_errors.png")

write_cosine_equilibrium_linearization_validation(RUN_ROOT / "cosine_equilibrium_linearization")
publish("cosine_equilibrium_linearization", DOCS_ROOT / "cosine_equilibrium_linearization",
        "cosine_equilibrium_linearization_errors.png")

write_periodic_current_sheet_eigenvalue_validation(
    RUN_ROOT / "periodic_current_sheet_eigenvalue"
)
publish("periodic_current_sheet_eigenvalue", DOCS_ROOT / "periodic_current_sheet_eigenvalue",
        "periodic_current_sheet_spectrum.png")

write_periodic_current_sheet_timedomain_validation(
    RUN_ROOT / "periodic_current_sheet_timedomain"
)
publish("periodic_current_sheet_timedomain", DOCS_ROOT / "periodic_current_sheet_timedomain",
        "periodic_current_sheet_timedomain.png")

write_periodic_current_sheet_nonlinear_bridge_validation(
    RUN_ROOT / "periodic_current_sheet_nonlinear_bridge"
)
publish("periodic_current_sheet_nonlinear_bridge",
        DOCS_ROOT / "periodic_current_sheet_nonlinear_bridge",
        "periodic_current_sheet_nonlinear_bridge.png")
print("linearization and current-sheet validations done", flush=True)

write_periodic_double_harris_nonlinear_growth_validation(
    RUN_ROOT / "periodic_double_harris_nonlinear_growth"
)
publish("periodic_double_harris_nonlinear_growth",
        DOCS_ROOT / "periodic_double_harris_nonlinear_growth",
        "periodic_double_harris_nonlinear_growth.png")

media_preset = double_harris_seeded_long_run_presets()["readme_release_media"]
write_periodic_double_harris_seeded_long_run_validation(
    RUN_ROOT / "periodic_double_harris_seeded_long_run",
    shape=LONG_RUN_SHAPE,
    t_end=float(media_preset["t_end"]),
    save_every=int(media_preset["save_every"]),
    fit_window=tuple(media_preset["fit_window"]),
    duration_label=str(media_preset["duration_label"]),
    movies=True,
)
publish("periodic_double_harris_seeded_long_run",
        DOCS_ROOT / "periodic_double_harris_seeded_long_run" / "figures",
        "periodic_double_harris_seeded_long_run.png",
        "periodic_double_harris_flux.gif",
        "periodic_double_harris_current.gif")

write_periodic_double_harris_convergence_validation(
    RUN_ROOT / "periodic_double_harris_convergence",
    resolutions=CONVERGENCE_RESOLUTIONS,
    dt_values=CONVERGENCE_DT_VALUES,
    reference_resolution=CONVERGENCE_RESOLUTIONS[0],
    t_end=CONVERGENCE_T_END,
    fit_window=CONVERGENCE_FIT_WINDOW,
)
publish("periodic_double_harris_convergence",
        DOCS_ROOT / "periodic_double_harris_convergence",
        "periodic_double_harris_convergence.png")
print("double-Harris validations done", flush=True)

write_nonlinear_energy_budget_validation(RUN_ROOT / "nonlinear_energy_budget")
publish("nonlinear_energy_budget", DOCS_ROOT / "nonlinear_energy_budget",
        "nonlinear_energy_budget.png")

write_diffusion_eigenvalue_validation(RUN_ROOT / "diffusion_eigenvalue")
publish("diffusion_eigenvalue", DOCS_ROOT / "diffusion_eigenvalue",
        "diffusion_eigenvalue_errors.png")

write_power_iteration_validation(RUN_ROOT / "power_iteration")
publish("power_iteration", DOCS_ROOT / "power_iteration", "power_iteration_history.png")

write_arnoldi_validation(RUN_ROOT / "arnoldi")
publish("arnoldi", DOCS_ROOT / "arnoldi", "arnoldi_ritz_values.png")

# --- Results ---------------------------------------------------------------
print(f"Done in {time.perf_counter() - start:.1f} s; figures copied under {DOCS_ROOT}")
