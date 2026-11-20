"""Run the plasmoid, forced-turbulence, inverse-problem and validation-media examples."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _example_runner import run_example

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _nonempty(*paths: Path) -> None:
    for path in paths:
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


@pytest.fixture(scope="module")
def plasmoid_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        out = tmp_path_factory.mktemp("plasmoids") / "out"
        ns = run_example(
            EXAMPLES / "plasmoid_chain" / "plasmoids.py",
            out.parent,
            {
                "OUTDIR": out,
                "NX": 32,
                "NY": 64,
                "T_END": 0.04,
                "DT": 0.004,
                "SAVE_EVERY": 5,
                "SEED_TYPE": "noise",
                "CROP": (0.1, 0.6),
                "FIELD": "island",
                "SCALE": "fixed",
            },
            monkeypatch=mp,
        )
    finally:
        mp.undo()
    return out, ns


def test_plasmoids_outputs(plasmoid_run):
    out, ns = plasmoid_run
    _nonempty(
        out / "plasmoids.gif",
        out / "plasmoids_final.png",
        out / "plasmoids_diagnostics.png",
        out / "plasmoids_diagnostics.npz",
        out / "plasmoids_final_state.npz",
        out / "plasmoids_trajectory.npz",
    )
    assert ns["num_blocks"] == 2
    assert ns["eta4"] > 0.0
    assert len(ns["t_history"]) == 3
    assert np.isfinite(ns["psi_frames"][-1]).all()
    diag = np.load(out / "plasmoids_diagnostics.npz")
    assert diag["y_mode_amplitudes"].shape == (3, len(ns["TRACKED_Y_MODES"]))
    assert np.all(diag["magnetic_energy"] > 0.0)


def test_plasmoids_coherent_seed(tmp_path, monkeypatch):
    out = tmp_path / "out"
    ns = run_example(
        EXAMPLES / "plasmoid_chain" / "plasmoids.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 32,
            "NY": 64,
            "T_END": 0.02,
            "DT": 0.004,
            "SAVE_EVERY": 5,
            "ETA4": 1.0e-6,
            "FIELD": "jz",
        },
        monkeypatch=monkeypatch,
    )
    assert ns["eta4"] == 1.0e-6
    assert ns["coherent_amplitude"] == pytest.approx(0.01 * ns["WIDTH"])
    _nonempty(out / "plasmoids.gif", out / "plasmoids_trajectory.npz")


def test_forced_turbulence_outputs(tmp_path, monkeypatch):
    out = tmp_path / "out"
    ns = run_example(
        EXAMPLES / "plasmoid_chain" / "forced_turbulence.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 32,
            "NY": 64,
            "T_END": 0.1,
            "DT": 0.005,
            "SAVE_EVERY": 10,
            "GIF_FRAMES": 3,
        },
        monkeypatch=monkeypatch,
    )
    gif = out / "forced_turbulent_reconnection.gif"
    traj = out / "forced_turbulence_trajectory.npz"
    _nonempty(gif, traj)
    data = np.load(traj)
    assert data["psi"].shape == (3, 32, 64)
    assert data["current_density"].shape == data["psi"].shape
    assert np.isfinite(data["psi"]).all()
    assert ns["num_blocks"] == 2


def _fake_optax() -> types.ModuleType:
    """Minimal Adam with the optax interface, used when optax is not installed."""
    mod = types.ModuleType("optax")

    def adam(learning_rate, b1=0.9, b2=0.999, eps=1.0e-8):
        def init(params):
            return (jnp.zeros_like(params), jnp.zeros_like(params), 0)

        def update(grads, state):
            m, v, count = state
            count += 1
            m = b1 * m + (1.0 - b1) * grads
            v = b2 * v + (1.0 - b2) * grads**2
            m_hat = m / (1.0 - b1**count)
            v_hat = v / (1.0 - b2**count)
            return -learning_rate * m_hat / (jnp.sqrt(v_hat) + eps), (m, v, count)

        return types.SimpleNamespace(init=init, update=update)

    mod.adam = adam
    mod.apply_updates = lambda params, updates: params + updates
    return mod


def test_effective_resistivity_matching(tmp_path, monkeypatch):
    # optax is optional in the test environment: fall back to a minimal Adam stand-in so
    # the rest of the script (forward model, AD gradient, figures, outputs) is still run.
    try:
        import optax  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "optax", _fake_optax())
    out = tmp_path / "out"
    ns = run_example(
        EXAMPLES / "current_sheet" / "effective_resistivity_matching.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 16,
            "T_END": 0.1,
            "DT": 0.01,
            "N_OBS": 1,
            "N_FULL_PROBES": 3,
            "OPT_STEPS": 3,
        },
        monkeypatch=monkeypatch,
    )
    _nonempty(
        out / "loss_history.png",
        out / "probe_fits.png",
        out / "eta_fields.png",
        out / "evolution.gif",
        out / "effective_resistivity_matching.npz",
    )
    assert len(ns["loss_history"]) == 3
    assert np.isfinite(ns["loss_history"]).all()
    assert ns["probes_obs"].shape == (2, 9, 3)
    assert -1.0 <= ns["correlation"] <= 1.0
    data = np.load(out / "effective_resistivity_matching.npz")
    assert data["eta_optimized"].shape == (16, 16)
    # The optimization must not make the fit worse than the wiped start.
    assert ns["loss_history"][-1] <= float(ns["initial_loss"]) * 1.0001


def test_effective_resistivity_center_probes_and_target_npz(tmp_path, monkeypatch):
    try:
        import optax  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "optax", _fake_optax())
    reference = tmp_path / "reference.npz"
    np.savez(reference, probes=np.random.default_rng(0).normal(size=(2, 9, 3)))
    out = tmp_path / "out"
    ns = run_example(
        EXAMPLES / "current_sheet" / "effective_resistivity_matching.py",
        tmp_path,
        {
            "OUTDIR": out,
            "NX": 16,
            "NY": 16,
            "T_END": 0.1,
            "DT": 0.01,
            "N_OBS": 1,
            "PROBE_STYLE": "center",
            "PROBE_EXTENT": 1,
            "OPT_STEPS": 1,
            "TARGET_NPZ": str(reference),
        },
        monkeypatch=monkeypatch,
    )
    assert ns["probe_xy"].shape == (9, 2)
    _nonempty(out / "eta_fields.png")


def test_make_validation_media_copies_figures(tmp_path, monkeypatch):
    import mhx.benchmarks as benchmarks

    figure_names = [
        "decay_amplitude.png", "decay_energy.png", "decay_relative_error.png",
        "fkr_scaling.png", "plasmoid_scaling.png", "ideal_tearing_scaling.png",
        "fkr_constant_psi_window.png", "fkr_growth_rate.png", "harris_delta_prime.png",
        "linear_tearing_dispersion.png", "linear_tearing_eigenvalue.png",
        "linear_tearing_layer.png", "linear_tearing_timedomain.png",
        "linearized_rhs_errors.png", "reduced_mhd_linear_eigenmode_errors.png",
        "cosine_equilibrium_linearization_errors.png", "periodic_current_sheet_spectrum.png",
        "periodic_current_sheet_timedomain.png", "periodic_current_sheet_nonlinear_bridge.png",
        "periodic_double_harris_nonlinear_growth.png",
        "periodic_double_harris_seeded_long_run.png", "periodic_double_harris_flux.gif",
        "periodic_double_harris_current.gif", "periodic_double_harris_convergence.png",
        "nonlinear_energy_budget.png",
        "diffusion_eigenvalue_errors.png", "power_iteration_history.png",
        "arnoldi_ritz_values.png",
    ]
    calls: list[tuple[str, dict]] = []

    def make_fake(name):
        def fake(run_dir, **kwargs):
            calls.append((name, kwargs))
            figures = Path(run_dir) / "figures"
            figures.mkdir(parents=True, exist_ok=True)
            for fig in figure_names:
                (figures / fig).write_bytes(b"x")

        return fake

    for name in dir(benchmarks):
        if name.startswith("write_"):
            monkeypatch.setattr(benchmarks, name, make_fake(name))

    runs = tmp_path / "runs"
    docs = tmp_path / "docs"
    run_example(
        EXAMPLES / "make_validation_media.py",
        tmp_path,
        {"RUN_ROOT": runs, "DOCS_ROOT": docs},
        monkeypatch=monkeypatch,
    )
    _nonempty(
        docs / "exact_decay" / "decay_energy.png",
        docs / "arnoldi" / "arnoldi_ritz_values.png",
        docs / "periodic_double_harris_seeded_long_run" / "figures"
        / "periodic_double_harris_flux.gif",
        docs / "power_iteration" / "power_iteration_history.png",
    )
    names = [name for name, _ in calls]
    assert "write_periodic_double_harris_seeded_long_run_validation" in names
    assert len(names) == 22
    long_kwargs = dict(calls)["write_periodic_double_harris_seeded_long_run_validation"]
    assert long_kwargs["movies"] is True
    assert jax.config.jax_enable_x64
