from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from mhx.benchmarks.seed_robust_qi import (
    DEFAULT_QI_METRICS,
    SEED_ROBUST_QI_SCHEMA,
    QIMetricGate,
    compute_metric_statistics,
    generate_seed_ensemble,
    make_seeded_initial_state,
    run_seed_robust_qi,
    run_seed_robust_qi_validation,
    seeded_perturbation,
    seeded_tearing_initial_state,
    write_seed_robust_qi,
    write_seed_robust_qi_validation,
)
from mhx.config import MeshConfig
from mhx.grids import CartesianGrid
from mhx.state import ReducedMHDState


def test_generate_seed_ensemble_is_deterministic_and_unique() -> None:
    seeds = generate_seed_ensemble(12345, 6)
    repeated = generate_seed_ensemble(12345, 6)
    different = generate_seed_ensemble(12346, 6)

    assert seeds.dtype == np.uint32
    np.testing.assert_array_equal(seeds, repeated)
    assert len(set(int(seed) for seed in seeds)) == 6
    assert not np.array_equal(seeds, different)


def test_seeded_perturbation_has_controlled_rms_and_zero_mean() -> None:
    perturbation = seeded_perturbation((8, 8), 7, amplitude=2.0e-6)
    repeated = seeded_perturbation((8, 8), 7, amplitude=2.0e-6)

    np.testing.assert_allclose(perturbation, repeated)
    assert abs(float(np.mean(perturbation))) < 1.0e-20
    assert float(np.sqrt(np.mean(perturbation**2))) == pytest.approx(2.0e-6)


def test_seeded_grid_helpers_and_base_state_path() -> None:
    grid = CartesianGrid.from_mesh_config(MeshConfig(shape=(8, 8)))
    perturbation = seeded_perturbation(grid, 7, noise_amplitude=2.0e-6)
    state = seeded_tearing_initial_state(grid, seed=7, noise_amplitude=2.0e-6)
    ensemble = generate_seed_ensemble(grid, seeds=(7, 8), noise_amplitude=2.0e-6)
    base_state = ReducedMHDState(psi=jnp.zeros((8, 8)), omega=jnp.zeros((8, 8)))

    assert perturbation.shape == (8, 8)
    assert state.psi.shape == (8, 8)
    assert len(ensemble) == 2
    seeded_state = make_seeded_initial_state(
        base_state,
        seed=7,
        psi_noise_amplitude=2.0e-6,
        omega_noise_amplitude=1.0e-6,
    )
    assert seeded_state.psi.shape == (8, 8)
    assert float(jnp.std(seeded_state.omega)) > 0.0


def test_compute_metric_statistics_applies_cv_and_divergence_gates() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 0.0],
            [1.01, 2.002, 2.0e-13],
            [0.99, 1.998, 1.0e-13],
        ],
        dtype=np.float64,
    )
    summaries = compute_metric_statistics(
        values,
        ("gamma_fit", "final_total_energy", "final_magnetic_divergence_linf"),
        gates={
            "gamma_fit": QIMetricGate(max_abs_cv=2.0e-2),
            "final_total_energy": QIMetricGate(max_abs_cv=2.0e-3),
            "final_magnetic_divergence_linf": QIMetricGate(
                max_abs_mean=2.0e-13,
                max_abs_max=3.0e-13,
            ),
        },
    )

    assert summaries["gamma_fit"]["mean"] == pytest.approx(1.0)
    assert summaries["gamma_fit"]["std"] == pytest.approx(0.01)
    assert summaries["gamma_fit"]["cv"] == pytest.approx(0.01)
    assert summaries["gamma_fit"]["passed"] is True
    assert summaries["final_total_energy"]["cv"] == pytest.approx(0.001)
    assert summaries["final_total_energy"]["passed"] is True
    assert summaries["final_magnetic_divergence_linf"]["passed"] is True


def test_metric_statistics_legacy_and_error_paths() -> None:
    summary = compute_metric_statistics(np.asarray([1.0, 2.0, 3.0]))
    assert summary["mean"] == pytest.approx(2.0)
    failed = compute_metric_statistics(
        np.asarray([[1.0, 3.0], [2.0, 4.0]]),
        ("cv_metric", "max_metric"),
        gates={
            "cv_metric": QIMetricGate(statistic="cv", threshold=1.0e-3),
            "max_metric": QIMetricGate(statistic="max", threshold=3.5),
        },
    )
    assert failed["cv_metric"]["passed"] is False
    assert failed["max_metric"]["passed"] is False
    with pytest.raises(ValueError, match="2D"):
        compute_metric_statistics(np.asarray([1.0, 2.0]), ("bad",))
    with pytest.raises(ValueError, match="column count"):
        compute_metric_statistics(np.ones((2, 2)), ("one",))


def test_run_seed_robust_qi_validation_is_deterministic_and_physics_gated() -> None:
    result = run_seed_robust_qi_validation(
        shape=(8, 8),
        seed_count=4,
        base_seed=123,
        steps=8,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )
    repeated = run_seed_robust_qi_validation(
        shape=(8, 8),
        seed_count=4,
        base_seed=123,
        steps=8,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )

    assert result.validation["passed"] is True
    assert result.metric_names == DEFAULT_QI_METRICS
    np.testing.assert_array_equal(result.seeds, repeated.seeds)
    np.testing.assert_allclose(result.metric_values, repeated.metric_values)
    assert result.summaries["gamma_fit"]["cv"] < 5.0e-2
    assert result.summaries["final_total_energy"]["cv"] < 1.0e-3
    assert result.summaries["final_magnetic_divergence_linf"]["max"] <= 1.0e-10


def test_write_seed_robust_qi_validation_artifacts(tmp_path) -> None:
    manifest_path, validation = write_seed_robust_qi_validation(
        tmp_path,
        shape=(8, 8),
        seed_count=4,
        base_seed=123,
        steps=8,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    assert diagnostics["schema"] == SEED_ROBUST_QI_SCHEMA
    assert diagnostics["seed_count"] == 4
    assert (tmp_path / "validation.json").exists()
    assert (tmp_path / "ensemble.npz").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "figures" / "qi_summary.png").stat().st_size > 0
    with np.load(tmp_path / "ensemble.npz") as data:
        assert str(data["schema"]) == SEED_ROBUST_QI_SCHEMA
        assert data["metric_values"].shape == (4, len(DEFAULT_QI_METRICS))


def test_write_seed_robust_qi_low_level_artifacts(tmp_path) -> None:
    manifest_path, validation = write_seed_robust_qi(
        tmp_path,
        seeds=(0, 1),
        shape=(8, 8),
        t_end=0.03,
        dt=1.0e-2,
        max_gamma_normalized_std=1.0,
        max_energy_normalized_std=1.0,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert (tmp_path / "seed_ensemble.npz").exists()
    assert (tmp_path / "trajectory_seed_0.npz").exists()
    assert (tmp_path / "figures" / "seed_robust_qi_metrics.png").stat().st_size > 0
    assert (tmp_path / "figures" / "seed_energy_spread.png").stat().st_size > 0


def test_write_seed_robust_qi_validation_without_figures(tmp_path) -> None:
    manifest_path, validation = write_seed_robust_qi_validation(
        tmp_path,
        write_figures=False,
        shape=(8, 8),
        seed_count=2,
        base_seed=123,
        steps=4,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )

    manifest = json.loads(manifest_path.read_text())
    assert validation["passed"] is True
    assert "qi_summary" not in manifest["outputs"]


def test_seed_robust_qi_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least two seeds"):
        run_seed_robust_qi(seeds=(0,), shape=(8, 8))
    with pytest.raises(ValueError, match="unique"):
        run_seed_robust_qi(seeds=(0, 0), shape=(8, 8))
    with pytest.raises(ValueError, match="shape"):
        run_seed_robust_qi(seeds=(0, 1), shape=(7, 8))
    with pytest.raises(ValueError, match="t_end"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), t_end=0.0)
    with pytest.raises(ValueError, match="dt"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), dt=0.0)
    with pytest.raises(ValueError, match="save_every"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), save_every=0)
    with pytest.raises(ValueError, match="non-negative"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), resistivity=-1.0)
    with pytest.raises(ValueError, match="positive"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), perturbation_amplitude=0.0)
    with pytest.raises(ValueError, match="noise_amplitude"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), noise_amplitude=-1.0)
    with pytest.raises(ValueError, match="max_gamma"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), max_gamma_normalized_std=-1.0)
    with pytest.raises(ValueError, match="max_energy"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), max_energy_normalized_std=-1.0)
    with pytest.raises(ValueError, match="max_relative"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), max_relative_energy_growth=-1.0)
    with pytest.raises(ValueError, match="max_divergence"):
        run_seed_robust_qi(seeds=(0, 1), shape=(8, 8), max_divergence_linf=-1.0)
    with pytest.raises(ValueError, match="seed is required"):
        seeded_perturbation((8, 8))
    with pytest.raises(ValueError, match="amplitude"):
        seeded_perturbation((8, 8), 0, amplitude=-1.0)
    with pytest.raises(ValueError, match="shape"):
        seeded_perturbation((3, 8), 0, amplitude=1.0)
    with pytest.raises(ValueError, match="count"):
        generate_seed_ensemble(1)
    with pytest.raises(ValueError, match="count"):
        generate_seed_ensemble(1, 0)
    with pytest.raises(ValueError, match="base_seed"):
        generate_seed_ensemble(-1, 1)
