from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mhx.cli.main import app
from mhx.neural_ode import (
    NEURAL_ODE_BASELINE_SCHEMA,
    NEURAL_ODE_CALIBRATION_SCHEMA,
    NEURAL_ODE_DATASET_SCHEMA,
    NEURAL_ODE_EXPERIMENT_SCHEMA,
    NEURAL_ODE_LATENT_METRICS_SCHEMA,
    NEURAL_ODE_LATENT_MODEL_SCHEMA,
    NEURAL_ODE_REPRODUCIBILITY_GATES_SCHEMA,
    NEURAL_ODE_SPLIT_SCHEMA,
    NEURAL_ODE_TRAINING_GATES_SCHEMA,
    build_seed_qi_trajectory_dataset,
    evaluate_baselines,
    fit_latent_ode,
    make_train_val_test_split,
    write_neural_ode_reproducibility_bundle,
    write_neural_ode_training_bundle,
)


def test_seed_qi_trajectory_dataset_is_deterministic() -> None:
    kwargs = {
        "shape": (8, 8),
        "seeds": (0, 1, 2, 3),
        "steps": 4,
        "dt": 1.0e-2,
        "psi_noise_amplitude": 1.0e-8,
    }
    dataset = build_seed_qi_trajectory_dataset(**kwargs)
    repeated = build_seed_qi_trajectory_dataset(**kwargs)

    assert dataset.diagnostics["schema"] == NEURAL_ODE_DATASET_SCHEMA
    assert dataset.features.shape == (4, 4, 7)
    assert dataset.targets.shape == (4, 4, 3)
    assert dataset.target_names == (
        "mode_amplitude",
        "total_energy",
        "magnetic_divergence_linf",
    )
    np.testing.assert_array_equal(dataset.seeds, repeated.seeds)
    np.testing.assert_allclose(dataset.features, repeated.features)
    np.testing.assert_allclose(dataset.targets, repeated.targets)
    assert np.isfinite(dataset.targets).all()
    assert dataset.diagnostics["source_validation_passed"] is True


def test_split_manifest_is_disjoint_complete_and_stable() -> None:
    split = make_train_val_test_split((10, 20, 30, 40, 50, 60), split_seed=99)
    repeated = make_train_val_test_split((10, 20, 30, 40, 50, 60), split_seed=99)

    assert split.diagnostics["schema"] == NEURAL_ODE_SPLIT_SCHEMA
    assert split.diagnostics["passed"] is True
    assert split == repeated
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert set(split.train) | set(split.validation) | set(split.test) == {
        10,
        20,
        30,
        40,
        50,
        60,
    }


def test_baselines_report_finite_scores_and_calibration() -> None:
    dataset = build_seed_qi_trajectory_dataset(
        shape=(8, 8),
        seeds=(0, 1, 2, 3),
        steps=4,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )
    split = make_train_val_test_split(dataset.seeds, split_seed=7)
    evaluation = evaluate_baselines(dataset, split, observation_count=2)

    assert evaluation.metrics["schema"] == NEURAL_ODE_BASELINE_SCHEMA
    assert evaluation.calibration["schema"] == NEURAL_ODE_CALIBRATION_SCHEMA
    assert evaluation.metrics["passed"] is True
    for baseline_name in ("persistence", "linear_prefix", "train_mean_time"):
        assert baseline_name in evaluation.predictions
        assert evaluation.predictions[baseline_name].shape == dataset.targets.shape
        assert evaluation.metrics["baselines"][baseline_name]["test"]["rmse"] >= 0.0
        assert 0.0 <= evaluation.calibration["baselines"][baseline_name]["coverage"]["test"][
            "coverage_2sigma_mean"
        ] <= 1.0


def test_latent_ode_fit_is_deterministic_and_schema_valid() -> None:
    dataset = build_seed_qi_trajectory_dataset(
        shape=(8, 8),
        seeds=(0, 1, 2, 3),
        steps=5,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
    )
    split = make_train_val_test_split(dataset.seeds, split_seed=7)
    baseline = evaluate_baselines(dataset, split, observation_count=2)
    fit = fit_latent_ode(
        dataset,
        split,
        baseline,
        observation_count=2,
        hidden_size=4,
        random_seed=11,
    )
    repeated = fit_latent_ode(
        dataset,
        split,
        baseline,
        observation_count=2,
        hidden_size=4,
        random_seed=11,
    )

    assert fit.model["schema"] == NEURAL_ODE_LATENT_MODEL_SCHEMA
    assert fit.metrics["schema"] == NEURAL_ODE_LATENT_METRICS_SCHEMA
    assert fit.validation["schema"] == NEURAL_ODE_TRAINING_GATES_SCHEMA
    assert fit.validation["passed"] is True
    assert fit.predictions.shape == dataset.targets.shape
    np.testing.assert_allclose(fit.predictions, repeated.predictions)
    assert fit.metrics["latent_ode_test_rmse"] >= 0.0
    assert np.isfinite(fit.metrics["test_rmse_ratio_to_best_baseline"])


def test_write_neural_ode_bundle_artifacts_and_npz_schema(tmp_path) -> None:
    manifest_path, validation = write_neural_ode_reproducibility_bundle(
        tmp_path,
        shape=(8, 8),
        seeds=(0, 1, 2, 3),
        steps=4,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
        write_figures=True,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["schema"] == NEURAL_ODE_REPRODUCIBILITY_GATES_SCHEMA
    assert validation["passed"] is True
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["claim_level"] == "validation"
    assert manifest["config"]["schema"] == NEURAL_ODE_EXPERIMENT_SCHEMA
    assert manifest["hashes"]["dataset"]
    assert manifest["hashes"]["validation"]
    assert (tmp_path / "figures" / "dataset_targets.png").stat().st_size > 0
    assert (tmp_path / "figures" / "baseline_rmse.png").stat().st_size > 0
    assert (tmp_path / "figures" / "calibration_coverage.png").stat().st_size > 0

    with np.load(tmp_path / "dataset.npz") as data:
        assert str(data["schema"]) == NEURAL_ODE_DATASET_SCHEMA
        assert data["features"].shape == (4, 4, 7)
        assert data["targets"].shape == (4, 4, 3)
        assert data["feature_names"].tolist() == [
            "mode_amplitude",
            "magnetic_energy",
            "kinetic_energy",
            "total_energy",
            "magnetic_divergence_linf",
            "psi_l2",
            "omega_l2",
        ]
    assert json.loads((tmp_path / "splits.json").read_text())["schema"] == NEURAL_ODE_SPLIT_SCHEMA
    assert json.loads((tmp_path / "baseline_metrics.json").read_text())[
        "schema"
    ] == NEURAL_ODE_BASELINE_SCHEMA
    assert json.loads((tmp_path / "calibration.json").read_text())[
        "schema"
    ] == NEURAL_ODE_CALIBRATION_SCHEMA
    assert json.loads((tmp_path / "validation.json").read_text())[
        "schema"
    ] == NEURAL_ODE_REPRODUCIBILITY_GATES_SCHEMA


def test_write_neural_ode_training_bundle_and_cli(tmp_path) -> None:
    manifest_path, validation = write_neural_ode_training_bundle(
        tmp_path / "fit",
        shape=(8, 8),
        seeds=(0, 1, 2, 3),
        steps=5,
        dt=1.0e-2,
        psi_noise_amplitude=1.0e-8,
        hidden_size=4,
        write_figures=True,
    )

    assert manifest_path == tmp_path / "fit" / "manifest.json"
    assert validation["schema"] == NEURAL_ODE_TRAINING_GATES_SCHEMA
    assert validation["passed"] is True
    manifest = json.loads(manifest_path.read_text())
    assert manifest["outputs"]["latent_ode_model"] == "latent_ode_model.json"
    assert manifest["outputs"]["latent_ode_predictions"] == "latent_ode_predictions.npz"
    assert (tmp_path / "fit" / "figures" / "latent_ode_predictions.png").stat().st_size > 0
    with np.load(tmp_path / "fit" / "latent_ode_predictions.npz") as data:
        assert str(data["schema"]) == NEURAL_ODE_LATENT_METRICS_SCHEMA
        assert data["predictions"].shape == data["targets"].shape

    outdir = tmp_path / "cli-fit"
    result = CliRunner().invoke(
        app,
        [
            "neural-ode",
            "train",
            "--outdir",
            str(outdir),
            "--seeds",
            "0,1,2,3",
            "--nx",
            "8",
            "--ny",
            "8",
            "--steps",
            "5",
            "--hidden-size",
            "4",
            "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (outdir / "latent_ode_model.json").exists()
    assert not (outdir / "figures").exists()


def test_neural_ode_cli_writes_bundle(tmp_path) -> None:
    outdir = tmp_path / "cli-neural-ode"
    result = CliRunner().invoke(
        app,
        [
            "neural-ode",
            "dataset",
            "--outdir",
            str(outdir),
            "--seeds",
            "0,1,2,3",
            "--nx",
            "8",
            "--ny",
            "8",
            "--steps",
            "4",
            "--no-figures",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (outdir / "manifest.json").exists()
    assert (outdir / "dataset.npz").exists()
    assert not (outdir / "figures").exists()


def test_neural_ode_reproducibility_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="steps"):
        build_seed_qi_trajectory_dataset(shape=(8, 8), seeds=(0, 1, 2), steps=2)
    with pytest.raises(ValueError, match="target"):
        build_seed_qi_trajectory_dataset(
            shape=(8, 8),
            seeds=(0, 1, 2),
            steps=4,
            target_names=("missing",),
        )
    with pytest.raises(ValueError, match="three samples"):
        make_train_val_test_split((0, 1))
    with pytest.raises(ValueError, match="unique"):
        make_train_val_test_split((0, 0, 1))
    with pytest.raises(ValueError, match="train_fraction"):
        make_train_val_test_split((0, 1, 2), train_fraction=0.0)
    with pytest.raises(ValueError, match="validation_fraction"):
        make_train_val_test_split((0, 1, 2), validation_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction \\+ validation_fraction"):
        make_train_val_test_split((0, 1, 2), train_fraction=0.8, validation_fraction=0.3)
    dataset = build_seed_qi_trajectory_dataset(
        shape=(8, 8),
        seeds=(0, 1, 2),
        steps=4,
    )
    split = make_train_val_test_split(dataset.seeds)
    with pytest.raises(ValueError, match="observation_count"):
        evaluate_baselines(dataset, split, observation_count=0)
    with pytest.raises(ValueError, match="leave at least one"):
        evaluate_baselines(dataset, split, observation_count=4)
    baseline = evaluate_baselines(dataset, split, observation_count=1)
    with pytest.raises(ValueError, match="hidden_size"):
        fit_latent_ode(dataset, split, baseline, hidden_size=0)
    with pytest.raises(ValueError, match="ridge"):
        fit_latent_ode(dataset, split, baseline, ridge=-1.0)
    with pytest.raises(ValueError, match="unknown sample id"):
        evaluate_baselines(
            dataset,
            make_train_val_test_split((0, 1, 99)),
            observation_count=1,
        )
    with pytest.raises(ValueError, match="shape"):
        build_seed_qi_trajectory_dataset(shape=(7, 8), seeds=(0, 1, 2), steps=4)
    prediction = evaluate_baselines(dataset, split, observation_count=1)
    assert prediction.metrics["passed"] is True
