"""Deterministic neural-ODE dataset, baseline, and fitted latent-ODE artifacts.

This module freezes the data contract that neural-ODE experiments consume and
ships a deterministic random-feature latent ODE fit.  The fit is deliberately
small enough for CI, but it is a real train/validation/test experiment with
baselines, calibration, model parameters, predictions, figures, and hashes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mhx.benchmarks.seed_robust_qi import (
    DEFAULT_QI_METRICS,
    run_seed_robust_qi_validation,
)
from mhx.config import MeshConfig
from mhx.diagnostics import (
    magnetic_divergence_linf,
    trajectory_energies,
    trajectory_mode_amplitude,
)
from mhx.grids import CartesianGrid
from mhx.io import write_manifest
from mhx.state import ReducedMHDTrajectory

NEURAL_ODE_DATASET_SCHEMA = "mhx.neural_ode.dataset.v1"
NEURAL_ODE_SPLIT_SCHEMA = "mhx.neural_ode.splits.v1"
NEURAL_ODE_BASELINE_SCHEMA = "mhx.neural_ode.baselines.v1"
NEURAL_ODE_CALIBRATION_SCHEMA = "mhx.neural_ode.calibration.v1"
NEURAL_ODE_EXPERIMENT_SCHEMA = "mhx.neural_ode.experiment_spec.v1"
NEURAL_ODE_REPRODUCIBILITY_GATES_SCHEMA = "mhx.neural_ode.reproducibility.gates.v1"
NEURAL_ODE_LATENT_MODEL_SCHEMA = "mhx.neural_ode.latent_model.v1"
NEURAL_ODE_LATENT_METRICS_SCHEMA = "mhx.neural_ode.latent_metrics.v1"
NEURAL_ODE_FAILURE_MODES_SCHEMA = "mhx.neural_ode.failure_modes.v1"
NEURAL_ODE_TRAINING_GATES_SCHEMA = "mhx.neural_ode.training.gates.v1"

DEFAULT_FEATURE_NAMES = (
    "mode_amplitude",
    "magnetic_energy",
    "kinetic_energy",
    "total_energy",
    "magnetic_divergence_linf",
    "psi_l2",
    "omega_l2",
)
DEFAULT_TARGET_NAMES = (
    "mode_amplitude",
    "total_energy",
    "magnetic_divergence_linf",
)


@dataclass(frozen=True)
class NeuralODEDataset:
    """Array bundle for deterministic reduced-MHD neural-ODE experiments."""

    seeds: np.ndarray
    times: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    diagnostics: dict[str, Any]
    source_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SplitManifest:
    """Deterministic train/validation/test sample split."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BaselineEvaluation:
    """Baseline metrics, calibration summaries, and prediction arrays."""

    metrics: dict[str, Any]
    calibration: dict[str, Any]
    predictions: dict[str, np.ndarray]


@dataclass(frozen=True)
class LatentODEFit:
    """Deterministic random-feature latent-ODE fit and forecast metrics."""

    model: dict[str, Any]
    metrics: dict[str, Any]
    predictions: np.ndarray
    validation: dict[str, Any]


def build_seed_qi_trajectory_dataset(
    *,
    shape: tuple[int, int] = (16, 16),
    seeds: Sequence[int] | None = None,
    seed_count: int = 6,
    base_seed: int = 20240511,
    steps: int = 24,
    dt: float = 1.0e-2,
    save_every: int = 1,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    psi_noise_amplitude: float = 1.0e-8,
    mode: tuple[int, int] = (1, 1),
    target_names: Sequence[str] = DEFAULT_TARGET_NAMES,
) -> NeuralODEDataset:
    """Build a deterministic seed-QI trajectory dataset for neural-ODE studies.

    The dataset shape is ``(n_seed, n_time, n_feature)``.  Targets are selected
    by name from the feature tensor so baseline and neural-ODE experiments use
    exactly the same time grid and target convention.
    """
    if steps < 3:
        raise ValueError("steps must be >= 3 so prefix baselines have future targets")
    if seed_count < 3 and seeds is None:
        raise ValueError("seed_count must be >= 3 for train/validation/test splits")
    normalized_shape = _normalize_shape(shape)
    source_result = run_seed_robust_qi_validation(
        shape=normalized_shape,
        seed_count=seed_count,
        base_seed=base_seed,
        seeds=seeds,
        perturbation_amplitude=perturbation_amplitude,
        psi_noise_amplitude=psi_noise_amplitude,
        resistivity=resistivity,
        viscosity=viscosity,
        dt=dt,
        steps=steps,
        save_every=save_every,
        mode=mode,
        metric_names=DEFAULT_QI_METRICS,
    )
    if len(source_result.seeds) < 3:
        raise ValueError("at least three unique seeds are required")
    grid = CartesianGrid.from_mesh_config(
        MeshConfig(shape=normalized_shape, lower=(0.0, 0.0), upper=(2.0 * np.pi, 2.0 * np.pi))
    )
    feature_rows = [
        _trajectory_feature_matrix(
            trajectory,
            lengths=grid.lengths,
            mode=mode,
        )
        for trajectory in source_result.trajectories
    ]
    features = np.stack(feature_rows).astype(np.float64)
    times = np.asarray(source_result.trajectories[0].times, dtype=np.float64)
    feature_names = DEFAULT_FEATURE_NAMES
    normalized_targets = tuple(str(name) for name in target_names)
    feature_indices = {name: index for index, name in enumerate(feature_names)}
    missing = sorted(set(normalized_targets) - set(feature_indices))
    if missing:
        raise ValueError(f"unknown target names: {', '.join(missing)}")
    target_indices = [feature_indices[name] for name in normalized_targets]
    targets = features[:, :, target_indices]
    diagnostics = {
        "schema": NEURAL_ODE_DATASET_SCHEMA,
        "source": "seed_robust_qi_validation",
        "claim_level": "validation",
        "claim_boundary": (
            "Deterministic FAST dataset for neural-ODE experiment wiring and "
            "baseline comparisons; not a trained surrogate or production UQ result."
        ),
        "shape": list(normalized_shape),
        "seeds": [int(seed) for seed in source_result.seeds],
        "seed_count": int(len(source_result.seeds)),
        "base_seed": int(base_seed),
        "steps": int(steps),
        "dt": float(dt),
        "save_every": int(save_every),
        "resistivity": float(resistivity),
        "viscosity": float(viscosity),
        "perturbation_amplitude": float(perturbation_amplitude),
        "psi_noise_amplitude": float(psi_noise_amplitude),
        "mode": [int(mode[0]), int(mode[1])],
        "feature_names": list(feature_names),
        "target_names": list(normalized_targets),
        "array_shapes": {
            "seeds": list(np.asarray(source_result.seeds).shape),
            "times": list(times.shape),
            "features": list(features.shape),
            "targets": list(targets.shape),
        },
        "source_validation_passed": bool(source_result.validation["passed"]),
    }
    return NeuralODEDataset(
        seeds=np.asarray(source_result.seeds, dtype=np.int64),
        times=times,
        features=features,
        targets=targets,
        feature_names=feature_names,
        target_names=normalized_targets,
        diagnostics=diagnostics,
        source_diagnostics=source_result.diagnostics,
    )


def make_train_val_test_split(
    sample_ids: Sequence[int],
    *,
    split_seed: int = 0,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
) -> SplitManifest:
    """Return deterministic disjoint train/validation/test split indices."""
    ids = tuple(int(item) for item in sample_ids)
    if len(ids) < 3:
        raise ValueError("at least three samples are required for train/validation/test splits")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    permutation = np.random.default_rng(int(split_seed)).permutation(np.asarray(ids))
    train_count = max(1, int(round(train_fraction * len(ids))))
    validation_count = max(1, int(round(validation_fraction * len(ids))))
    if train_count + validation_count >= len(ids):
        overflow = train_count + validation_count - len(ids) + 1
        train_count = max(1, train_count - overflow)
    test_count = len(ids) - train_count - validation_count
    if test_count < 1:
        raise ValueError("split fractions left no test samples")
    train = tuple(int(item) for item in sorted(permutation[:train_count]))
    validation = tuple(
        int(item) for item in sorted(permutation[train_count : train_count + validation_count])
    )
    test = tuple(int(item) for item in sorted(permutation[train_count + validation_count :]))
    diagnostics = {
        "schema": NEURAL_ODE_SPLIT_SCHEMA,
        "split_seed": int(split_seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_count / len(ids)),
        "n_samples": int(len(ids)),
        "train": list(train),
        "validation": list(validation),
        "test": list(test),
        "checks": {
            "disjoint": bool(
                set(train).isdisjoint(validation)
                and set(train).isdisjoint(test)
                and set(validation).isdisjoint(test)
            ),
            "complete": bool(set(train) | set(validation) | set(test) == set(ids)),
            "nonempty_train_validation_test": bool(train and validation and test),
        },
    }
    diagnostics["passed"] = all(diagnostics["checks"].values())
    return SplitManifest(train=train, validation=validation, test=test, diagnostics=diagnostics)


def evaluate_baselines(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    *,
    observation_count: int = 2,
) -> BaselineEvaluation:
    """Evaluate deterministic no-training baselines and calibration coverage."""
    if observation_count < 1:
        raise ValueError("observation_count must be >= 1")
    if observation_count >= dataset.times.size:
        raise ValueError("observation_count must leave at least one forecast target")
    targets = np.asarray(dataset.targets, dtype=np.float64)
    train_indices = _indices_for_ids(dataset.seeds, split.train)
    validation_indices = _indices_for_ids(dataset.seeds, split.validation)
    test_indices = _indices_for_ids(dataset.seeds, split.test)
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    predictions = {
        "persistence": _persistence_prediction(targets, observation_count=observation_count),
        "linear_prefix": _linear_prefix_prediction(
            dataset.times,
            targets,
            observation_count=observation_count,
        ),
        "train_mean_time": _train_mean_time_prediction(targets, train_indices),
    }
    metrics: dict[str, Any] = {
        "schema": NEURAL_ODE_BASELINE_SCHEMA,
        "observation_count": int(observation_count),
        "forecast_time_start": float(dataset.times[observation_count]),
        "target_names": list(dataset.target_names),
        "splits": split.diagnostics,
        "baselines": {},
    }
    calibration: dict[str, Any] = {
        "schema": NEURAL_ODE_CALIBRATION_SCHEMA,
        "observation_count": int(observation_count),
        "target_names": list(dataset.target_names),
        "baselines": {},
    }
    forecast_slice = slice(observation_count, None)
    for baseline_name, prediction in predictions.items():
        train_residual = targets[train_indices, forecast_slice, :] - prediction[
            train_indices, forecast_slice, :
        ]
        sigma = _target_sigma(train_residual)
        baseline_metrics = {}
        baseline_calibration = {
            "sigma_by_target": {
                name: float(value) for name, value in zip(dataset.target_names, sigma, strict=True)
            },
            "coverage": {},
        }
        for split_name, indices in split_indices.items():
            residual = targets[indices, forecast_slice, :] - prediction[indices, forecast_slice, :]
            baseline_metrics[split_name] = _forecast_scores(residual, dataset.target_names)
            baseline_calibration["coverage"][split_name] = _calibration_scores(
                residual,
                sigma,
                dataset.target_names,
            )
        metrics["baselines"][baseline_name] = baseline_metrics
        calibration["baselines"][baseline_name] = baseline_calibration
    metrics["checks"] = {
        "finite_targets": bool(np.isfinite(targets).all()),
        "finite_predictions": bool(
            all(np.isfinite(prediction).all() for prediction in predictions.values())
        ),
        "split_manifest_passed": bool(split.diagnostics["passed"]),
    }
    metrics["passed"] = all(metrics["checks"].values())
    calibration["passed"] = bool(metrics["passed"])
    return BaselineEvaluation(metrics=metrics, calibration=calibration, predictions=predictions)


def fit_latent_ode(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
    *,
    observation_count: int = 2,
    hidden_size: int = 8,
    ridge: float = 1.0e-8,
    random_seed: int = 0,
) -> LatentODEFit:
    """Fit a deterministic random-feature neural ODE to train trajectories.

    The model uses the autonomous form ``dz/dt = W phi(z)`` where ``z`` is the
    target vector and ``phi`` concatenates ``z``, random tanh features, and a
    constant bias.  ``W`` is solved by ridge regression from train-set finite
    differences, then integrated forward with explicit Euler on the saved time
    grid.
    """
    if observation_count < 1:
        raise ValueError("observation_count must be >= 1")
    if observation_count >= dataset.times.size:
        raise ValueError("observation_count must leave forecast targets")
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    targets = np.asarray(dataset.targets, dtype=np.float64)
    train_indices = _indices_for_ids(dataset.seeds, split.train)
    target_dim = targets.shape[-1]
    rng = np.random.default_rng(int(random_seed))
    feature_weights = rng.normal(
        scale=1.0 / np.sqrt(max(1, target_dim)),
        size=(target_dim, hidden_size),
    )
    feature_bias = rng.normal(scale=0.1, size=(hidden_size,))
    train_states = targets[train_indices, :-1, :].reshape((-1, target_dim))
    dt = np.diff(dataset.times)
    derivatives = (
        np.diff(targets[train_indices], axis=1)
        / dt.reshape((1, -1, 1))
    ).reshape((-1, target_dim))
    design = _latent_features(train_states, feature_weights, feature_bias)
    lhs = design.T @ design + ridge * np.eye(design.shape[1], dtype=np.float64)
    rhs = design.T @ derivatives
    coefficients = np.linalg.solve(lhs, rhs)
    predictions = _integrate_latent_ode(
        targets,
        dataset.times,
        coefficients,
        feature_weights,
        feature_bias,
        observation_count=observation_count,
    )
    metrics = _latent_ode_metrics(
        dataset,
        split,
        baseline,
        predictions,
        observation_count=observation_count,
    )
    model = {
        "schema": NEURAL_ODE_LATENT_MODEL_SCHEMA,
        "model_family": "ridge_random_feature_autonomous_latent_ode",
        "target_names": list(dataset.target_names),
        "observation_count": int(observation_count),
        "hidden_size": int(hidden_size),
        "ridge": float(ridge),
        "random_seed": int(random_seed),
        "feature_weights": feature_weights.tolist(),
        "feature_bias": feature_bias.tolist(),
        "coefficients": coefficients.tolist(),
        "equation": "dz_dt = coefficients.T @ [z, tanh(z @ feature_weights + feature_bias), 1]",
    }
    checks = {
        "finite_coefficients": bool(np.isfinite(coefficients).all()),
        "finite_predictions": bool(np.isfinite(predictions).all()),
        "finite_metrics": bool(metrics["checks"]["finite_metrics"]),
        "split_manifest_passed": bool(split.diagnostics["passed"]),
        "has_test_forecast": bool(metrics["splits"]["test"]["sample_count"] >= 1),
    }
    validation = {
        "schema": NEURAL_ODE_TRAINING_GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "model_schema": NEURAL_ODE_LATENT_MODEL_SCHEMA,
            "metrics_schema": NEURAL_ODE_LATENT_METRICS_SCHEMA,
            "claim_level": "validation",
            "claim_boundary": (
                "Deterministic fitted latent-ODE experiment on FAST seed-QI "
                "trajectories; not a production surrogate or extrapolation claim."
            ),
        },
    }
    return LatentODEFit(
        model=model,
        metrics=metrics,
        predictions=predictions,
        validation=validation,
    )


def evaluate_latent_ode_failure_modes(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
    latent: LatentODEFit,
    *,
    observation_count: int = 2,
) -> dict[str, Any]:
    """Return deterministic failure-mode diagnostics for the fitted latent ODE."""
    if observation_count < 1:
        raise ValueError("observation_count must be >= 1")
    if observation_count >= dataset.times.size:
        raise ValueError("observation_count must leave forecast targets")
    forecast_indices = np.arange(observation_count, dataset.times.size)
    midpoint = max(1, forecast_indices.size // 2)
    early_indices = forecast_indices[:midpoint]
    late_indices = (
        forecast_indices[midpoint:]
        if midpoint < forecast_indices.size
        else forecast_indices
    )
    split_indices = {
        "train": _indices_for_ids(dataset.seeds, split.train),
        "validation": _indices_for_ids(dataset.seeds, split.validation),
        "test": _indices_for_ids(dataset.seeds, split.test),
    }
    residual = np.asarray(dataset.targets - latent.predictions, dtype=np.float64)
    train_rmse = _rmse_for_indices(residual, split_indices["train"], forecast_indices)
    test_rmse = _rmse_for_indices(residual, split_indices["test"], forecast_indices)
    early_test_rmse = _rmse_for_indices(residual, split_indices["test"], early_indices)
    late_test_rmse = _rmse_for_indices(residual, split_indices["test"], late_indices)
    best_baseline_test_rmse = min(
        float(metrics["test"]["rmse"])
        for metrics in baseline.metrics["baselines"].values()
    )
    ratios = {
        "test_to_train_rmse": test_rmse / max(train_rmse, np.finfo(np.float64).eps),
        "late_to_early_test_rmse": late_test_rmse
        / max(early_test_rmse, np.finfo(np.float64).eps),
        "latent_to_best_baseline_test_rmse": test_rmse
        / max(best_baseline_test_rmse, np.finfo(np.float64).eps),
    }
    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "early_test_rmse": early_test_rmse,
        "late_test_rmse": late_test_rmse,
        "best_baseline_test_rmse": best_baseline_test_rmse,
        "ratios": {key: float(value) for key, value in ratios.items()},
    }
    warnings = [
        name
        for name, value in ratios.items()
        if value > 2.0
    ]
    checks = {
        "finite_metrics": bool(
            np.isfinite(
                [
                    train_rmse,
                    test_rmse,
                    early_test_rmse,
                    late_test_rmse,
                    best_baseline_test_rmse,
                    *ratios.values(),
                ]
            ).all()
        ),
        "has_test_samples": bool(split_indices["test"].size >= 1),
        "has_long_horizon_window": bool(late_indices.size >= 1),
        "split_manifest_passed": bool(split.diagnostics["passed"]),
    }
    return {
        "schema": NEURAL_ODE_FAILURE_MODES_SCHEMA,
        "claim_level": "validation",
        "claim_boundary": (
            "FAST failure-mode diagnostics for seed extrapolation and "
            "long-horizon drift; warnings are reported rather than hidden."
        ),
        "observation_count": int(observation_count),
        "target_names": list(dataset.target_names),
        "forecast_time_start": float(dataset.times[observation_count]),
        "early_window": [
            float(dataset.times[int(early_indices[0])]),
            float(dataset.times[int(early_indices[-1])]),
        ],
        "late_window": [
            float(dataset.times[int(late_indices[0])]),
            float(dataset.times[int(late_indices[-1])]),
        ],
        "metrics": metrics,
        "warnings": warnings,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_neural_ode_training_bundle(
    outdir: str | Path,
    *,
    shape: tuple[int, int] = (16, 16),
    seeds: Sequence[int] | None = None,
    seed_count: int = 6,
    base_seed: int = 20240511,
    split_seed: int = 314159,
    steps: int = 24,
    dt: float = 1.0e-2,
    save_every: int = 1,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    psi_noise_amplitude: float = 1.0e-8,
    observation_count: int = 2,
    hidden_size: int = 8,
    ridge: float = 1.0e-8,
    model_seed: int = 0,
    write_figures: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Write dataset, baselines, fitted latent ODE, predictions, and figures."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_seed_qi_trajectory_dataset(
        shape=shape,
        seeds=seeds,
        seed_count=seed_count,
        base_seed=base_seed,
        steps=steps,
        dt=dt,
        save_every=save_every,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        psi_noise_amplitude=psi_noise_amplitude,
    )
    split = make_train_val_test_split(dataset.seeds, split_seed=split_seed)
    baseline = evaluate_baselines(dataset, split, observation_count=observation_count)
    latent = fit_latent_ode(
        dataset,
        split,
        baseline,
        observation_count=observation_count,
        hidden_size=hidden_size,
        ridge=ridge,
        random_seed=model_seed,
    )
    failure_modes = evaluate_latent_ode_failure_modes(
        dataset,
        split,
        baseline,
        latent,
        observation_count=observation_count,
    )
    experiment = _experiment_spec(dataset, split, baseline)
    experiment["protocol"]["trainable_latent_ode"] = {
        "model_schema": NEURAL_ODE_LATENT_MODEL_SCHEMA,
        "metrics_schema": NEURAL_ODE_LATENT_METRICS_SCHEMA,
        "hidden_size": int(hidden_size),
        "ridge": float(ridge),
        "model_seed": int(model_seed),
    }
    validation = _training_validation_report(
        dataset,
        split,
        baseline,
        latent,
        failure_modes,
        experiment,
    )

    dataset_path = output_dir / "dataset.npz"
    splits_path = output_dir / "splits.json"
    baseline_metrics_path = output_dir / "baseline_metrics.json"
    calibration_path = output_dir / "calibration.json"
    model_path = output_dir / "latent_ode_model.json"
    latent_metrics_path = output_dir / "latent_ode_metrics.json"
    predictions_path = output_dir / "latent_ode_predictions.npz"
    failure_modes_path = output_dir / "failure_modes.json"
    experiment_path = output_dir / "experiment_spec.json"
    validation_path = output_dir / "validation.json"
    manifest_path = output_dir / "manifest.json"

    _write_dataset_npz(dataset_path, dataset)
    splits_path.write_text(
        json.dumps(split.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    baseline_metrics_path.write_text(
        json.dumps(baseline.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    calibration_path.write_text(
        json.dumps(baseline.calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model_path.write_text(json.dumps(latent.model, indent=2, sort_keys=True), encoding="utf-8")
    latent_metrics_path.write_text(
        json.dumps(latent.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failure_modes_path.write_text(
        json.dumps(failure_modes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        predictions_path,
        schema=np.asarray(NEURAL_ODE_LATENT_METRICS_SCHEMA),
        predictions=latent.predictions,
        targets=dataset.targets,
        times=dataset.times,
        seeds=dataset.seeds,
        target_names=np.asarray(dataset.target_names),
    )
    experiment_path.write_text(json.dumps(experiment, indent=2, sort_keys=True), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    outputs = {
        "dataset": dataset_path.name,
        "splits": splits_path.name,
        "baseline_metrics": baseline_metrics_path.name,
        "calibration": calibration_path.name,
        "latent_ode_model": model_path.name,
        "latent_ode_metrics": latent_metrics_path.name,
        "latent_ode_predictions": predictions_path.name,
        "failure_modes": failure_modes_path.name,
        "experiment_spec": experiment_path.name,
        "validation": validation_path.name,
    }
    if write_figures:
        figure_paths = _write_neural_ode_training_figures(
            dataset,
            baseline,
            latent,
            failure_modes,
            output_dir / "figures",
        )
        outputs.update(
            {
                name: path.relative_to(output_dir).as_posix()
                for name, path in figure_paths.items()
            }
        )
    write_manifest(
        manifest_path,
        config=experiment,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Deterministic fitted latent-ODE experiment with frozen splits, "
            "baselines, metrics, calibration, and predictions."
        ),
    )
    return manifest_path, validation


def write_neural_ode_reproducibility_bundle(
    outdir: str | Path,
    *,
    shape: tuple[int, int] = (16, 16),
    seeds: Sequence[int] | None = None,
    seed_count: int = 6,
    base_seed: int = 20240511,
    split_seed: int = 314159,
    steps: int = 24,
    dt: float = 1.0e-2,
    save_every: int = 1,
    resistivity: float = 1.0e-3,
    viscosity: float = 1.0e-3,
    perturbation_amplitude: float = 1.0e-3,
    psi_noise_amplitude: float = 1.0e-8,
    observation_count: int = 2,
    write_figures: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Write a complete deterministic neural-ODE reproducibility bundle."""
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_seed_qi_trajectory_dataset(
        shape=shape,
        seeds=seeds,
        seed_count=seed_count,
        base_seed=base_seed,
        steps=steps,
        dt=dt,
        save_every=save_every,
        resistivity=resistivity,
        viscosity=viscosity,
        perturbation_amplitude=perturbation_amplitude,
        psi_noise_amplitude=psi_noise_amplitude,
    )
    split = make_train_val_test_split(dataset.seeds, split_seed=split_seed)
    baseline = evaluate_baselines(dataset, split, observation_count=observation_count)
    experiment = _experiment_spec(dataset, split, baseline)
    validation = _validation_report(dataset, split, baseline, experiment)

    dataset_path = output_dir / "dataset.npz"
    splits_path = output_dir / "splits.json"
    metrics_path = output_dir / "baseline_metrics.json"
    calibration_path = output_dir / "calibration.json"
    experiment_path = output_dir / "experiment_spec.json"
    validation_path = output_dir / "validation.json"
    manifest_path = output_dir / "manifest.json"
    _write_dataset_npz(dataset_path, dataset)
    splits_path.write_text(
        json.dumps(split.diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(baseline.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    calibration_path.write_text(
        json.dumps(baseline.calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    experiment_path.write_text(
        json.dumps(experiment, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    outputs = {
        "dataset": dataset_path.name,
        "splits": splits_path.name,
        "baseline_metrics": metrics_path.name,
        "calibration": calibration_path.name,
        "experiment_spec": experiment_path.name,
        "validation": validation_path.name,
    }
    if write_figures:
        figure_paths = _write_neural_ode_figures(dataset, baseline, output_dir / "figures")
        outputs.update(
            {
                name: path.relative_to(output_dir).as_posix()
                for name, path in figure_paths.items()
            }
        )
    write_manifest(
        manifest_path,
        config=experiment,
        outputs=outputs,
        claim_level="validation",
        claim_scope=(
            "Deterministic neural-ODE reproducibility bundle with dataset splits, "
            "cheap baselines, and calibration summaries; no expensive neural training."
        ),
    )
    return manifest_path, validation


def _write_dataset_npz(path: Path, dataset: NeuralODEDataset) -> Path:
    np.savez_compressed(
        path,
        schema=np.asarray(NEURAL_ODE_DATASET_SCHEMA),
        seeds=dataset.seeds,
        times=dataset.times,
        features=dataset.features,
        targets=dataset.targets,
        feature_names=np.asarray(dataset.feature_names),
        target_names=np.asarray(dataset.target_names),
        diagnostics_json=np.asarray(json.dumps(dataset.diagnostics, sort_keys=True)),
        source_diagnostics_json=np.asarray(json.dumps(dataset.source_diagnostics, sort_keys=True)),
    )
    return path


def _latent_features(
    states: np.ndarray,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
) -> np.ndarray:
    nonlinear = np.tanh(states @ feature_weights + feature_bias.reshape((1, -1)))
    bias = np.ones((states.shape[0], 1), dtype=np.float64)
    return np.concatenate((states, nonlinear, bias), axis=1)


def _integrate_latent_ode(
    targets: np.ndarray,
    times: np.ndarray,
    coefficients: np.ndarray,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    *,
    observation_count: int,
) -> np.ndarray:
    predictions = np.array(targets, dtype=np.float64, copy=True)
    start_index = observation_count - 1
    predictions[:, start_index, :] = targets[:, start_index, :]
    for time_index in range(start_index, times.size - 1):
        dt = float(times[time_index + 1] - times[time_index])
        states = predictions[:, time_index, :]
        derivative = _latent_features(states, feature_weights, feature_bias) @ coefficients
        predictions[:, time_index + 1, :] = states + dt * derivative
    return predictions


def _latent_ode_metrics(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
    predictions: np.ndarray,
    *,
    observation_count: int,
) -> dict[str, Any]:
    forecast_slice = slice(observation_count, None)
    split_indices = {
        "train": _indices_for_ids(dataset.seeds, split.train),
        "validation": _indices_for_ids(dataset.seeds, split.validation),
        "test": _indices_for_ids(dataset.seeds, split.test),
    }
    split_scores = {}
    for split_name, indices in split_indices.items():
        residual = (
            dataset.targets[indices, forecast_slice, :]
            - predictions[indices, forecast_slice, :]
        )
        scores = _forecast_scores(residual, dataset.target_names)
        scores["sample_count"] = int(indices.size)
        split_scores[split_name] = scores
    best_baseline_test_rmse = min(
        float(metrics["test"]["rmse"])
        for metrics in baseline.metrics["baselines"].values()
    )
    latent_test_rmse = float(split_scores["test"]["rmse"])
    checks = {
        "finite_metrics": bool(
            np.isfinite(
                [
                    split_scores["train"]["rmse"],
                    split_scores["validation"]["rmse"],
                    latent_test_rmse,
                    best_baseline_test_rmse,
                ]
            ).all()
        ),
        "nonnegative_errors": bool(latent_test_rmse >= 0.0 and best_baseline_test_rmse >= 0.0),
    }
    return {
        "schema": NEURAL_ODE_LATENT_METRICS_SCHEMA,
        "observation_count": int(observation_count),
        "target_names": list(dataset.target_names),
        "splits": split_scores,
        "best_baseline_test_rmse": best_baseline_test_rmse,
        "latent_ode_test_rmse": latent_test_rmse,
        "test_rmse_ratio_to_best_baseline": latent_test_rmse
        / max(best_baseline_test_rmse, np.finfo(np.float64).eps),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _training_validation_report(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
    latent: LatentODEFit,
    failure_modes: Mapping[str, Any],
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_seed_qi_validation_passed": bool(dataset.diagnostics["source_validation_passed"]),
        "split_manifest_passed": bool(split.diagnostics["passed"]),
        "baseline_metrics_passed": bool(baseline.metrics["passed"]),
        "latent_training_passed": bool(latent.validation["passed"]),
        "latent_metrics_passed": bool(latent.metrics["passed"]),
        "predictions_match_target_shape": bool(latent.predictions.shape == dataset.targets.shape),
        "failure_modes_reported": bool(
            failure_modes.get("schema") == NEURAL_ODE_FAILURE_MODES_SCHEMA
            and failure_modes.get("passed")
        ),
    }
    return {
        "schema": NEURAL_ODE_TRAINING_GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            **experiment,
            "latent_ode": {
                "model_schema": NEURAL_ODE_LATENT_MODEL_SCHEMA,
                "metrics_schema": NEURAL_ODE_LATENT_METRICS_SCHEMA,
                "test_rmse_ratio_to_best_baseline": latent.metrics[
                    "test_rmse_ratio_to_best_baseline"
                ],
                "failure_mode_schema": NEURAL_ODE_FAILURE_MODES_SCHEMA,
                "failure_warnings": list(failure_modes.get("warnings", [])),
            },
        },
    }


def _write_neural_ode_training_figures(
    dataset: NeuralODEDataset,
    baseline: BaselineEvaluation,
    latent: LatentODEFit,
    failure_modes: Mapping[str, Any],
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = figure_dir / "latent_ode_predictions.png"
    rmse_path = figure_dir / "latent_ode_rmse_comparison.png"
    failure_path = figure_dir / "latent_ode_failure_modes.png"
    target_index = 0
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for seed_index, seed in enumerate(dataset.seeds):
        ax.plot(
            dataset.times,
            dataset.targets[seed_index, :, target_index],
            color="0.75",
            linewidth=1.0,
        )
        ax.plot(
            dataset.times,
            latent.predictions[seed_index, :, target_index],
            linestyle="--",
            linewidth=1.2,
            label=f"seed {int(seed)}",
        )
    ax.set_title(f"Latent ODE forecasts: {dataset.target_names[target_index]}")
    ax.set_xlabel("time")
    ax.set_ylabel(dataset.target_names[target_index])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="x-small", ncols=2)
    fig.savefig(prediction_path, dpi=180)
    plt.close(fig)

    baseline_names = list(baseline.metrics["baselines"])
    rmse_values = [
        baseline.metrics["baselines"][name]["test"]["rmse"]
        for name in baseline_names
    ]
    labels = [*baseline_names, "latent_ode"]
    values = [*rmse_values, latent.metrics["latent_ode_test_rmse"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.bar(labels, values, color=["#9ecae1"] * len(baseline_names) + ["#31a354"])
    ax.set_ylabel("test RMSE")
    ax.set_title("Fitted latent ODE vs deterministic baselines")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(rmse_path, dpi=180)
    plt.close(fig)
    ratio_items = dict(failure_modes["metrics"]["ratios"])
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.bar(
        [label.replace("_", "\n") for label in ratio_items],
        list(ratio_items.values()),
        color="#fdae6b",
    )
    ax.axhline(1.0, color="0.25", linewidth=1.0, linestyle="--")
    ax.set_ylabel("RMSE ratio")
    ax.set_title("Latent ODE failure-mode probes")
    fig.savefig(failure_path, dpi=180)
    plt.close(fig)
    return {
        "latent_ode_prediction_figure": prediction_path,
        "latent_ode_rmse_comparison": rmse_path,
        "latent_ode_failure_modes": failure_path,
    }


def _trajectory_feature_matrix(
    trajectory: ReducedMHDTrajectory,
    *,
    lengths: tuple[float, float],
    mode: tuple[int, int],
) -> np.ndarray:
    energies = trajectory_energies(trajectory, lengths=lengths)
    mode_amplitude = np.asarray(trajectory_mode_amplitude(trajectory, mode=mode), dtype=np.float64)
    divergence = np.asarray(
        [
            float(
                magnetic_divergence_linf(
                    trajectory.states.__class__(
                        psi=trajectory.states.psi[index],
                        omega=trajectory.states.omega[index],
                    ),
                    lengths=lengths,
                )
            )
            for index in range(trajectory.states.psi.shape[0])
        ],
        dtype=np.float64,
    )
    psi_l2 = np.sqrt(np.mean(np.asarray(trajectory.states.psi, dtype=np.float64) ** 2, axis=(1, 2)))
    omega_l2 = np.sqrt(
        np.mean(np.asarray(trajectory.states.omega, dtype=np.float64) ** 2, axis=(1, 2))
    )
    return np.stack(
        (
            mode_amplitude,
            np.asarray(energies["magnetic"], dtype=np.float64),
            np.asarray(energies["kinetic"], dtype=np.float64),
            np.asarray(energies["total"], dtype=np.float64),
            divergence,
            psi_l2,
            omega_l2,
        ),
        axis=1,
    )


def _persistence_prediction(targets: np.ndarray, *, observation_count: int) -> np.ndarray:
    anchor = targets[:, observation_count - 1 : observation_count, :]
    return np.broadcast_to(anchor, targets.shape).copy()


def _linear_prefix_prediction(
    times: np.ndarray,
    targets: np.ndarray,
    *,
    observation_count: int,
) -> np.ndarray:
    if observation_count < 2:
        return _persistence_prediction(targets, observation_count=observation_count)
    first_time = float(times[0])
    last_time = float(times[observation_count - 1])
    denominator = max(last_time - first_time, np.finfo(np.float64).eps)
    first_value = targets[:, 0:1, :]
    last_value = targets[:, observation_count - 1 : observation_count, :]
    slope = (last_value - first_value) / denominator
    prediction = first_value + slope * (times.reshape((1, -1, 1)) - first_time)
    return prediction.astype(np.float64)


def _train_mean_time_prediction(targets: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    mean_history = np.mean(targets[train_indices], axis=0, keepdims=True)
    return np.broadcast_to(mean_history, targets.shape).copy()


def _forecast_scores(
    residual: np.ndarray,
    target_names: tuple[str, ...],
) -> dict[str, Any]:
    absolute = np.abs(residual)
    squared = residual**2
    mae_by_target = np.mean(absolute, axis=(0, 1))
    rmse_by_target = np.sqrt(np.mean(squared, axis=(0, 1)))
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "max_abs_error": float(np.max(absolute)),
        "mae_by_target": {
            name: float(value) for name, value in zip(target_names, mae_by_target, strict=True)
        },
        "rmse_by_target": {
            name: float(value) for name, value in zip(target_names, rmse_by_target, strict=True)
        },
    }


def _rmse_for_indices(
    residual: np.ndarray,
    sample_indices: np.ndarray,
    time_indices: np.ndarray,
) -> float:
    selected = residual[np.ix_(sample_indices, time_indices, np.arange(residual.shape[-1]))]
    return float(np.sqrt(np.mean(selected**2)))


def _target_sigma(train_residual: np.ndarray) -> np.ndarray:
    sigma = np.std(train_residual, axis=(0, 1), ddof=0)
    return np.maximum(sigma, np.finfo(np.float64).eps)


def _calibration_scores(
    residual: np.ndarray,
    sigma: np.ndarray,
    target_names: tuple[str, ...],
) -> dict[str, Any]:
    normalized = np.abs(residual) / sigma.reshape((1, 1, -1))
    coverage_one = np.mean(normalized <= 1.0, axis=(0, 1))
    coverage_two = np.mean(normalized <= 2.0, axis=(0, 1))
    return {
        "coverage_1sigma_mean": float(np.mean(coverage_one)),
        "coverage_2sigma_mean": float(np.mean(coverage_two)),
        "coverage_1sigma_by_target": {
            name: float(value) for name, value in zip(target_names, coverage_one, strict=True)
        },
        "coverage_2sigma_by_target": {
            name: float(value) for name, value in zip(target_names, coverage_two, strict=True)
        },
    }


def _indices_for_ids(all_ids: np.ndarray, selected_ids: Sequence[int]) -> np.ndarray:
    index_by_id = {int(identifier): index for index, identifier in enumerate(all_ids)}
    try:
        return np.asarray([index_by_id[int(identifier)] for identifier in selected_ids], dtype=int)
    except KeyError as exc:
        raise ValueError(f"split references unknown sample id {exc.args[0]}") from exc


def _experiment_spec(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
) -> dict[str, Any]:
    return {
        "schema": NEURAL_ODE_EXPERIMENT_SCHEMA,
        "dataset_schema": NEURAL_ODE_DATASET_SCHEMA,
        "split_schema": NEURAL_ODE_SPLIT_SCHEMA,
        "baseline_schema": NEURAL_ODE_BASELINE_SCHEMA,
        "calibration_schema": NEURAL_ODE_CALIBRATION_SCHEMA,
        "claim_level": "validation",
        "dataset": dataset.diagnostics,
        "splits": split.diagnostics,
        "baseline_passed": bool(baseline.metrics["passed"]),
        "protocol": {
            "purpose": (
                "Freeze deterministic trajectory data, split manifests, cheap "
                "baselines, and fitted latent-ODE comparisons under one schema."
            ),
            "ci_scale_training_only": True,
            "neural_ode_contract": {
                "inputs": "features[n_seed, n_time, n_feature], times[n_time]",
                "targets": "targets[n_seed, n_time, n_target]",
                "required_comparisons": [
                    "persistence",
                    "linear_prefix",
                    "train_mean_time",
                    "fitted_latent_ode",
                ],
                "required_reports": [
                    "MAE/RMSE by split",
                    "1-sigma and 2-sigma empirical coverage",
                    "failure cases for seed extrapolation and long-horizon drift",
                ],
            },
        },
    }


def _validation_report(
    dataset: NeuralODEDataset,
    split: SplitManifest,
    baseline: BaselineEvaluation,
    experiment: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_seed_qi_validation_passed": bool(dataset.diagnostics["source_validation_passed"]),
        "split_manifest_passed": bool(split.diagnostics["passed"]),
        "baseline_metrics_finite": bool(baseline.metrics["checks"]["finite_targets"])
        and bool(baseline.metrics["checks"]["finite_predictions"]),
        "baseline_metrics_passed": bool(baseline.metrics["passed"]),
        "calibration_report_passed": bool(baseline.calibration["passed"]),
    }
    return {
        "schema": NEURAL_ODE_REPRODUCIBILITY_GATES_SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostics": experiment,
    }


def _write_neural_ode_figures(
    dataset: NeuralODEDataset,
    baseline: BaselineEvaluation,
    figure_dir: Path,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    target_path = figure_dir / "dataset_targets.png"
    baseline_path = figure_dir / "baseline_rmse.png"
    calibration_path = figure_dir / "calibration_coverage.png"

    fig, axes = plt.subplots(
        len(dataset.target_names),
        1,
        figsize=(7.2, 2.4 * len(dataset.target_names)),
        constrained_layout=True,
        squeeze=False,
    )
    for target_index, target_name in enumerate(dataset.target_names):
        axis = axes[target_index, 0]
        for seed_index, seed in enumerate(dataset.seeds):
            axis.plot(dataset.times, dataset.targets[seed_index, :, target_index], label=f"{seed}")
        axis.set_ylabel(target_name)
        axis.set_xlabel("time")
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_title("Seed-QI neural-ODE targets")
    axes[0, 0].legend(title="seed", fontsize="x-small", ncols=3)
    fig.savefig(target_path, dpi=180)
    plt.close(fig)

    baseline_names = tuple(baseline.metrics["baselines"])
    test_rmse = [
        baseline.metrics["baselines"][name]["test"]["rmse"]
        for name in baseline_names
    ]
    fig, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    axis.bar(baseline_names, test_rmse)
    axis.set_ylabel("test RMSE")
    axis.set_title("Neural-ODE baseline forecast errors")
    axis.tick_params(axis="x", rotation=20)
    fig.savefig(baseline_path, dpi=180)
    plt.close(fig)

    coverage_one = [
        baseline.calibration["baselines"][name]["coverage"]["test"]["coverage_1sigma_mean"]
        for name in baseline_names
    ]
    coverage_two = [
        baseline.calibration["baselines"][name]["coverage"]["test"]["coverage_2sigma_mean"]
        for name in baseline_names
    ]
    x_positions = np.arange(len(baseline_names))
    fig, axis = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    width = 0.35
    axis.bar(x_positions - width / 2, coverage_one, width, label="1σ")
    axis.bar(x_positions + width / 2, coverage_two, width, label="2σ")
    axis.axhline(0.6827, color="0.4", linestyle="--", linewidth=1.0, label="Gaussian 1σ")
    axis.axhline(0.9545, color="0.7", linestyle=":", linewidth=1.0, label="Gaussian 2σ")
    axis.set_xticks(x_positions, baseline_names, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("empirical test coverage")
    axis.set_title("Baseline residual calibration")
    axis.legend(frameon=False, fontsize="small")
    fig.savefig(calibration_path, dpi=180)
    plt.close(fig)
    return {
        "dataset_targets": target_path,
        "baseline_rmse": baseline_path,
        "calibration_coverage": calibration_path,
    }


def _normalize_shape(shape: tuple[int, int]) -> tuple[int, int]:
    normalized = tuple(int(item) for item in shape)
    if len(normalized) != 2 or min(normalized) < 8:
        raise ValueError("shape must contain at least 8 points in each direction")
    return normalized
