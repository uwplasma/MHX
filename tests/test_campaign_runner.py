from __future__ import annotations

import json

import numpy as np
import pytest

from mhx.benchmarks import (
    RUTHERFORD_FAST_CAMPAIGN_SCHEMA,
    run_rutherford_campaign_fast,
    write_rutherford_campaign_fast,
)


def test_rutherford_campaign_fast_writes_validation_artifacts(tmp_path) -> None:
    manifest_path, validation = run_rutherford_campaign_fast(
        tmp_path,
        seeds=[7, 8],
        shape=(8, 8),
        dt=1.0e-2,
        steps=4,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert validation["diagnostics"]["schema"] == RUTHERFORD_FAST_CAMPAIGN_SCHEMA
    assert validation["diagnostics"]["claim_level"] == "validation"
    assert validation["diagnostics"]["duration_fraction_of_template"] < 1.0
    assert all(validation["checks"].values())

    manifest = json.loads(manifest_path.read_text())
    assert manifest["claim_level"] == "validation"
    assert manifest["config"]["schema"] == RUTHERFORD_FAST_CAMPAIGN_SCHEMA
    assert manifest["outputs"]["histories"] == "rutherford_fast_histories.npz"

    history = np.load(tmp_path / "rutherford_fast_histories.npz")
    assert history["schema"] == RUTHERFORD_FAST_CAMPAIGN_SCHEMA
    assert history["seed"].tolist() == [7, 8]
    assert history["time"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03, 0.04])
    assert history["reconnected_flux"].shape == (2, 5)
    assert history["rutherford_island_width"].shape == (2, 5)
    assert history["total_energy"].shape == (2, 5)
    assert np.isfinite(history["reconnection_rate_proxy"]).all()
    assert np.max(history["final_magnetic_divergence_linf"]) <= validation["thresholds"][
        "max_magnetic_divergence_linf"
    ]
    assert (tmp_path / "figures" / "rutherford_fast_histories.png").stat().st_size > 0
    assert (tmp_path / "campaign_template.json").exists()


def test_rutherford_campaign_fast_is_seed_deterministic(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    run_rutherford_campaign_fast(
        first_dir,
        seeds=3,
        shape=(8, 8),
        steps=4,
        make_figures=False,
    )
    run_rutherford_campaign_fast(
        second_dir,
        seeds=3,
        shape=(8, 8),
        steps=4,
        make_figures=False,
    )

    first = np.load(first_dir / "rutherford_fast_histories.npz")
    second = np.load(second_dir / "rutherford_fast_histories.npz")
    for key in (
        "time",
        "reconnected_flux",
        "rutherford_island_width",
        "magnetic_energy",
        "kinetic_energy",
        "total_energy",
        "final_magnetic_divergence_linf",
        "current_density_linf",
    ):
        np.testing.assert_allclose(first[key], second[key])


def test_write_rutherford_campaign_fast_single_seed_artifacts(tmp_path) -> None:
    manifest_path, validation = write_rutherford_campaign_fast(
        tmp_path,
        shape=(8, 8),
        t_end=0.04,
        dt=1.0e-2,
        write_gif=True,
    )

    assert manifest_path == tmp_path / "manifest.json"
    assert validation["passed"] is True
    assert (tmp_path / "rutherford_history.npz").exists()
    assert (tmp_path / "trajectory.npz").exists()
    assert (tmp_path / "figures" / "rutherford_histories.png").stat().st_size > 0
    assert (tmp_path / "figures" / "flux_movie.gif").stat().st_size > 0


def test_rutherford_campaign_fast_rejects_invalid_controls(tmp_path) -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_rutherford_campaign_fast(tmp_path, seeds=())
    with pytest.raises(ValueError, match="shape"):
        run_rutherford_campaign_fast(tmp_path, shape=(7, 8))
    with pytest.raises(ValueError, match="dt"):
        run_rutherford_campaign_fast(tmp_path, dt=0.0)
    with pytest.raises(ValueError, match="steps"):
        run_rutherford_campaign_fast(tmp_path, steps=0)
    with pytest.raises(ValueError, match="save_every"):
        run_rutherford_campaign_fast(tmp_path, steps=5, save_every=2)
    with pytest.raises(ValueError, match="claim_level"):
        run_rutherford_campaign_fast(tmp_path, claim_level="production")
    with pytest.raises(ValueError, match="t_end"):
        run_rutherford_campaign_fast(shape=(8, 8), t_end=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        run_rutherford_campaign_fast(shape=(8, 8), resistivity=-1.0)
    with pytest.raises(ValueError, match="positive"):
        run_rutherford_campaign_fast(shape=(8, 8), perturbation_amplitude=0.0)
    with pytest.raises(ValueError, match="noise_amplitude"):
        run_rutherford_campaign_fast(shape=(8, 8), noise_amplitude=-1.0)
    with pytest.raises(ValueError, match="magnetic_shear"):
        run_rutherford_campaign_fast(shape=(8, 8), magnetic_shear=0.0)
    with pytest.raises(ValueError, match="max_relative"):
        run_rutherford_campaign_fast(shape=(8, 8), max_relative_energy_growth=-1.0)
    with pytest.raises(ValueError, match="max_divergence"):
        run_rutherford_campaign_fast(shape=(8, 8), max_divergence_linf=-1.0)
