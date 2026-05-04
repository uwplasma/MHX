from __future__ import annotations

import pytest

from mhx.config import (
    DiagnosticsConfig,
    MeshConfig,
    PhysicsConfig,
    RunConfig,
    TimeConfig,
    load_config,
)


def test_load_example_config() -> None:
    cfg = load_config("examples/linear_tearing.toml")
    assert cfg.name == "linear_tearing_smoke"
    assert cfg.mesh.shape == (32, 32)
    assert cfg.physics.resistivity == pytest.approx(1.0e-3)
    assert cfg.diagnostics.mode == (1, 1)
    assert cfg.diagnostics.fit_time_window == (0.02, 0.1)


def test_config_roundtrip_dict_and_toml() -> None:
    cfg = RunConfig()
    data = cfg.to_dict()
    assert data["mesh"]["shape"] == [32, 32]
    assert data["diagnostics"]["mode"] == [1, 1]
    assert data["diagnostics"]["fit_time_window"] is None
    assert "[mesh]" in cfg.to_toml()
    assert cfg.with_output_dir("outputs/other").output_dir.as_posix() == "outputs/other"


def test_config_validation_errors() -> None:
    with pytest.raises(ValueError, match="mesh.shape"):
        MeshConfig(shape=(2, 32)).validated()
    with pytest.raises(ValueError, match="must have length"):
        MeshConfig.from_mapping({"shape": [8]})
    with pytest.raises(ValueError, match="must not be None"):
        MeshConfig.from_mapping({"shape": None})
    with pytest.raises(ValueError, match="mesh.upper"):
        MeshConfig(upper=(0.0, 1.0)).validated()
    with pytest.raises(ValueError, match="time.t1"):
        TimeConfig(t0=1.0, t1=1.0).validated()
    with pytest.raises(ValueError, match="time.dt"):
        TimeConfig(dt=0.0).validated()
    with pytest.raises(ValueError, match="time.save_every"):
        TimeConfig(save_every=0).validated()
    with pytest.raises(ValueError, match="resistivity"):
        PhysicsConfig(resistivity=-1.0).validated()
    with pytest.raises(ValueError, match="viscosity"):
        PhysicsConfig(viscosity=-1.0).validated()
    with pytest.raises(ValueError, match="fit_time_window"):
        DiagnosticsConfig(fit_time_window=(1.0, 0.0)).validated()
