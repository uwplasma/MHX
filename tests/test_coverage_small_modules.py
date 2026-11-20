"""Edge-case and error-path tests for small MHX modules.

Covers the lazy ``mhx.run`` helper, JAX runtime configuration, manifest
writers, plugin loaders, the 3D simulation driver, sharding helpers and input
validation of the reduced-MHD equations and equilibria.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mhx
from mhx import ensemble as ensemble_module
from mhx import parallel as parallel_module
from mhx.diagnostics import reduced_mhd as diagnostics_module
from mhx.diagnostics.reduced_mhd import (
    compute_reduced_mhd_diagnostics,
    default_diagnostics_registry,
    load_diagnostics_entry_points,
)
from mhx.equations.reduced_mhd import (
    reduced_mhd_residual_norm,
    reduced_mhd_rhs,
    reduced_mhd_rhs_spectral,
)
from mhx.grids import CartesianGrid
from mhx.io import write_artifact_manifest
from mhx.numerics.spectral import pfft
from mhx.physics import PeriodicDoubleHarrisEquilibrium
from mhx.physics import terms as terms_module
from mhx.physics.terms import (
    PHYSICS_API_VERSION,
    build_physics_terms,
    default_physics_registry,
    load_physics_entry_points,
)
from mhx.runtime import configure_jax
from mhx.state import ReducedMHDParams, ReducedMHDState

TINY_CONFIG = """\
name = "tiny"

[mesh]
shape = [8, 8]
lower = [0.0, 0.0]
upper = [6.283185307179586, 6.283185307179586]
periodic = [true, true]

[time]
t1 = 0.02
dt = 0.01
save_every = 1
"""


# ---------------------------------------------------------------------------
# mhx.run and mhx.runtime
# ---------------------------------------------------------------------------


def test_lazy_run_helper_writes_manifest_in_requested_directory(tmp_path) -> None:
    config = tmp_path / "tiny.toml"
    config.write_text(TINY_CONFIG, encoding="utf-8")
    manifest = mhx.run(str(config), outdir=tmp_path / "out")
    assert manifest == tmp_path / "out" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["claim_level"] == "smoke"
    assert (tmp_path / "out" / "trajectory.npz").exists()


def test_lazy_run_helper_defaults_to_config_output_directory(tmp_path) -> None:
    config = tmp_path / "tiny.toml"
    target = tmp_path / "configured"
    config.write_text(f'output_dir = "{target.as_posix()}"\n' + TINY_CONFIG, encoding="utf-8")
    assert mhx.run(config) == target / "manifest.json"


def test_configure_jax_toggles_and_reports_precision() -> None:
    original = bool(jax.config.jax_enable_x64)
    try:
        assert configure_jax() is original
        assert configure_jax(enable_x64=False) is False
        assert bool(jax.config.jax_enable_x64) is False
        assert configure_jax(enable_x64=True) is True
        assert configure_jax() is True
    finally:
        jax.config.update("jax_enable_x64", original)


# ---------------------------------------------------------------------------
# Artifact manifest
# ---------------------------------------------------------------------------


def test_artifact_manifest_requires_existing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist or is not a directory"):
        write_artifact_manifest(tmp_path / "missing")
    plain_file = tmp_path / "file.txt"
    plain_file.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        write_artifact_manifest(plain_file)


def test_artifact_manifest_collects_valid_nested_claim_levels_only(tmp_path) -> None:
    root = tmp_path / "bundle"
    for name, text in {
        "good": json.dumps({"claim_level": "smoke"}),
        "no_level": json.dumps({"schema": "x"}),
        "non_string": json.dumps({"claim_level": 3}),
        "broken": "{not json",
    }.items():
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text(text, encoding="utf-8")
    manifest = write_artifact_manifest(root)
    assert manifest["claim_levels"] == {"good/manifest.json": "smoke"}
    listed = {item["path"] for item in manifest["files"]}
    assert listed == {
        f"{name}/manifest.json" for name in ("good", "no_level", "non_string", "broken")
    }
    assert "artifact_manifest.json" not in listed


# ---------------------------------------------------------------------------
# Plugin loading (physics and diagnostics)
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name, plugin):
        self.name = name
        self._plugin = plugin

    def load(self):
        return self._plugin


def test_physics_entry_point_without_register_hook_is_rejected(monkeypatch) -> None:
    class FakeEntryPoints(tuple):
        def select(self, *, group):
            return (_FakeEntryPoint("hookless", object()),) if group == "demo" else ()

    monkeypatch.setattr(terms_module.importlib_metadata, "entry_points", lambda: FakeEntryPoints())
    with pytest.raises(AttributeError, match="'hookless' in group 'demo'"):
        load_physics_entry_points(default_physics_registry(), ("demo",))


def test_diagnostics_entry_point_without_register_hook_is_rejected(monkeypatch) -> None:
    class FakeEntryPoints(tuple):
        def select(self, *, group):
            return (_FakeEntryPoint("hookless", object()),)

    monkeypatch.setattr(
        diagnostics_module.importlib_metadata, "entry_points", lambda: FakeEntryPoints()
    )
    with pytest.raises(AttributeError, match="register_diagnostics"):
        load_diagnostics_entry_points(default_diagnostics_registry(), ("demo",))


def test_entry_point_lookup_supports_legacy_mapping_api(monkeypatch) -> None:
    diagnostics_calls: list[object] = []
    physics_calls: list[object] = []
    # Both modules share ``importlib.metadata``, so one mapping serves both.
    legacy = {
        "demo_diagnostics": [_FakeEntryPoint("d", diagnostics_calls.append)],
        "demo_physics": [_FakeEntryPoint("p", physics_calls.append)],
    }
    monkeypatch.setattr(diagnostics_module.importlib_metadata, "entry_points", lambda: legacy)
    diagnostics_registry = default_diagnostics_registry()
    physics_registry = default_physics_registry()
    load_diagnostics_entry_points(diagnostics_registry, ("demo_diagnostics", "other"))
    load_physics_entry_points(physics_registry, ("demo_physics", "other"))
    assert diagnostics_calls == [diagnostics_registry]
    assert physics_calls == [physics_registry]


def test_build_physics_terms_loads_entry_point_groups(monkeypatch) -> None:
    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass(frozen=True)
    class Drive:
        amplitude: float = 0.0
        name: ClassVar[str] = "entry_drive"
        api_version: ClassVar[str] = PHYSICS_API_VERSION
        description: ClassVar[str] = "Entry-point drive."

        def rhs_addition(self, state, params, *, lengths):
            return state

    def register(registry):
        registry.register("entry_drive", lambda p: Drive(amplitude=float(p.get("amplitude", 0.0))))

    class FakeEntryPoints(tuple):
        def select(self, *, group):
            return (_FakeEntryPoint("hook", register),) if group == "demo" else ()

    monkeypatch.setattr(terms_module.importlib_metadata, "entry_points", lambda: FakeEntryPoints())
    (term,) = build_physics_terms(
        ("entry_drive",),
        {"entry_drive": {"amplitude": 0.25}},
        plugin_entry_point_groups=("demo",),
    )
    assert term.amplitude == 0.25


def test_compute_diagnostics_loads_entry_point_groups(monkeypatch) -> None:
    from mhx.diagnostics import DiagnosticSpec
    from mhx.time_integrators import evolve_rk4

    def register(registry):
        registry.register(
            DiagnosticSpec(
                name="entry_probe",
                description="Records the saved sample count.",
                output_keys=("entry_probe_samples",),
                compute=lambda context: {
                    "entry_probe_samples": int(context.trajectory.times.shape[0])
                },
            )
        )

    class FakeEntryPoints(tuple):
        def select(self, *, group):
            return (_FakeEntryPoint("hook", register),) if group == "demo" else ()

    monkeypatch.setattr(
        diagnostics_module.importlib_metadata, "entry_points", lambda: FakeEntryPoints()
    )
    grid = CartesianGrid(shape=(8, 8), lower=(0.0, 0.0), upper=(2 * np.pi, 2 * np.pi))
    state0 = PeriodicDoubleHarrisEquilibrium(perturbation_amplitude=1e-3).initial_state(grid)
    trajectory = evolve_rk4(
        state0, lambda s: ReducedMHDState(0 * s.psi, 0 * s.omega), dt=0.1, steps=2, save_every=1
    )
    result = compute_reduced_mhd_diagnostics(
        trajectory,
        initial_state=state0,
        lengths=grid.lengths,
        quantities=("entry_probe",),
        mode=(1, 1),
        fit_time_window=None,
        plugin_entry_point_groups=("demo",),
    )
    assert result["entry_probe_samples"] == 2
    assert result["diagnostic_quantities"] == ["entry_probe"]


@pytest.mark.parametrize(
    "module", [terms_module, diagnostics_module], ids=["physics", "diagnostics"]
)
def test_import_user_module_reports_missing_module(module, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ModuleNotFoundError, match="no_such_plugin_module_xyz"):
        module._import_user_module("no_such_plugin_module_xyz")


@pytest.mark.parametrize(
    "module", [terms_module, diagnostics_module], ids=["physics", "diagnostics"]
)
def test_import_user_module_file_fallback_with_cwd_on_path(module, tmp_path, monkeypatch) -> None:
    """The file-location fallback also works when the cwd is already importable."""
    (tmp_path / "late_plugin_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    real_import = module.importlib.import_module

    def stale_import(name, *args, **kwargs):
        if name == "late_plugin_mod":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", stale_import)
    try:
        loaded = module._import_user_module("late_plugin_mod")
    finally:
        monkeypatch.setattr(module.importlib, "import_module", real_import)
        sys.modules.pop("late_plugin_mod", None)
    assert loaded.VALUE == 7


@pytest.mark.parametrize(
    "module", [terms_module, diagnostics_module], ids=["physics", "diagnostics"]
)
def test_import_user_module_reraises_when_no_loader_spec(module, tmp_path, monkeypatch) -> None:
    (tmp_path / "specless_plugin_mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    real_import = module.importlib.import_module

    def stale_import(name, *args, **kwargs):
        if name == "specless_plugin_mod":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(module.importlib, "import_module", stale_import)
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)
    with pytest.raises(ModuleNotFoundError, match="specless_plugin_mod"):
        module._import_user_module("specless_plugin_mod")


# ---------------------------------------------------------------------------
# Reduced-MHD equations, equilibria, sharding helpers
# ---------------------------------------------------------------------------


def _state(shape=(8, 8)) -> ReducedMHDState:
    return ReducedMHDState(psi=jnp.ones(shape), omega=jnp.zeros(shape))


def test_spectral_rhs_validates_inputs() -> None:
    params = ReducedMHDParams(resistivity=1e-2, viscosity=1e-2)
    lengths = (2 * np.pi, 2 * np.pi)
    with pytest.raises(ValueError, match="same 2D shape"):
        reduced_mhd_rhs_spectral(
            ReducedMHDState(jnp.ones((2, 8, 8)), jnp.ones((2, 8, 8))), params, lengths=lengths
        )
    with pytest.raises(ValueError, match="same 2D shape"):
        reduced_mhd_rhs_spectral(
            ReducedMHDState(jnp.ones((8, 8)), jnp.ones((8, 4))), params, lengths=lengths
        )
    with pytest.raises(ValueError, match="expected 2 lengths, got 3"):
        reduced_mhd_rhs_spectral(_state(), params, lengths=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="dealiasing must be"):
        reduced_mhd_rhs_spectral(_state(), params, lengths=lengths, dealiasing="bad")


def test_spectral_rhs_matches_physical_rhs() -> None:
    params = ReducedMHDParams(resistivity=1e-2, viscosity=2e-2)
    lengths = (2 * np.pi, 2 * np.pi)
    grid = CartesianGrid(shape=(16, 16), lower=(0.0, 0.0), upper=lengths)
    state = PeriodicDoubleHarrisEquilibrium(perturbation_amplitude=1e-2).initial_state(grid)
    physical = reduced_mhd_rhs(state, params, lengths=lengths, dealiasing="two_thirds")
    state_hat = ReducedMHDState(jnp.fft.fft2(state.psi), jnp.fft.fft2(state.omega))
    spectral = reduced_mhd_rhs_spectral(state_hat, params, lengths=lengths, dealiasing="two_thirds")
    np.testing.assert_allclose(
        np.real(np.fft.ifft2(np.asarray(spectral.psi))), np.asarray(physical.psi), atol=1e-9
    )


def test_residual_norm_is_root_mean_square() -> None:
    state = ReducedMHDState(psi=3.0 * jnp.ones((4, 4)), omega=4.0 * jnp.ones((4, 4)))
    assert float(reduced_mhd_residual_norm(state)) == pytest.approx(5.0)
    assert float(reduced_mhd_residual_norm(_state((4, 4)))) == pytest.approx(1.0)


def test_double_harris_equilibrium_rejects_degenerate_parameters() -> None:
    grid = CartesianGrid(shape=(8, 8), lower=(0.0, 0.0), upper=(2 * np.pi, 2 * np.pi))
    with pytest.raises(ValueError, match="width must be positive"):
        PeriodicDoubleHarrisEquilibrium(width=0.0).initial_state(grid)
    with pytest.raises(ValueError, match="amplitude must be nonzero"):
        PeriodicDoubleHarrisEquilibrium(amplitude=0.0).initial_state(grid)


def test_shard_physical_is_identity_without_a_mesh() -> None:
    field = jnp.arange(8.0).reshape(2, 2, 2)
    assert pfft.shard_physical(field, None) is field


def test_shard_physical_distributes_over_devices() -> None:
    from jax.sharding import Mesh

    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("needs at least two JAX devices")
    mesh = Mesh(np.asarray(devices), axis_names=("x",))
    field = jnp.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    sharded = pfft.shard_physical(field, mesh)
    assert len(sharded.sharding.device_set) == 2
    np.testing.assert_array_equal(np.asarray(sharded), np.asarray(field))


def test_shard_map_falls_back_to_check_rep_on_older_jax(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def old_shard_map(function, *, mesh, in_specs, out_specs, **kwargs):
        if "check_vma" in kwargs:
            raise TypeError("unexpected keyword 'check_vma'")
        calls.append(kwargs)
        return "mapped"

    monkeypatch.setattr(jax, "shard_map", old_shard_map)
    assert (
        pfft._shard_map(lambda x: x, mesh=None, in_specs=None, out_specs=None) == "mapped"
        if False
        else pfft._shard_map(lambda x: x, None, None, None) == "mapped"
    )
    assert calls == [{"check_rep": False}]


def test_make_device_mesh_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="device_count must be at least 1"):
        parallel_module.make_device_mesh(0)
    with pytest.raises(ValueError, match="requested 1000 JAX devices"):
        parallel_module.make_device_mesh(1000)


def test_shard_batch_uses_experimental_shard_map_on_old_jax(monkeypatch) -> None:
    pytest.importorskip("jax.experimental.shard_map")
    mesh = parallel_module.make_device_mesh(1)
    monkeypatch.delattr(parallel_module.solvax, "shard_batch", raising=False)
    monkeypatch.delattr(jax, "shard_map")
    mapped = parallel_module.shard_batch(
        lambda values: 2.0 * values, mesh=mesh, input_rank=1, output_rank=1
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jit(mapped)(jnp.arange(4.0))), 2.0 * np.arange(4.0)
    )


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def test_host_array_gathers_arrays_spanning_processes(monkeypatch) -> None:
    from jax.experimental import multihost_utils

    class RemoteArray:
        is_fully_addressable = False

    gathered = np.arange(6.0).reshape(2, 3)
    seen = {}

    def fake_allgather(array, tiled):
        seen["tiled"] = tiled
        return gathered

    monkeypatch.setattr(multihost_utils, "process_allgather", fake_allgather)
    out = ensemble_module._host_array(RemoteArray())
    np.testing.assert_array_equal(out, gathered)
    assert seen == {"tiled": True}
    local = jnp.arange(3.0)
    np.testing.assert_array_equal(ensemble_module._host_array(local), np.arange(3.0))


def test_ensemble_final_save_step_is_included_when_stride_does_not_divide() -> None:
    cases = tuple(
        PeriodicDoubleHarrisEquilibrium(perturbation_amplitude=1.0e-3 * (case + 1))
        for case in range(2)
    )
    result = mhx.Simulation(
        shape=(8, 8), dt=0.001, t_end=0.003, save_every=2, verbose=False
    ).run_ensemble(cases)
    np.testing.assert_allclose(np.asarray(result.trajectory.times), [0.002, 0.003])
    assert result.trajectory.states.psi.shape == (2, 2, 8, 8)


# ---------------------------------------------------------------------------
# 3D driver
# ---------------------------------------------------------------------------


def _sim3d(**overrides):
    settings = dict(
        shape=(8, 8, 8),
        equations="mhd3d",
        equilibrium=mhx.OrszagTang3DEquilibrium(),
        viscosity=5.0e-3,
        resistivity=5.0e-3,
        dt=1.0e-2,
        t_end=0.04,
        save_every=2,
        verbose=False,
    )
    settings.update(overrides)
    return mhx.Simulation(**settings)


def test_mhd3d_save_accepts_explicit_npz_path(tmp_path) -> None:
    result = _sim3d().run()
    target = tmp_path / "nested" / "custom.npz"
    written = result.save(target)
    assert written == target
    payload = np.load(written)
    assert payload["velocity"].shape == (2, 3, 8, 8, 8)


def test_mhd3d_forcing_changes_the_evolution() -> None:
    baseline = _sim3d().run()
    zero = jax.tree.map(jnp.zeros_like, baseline.final_state)
    unchanged = _sim3d(forcing=zero).run()
    np.testing.assert_allclose(
        np.asarray(unchanged.final_state.v_hat), np.asarray(baseline.final_state.v_hat), atol=1e-12
    )

    kick = type(baseline.final_state)(
        v_hat=jnp.full_like(baseline.final_state.v_hat, 1.0e-2),
        b_hat=jnp.zeros_like(baseline.final_state.b_hat),
    )
    forced = _sim3d(forcing=kick).run()
    difference = np.max(
        np.abs(np.asarray(forced.final_state.v_hat) - np.asarray(baseline.final_state.v_hat))
    )
    assert difference > 1.0e-4


def test_run_mhd3d_rejects_indivisible_device_count() -> None:
    if jax.device_count() < 2:
        pytest.skip("needs at least two JAX devices")
    with pytest.raises(ValueError, match="divide evenly"):
        _sim3d(shape=(9, 8, 8), device_count=2).run()


def test_import_of_module_type_is_available() -> None:
    assert isinstance(sys.modules["mhx"], types.ModuleType)
    assert Path(mhx.__file__).name == "__init__.py"
