"""Tests for the differentiable Kelvin--Helmholtz benchmark module."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhx.benchmarks.kelvin_helmholtz import (
    KelvinHelmholtzConfig,
    KelvinHelmholtzDyeState,
    dye_entropy,
    dye_entropy_history,
    kelvin_helmholtz_dye_rhs,
    kelvin_helmholtz_entropy_jvp,
    kelvin_helmholtz_entropy_objective,
    kelvin_helmholtz_entropy_value_and_grad,
    kelvin_helmholtz_grid,
    kelvin_helmholtz_initial_state,
    kelvin_helmholtz_initial_state_from_config,
    run_kelvin_helmholtz_dye,
)
from mhx.equations.reduced_mhd import stream_function
from mhx.state import ReducedMHDParams

SMALL = KelvinHelmholtzConfig(shape=(16, 32), dt=1.0e-3, t_end=8.0e-3, save_every=4)


@pytest.fixture(scope="module")
def small_run():
    return run_kelvin_helmholtz_dye(SMALL)


def test_config_defaults_and_steps() -> None:
    cfg = KelvinHelmholtzConfig()
    assert cfg.validated() is cfg
    assert cfg.steps == 100
    assert KelvinHelmholtzConfig(dt=1.0e-3, t_end=8.0e-3).steps == 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shape": (2, 2)}, "entries must be >= 4"),
        ({"viscosity": -1.0}, "viscosity must be non-negative"),
        ({"resistivity": -1.0}, "resistivity must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"t_end": 0.0}, "t_end must be positive"),
        ({"save_every": 0}, "save_every must be >= 1"),
        ({"dt": 1.0, "t_end": 0.1}, "at least one step"),
        ({"shear_width": 0.0}, "shear_width must be positive"),
        ({"perturbation_width": -0.1}, "perturbation_width must be positive"),
        ({"flow_speed": 0.0}, "flow_speed must be positive"),
    ],
)
def test_config_validation_errors(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        KelvinHelmholtzConfig(**overrides).validated()


def test_grid_matches_config() -> None:
    grid = kelvin_helmholtz_grid(SMALL)
    assert tuple(grid.lengths) == pytest.approx((1.0, 2.0))
    x, y = grid.mesh()
    assert x.shape == y.shape == SMALL.shape
    with pytest.raises(ValueError, match="dt must be positive"):
        kelvin_helmholtz_grid(dataclasses.replace(SMALL, dt=-1.0))


def test_initial_state_dye_and_shear_layers() -> None:
    cfg = KelvinHelmholtzConfig(shape=(16, 128), perturbation_amplitude=0.0)
    grid = kelvin_helmholtz_grid(cfg)
    state = kelvin_helmholtz_initial_state_from_config(grid, cfg)
    _, y = grid.mesh()
    dye = np.asarray(state.dye)
    assert dye.shape == cfg.shape
    assert dye.min() >= 0.0 and dye.max() <= 1.0
    yv = np.asarray(y)
    inside = (yv > 0.7) & (yv < 1.3)
    outside = (yv < 0.3) | (yv > 1.7)
    # dye is 0 between the two shear layers and 1 outside them
    assert np.all(dye[inside] < 0.01)
    assert np.all(dye[outside] > 0.99)
    # Without perturbation the vorticity is -dux/dy: negative at y1, positive at y2.
    omega = np.asarray(state.mhd.omega)
    assert omega.shape == cfg.shape
    j1 = int(np.argmin(np.abs(yv[0] - cfg.y1)))
    j2 = int(np.argmin(np.abs(yv[0] - cfg.y2)))
    speed_over_w = cfg.flow_speed / cfg.shear_width
    for j, center, sign in ((j1, cfg.y1, -1.0), (j2, cfg.y2, 1.0)):
        analytic = sign * speed_over_w / np.cosh((yv[0, j] - center) / cfg.shear_width) ** 2
        assert omega[0, j] == pytest.approx(analytic, rel=1e-10)
        assert abs(omega[0, j]) > 0.9 * speed_over_w
    assert abs(omega[0, 0]) < 1e-6
    # omega is independent of x and has (almost) zero mean over the periodic box
    np.testing.assert_allclose(omega, omega[:1, :].repeat(omega.shape[0], 0), atol=1e-12)
    assert abs(omega.mean()) < 1e-3 * cfg.flow_speed / cfg.shear_width


def test_perturbation_scales_linearly() -> None:
    grid = kelvin_helmholtz_grid(SMALL)
    base = kelvin_helmholtz_initial_state(grid, perturbation_amplitude=0.0).mhd.omega
    one = kelvin_helmholtz_initial_state(grid, perturbation_amplitude=0.01).mhd.omega
    two = kelvin_helmholtz_initial_state(grid, perturbation_amplitude=0.02).mhd.omega
    np.testing.assert_allclose(two - base, 2.0 * (one - base), rtol=1e-10, atol=1e-14)
    assert float(jnp.max(jnp.abs(one - base))) > 0.0
    # The perturbation is a pure cos(kx x) mode, so it has zero x-mean.
    assert float(jnp.max(jnp.abs(jnp.mean(one - base, axis=0)))) < 1e-12


def test_initial_state_from_config_matches_direct_call() -> None:
    cfg = KelvinHelmholtzConfig(
        shape=(16, 32),
        shear_width=0.1,
        perturbation_width=0.3,
        perturbation_amplitude=0.05,
        flow_speed=0.7,
        y1=0.4,
        y2=1.6,
    )
    grid = kelvin_helmholtz_grid(cfg)
    from_cfg = kelvin_helmholtz_initial_state_from_config(grid, cfg)
    direct = kelvin_helmholtz_initial_state(
        grid,
        shear_width=0.1,
        perturbation_width=0.3,
        perturbation_amplitude=0.05,
        flow_speed=0.7,
        y1=0.4,
        y2=1.6,
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(from_cfg), jax.tree_util.tree_leaves(direct), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="flow_speed must be positive"):
        kelvin_helmholtz_initial_state_from_config(grid, dataclasses.replace(cfg, flow_speed=-1.0))


def test_stream_function_of_initial_vorticity_is_finite() -> None:
    grid = kelvin_helmholtz_grid(SMALL)
    state = kelvin_helmholtz_initial_state_from_config(grid, SMALL)
    phi = stream_function(state.mhd.omega, lengths=grid.lengths)
    assert phi.shape == SMALL.shape
    assert bool(jnp.all(jnp.isfinite(phi)))


def test_dye_rhs_conserves_mean_and_is_steady_without_perturbation() -> None:
    cfg = dataclasses.replace(SMALL, perturbation_amplitude=0.0, viscosity=0.0)
    grid = kelvin_helmholtz_grid(cfg)
    state = kelvin_helmholtz_initial_state_from_config(grid, cfg)
    params = ReducedMHDParams(resistivity=0.0, viscosity=0.0)
    rhs = kelvin_helmholtz_dye_rhs(state, params, lengths=grid.lengths)
    assert isinstance(rhs, KelvinHelmholtzDyeState)
    # Pure x-shear flow with an x-independent dye: nothing to advect.
    assert float(jnp.max(jnp.abs(rhs.dye))) < 1e-9
    assert float(jnp.max(jnp.abs(rhs.mhd.omega))) < 1e-6

    perturbed = kelvin_helmholtz_initial_state_from_config(grid, SMALL)
    visc = ReducedMHDParams(resistivity=0.0, viscosity=1.0e-2)
    rhs_p = kelvin_helmholtz_dye_rhs(perturbed, visc, lengths=grid.lengths)
    assert float(jnp.max(jnp.abs(rhs_p.dye))) > 0.0
    assert abs(float(jnp.mean(rhs_p.dye))) < 1e-9


def test_dye_entropy_closed_forms() -> None:
    grid = kelvin_helmholtz_grid(SMALL)
    area = 2.0
    const = jnp.full(SMALL.shape, 0.5)
    assert float(dye_entropy(const, grid)) == pytest.approx(area * 0.5 * np.log(2.0), rel=1e-12)
    # Pure dye (0 or 1) carries no entropy; values are clipped so the result stays finite.
    assert float(dye_entropy(jnp.zeros(SMALL.shape), grid)) == pytest.approx(0.0, abs=1e-10)
    assert float(dye_entropy(jnp.ones(SMALL.shape), grid)) == pytest.approx(0.0, abs=1e-12)
    # Clipping: out-of-range values behave like the nearest bound.
    assert float(dye_entropy(jnp.full(SMALL.shape, 2.0), grid)) == pytest.approx(0.0, abs=1e-12)
    two_valued = jnp.where(jnp.arange(SMALL.shape[1]) < 16, 0.25, 1.0) * jnp.ones(SMALL.shape)
    expected = area * 0.5 * (-0.25 * np.log(0.25))
    assert float(dye_entropy(two_valued, grid)) == pytest.approx(expected, rel=1e-12)


def test_run_result_structure(small_run) -> None:
    res = small_run
    n_saved = res.trajectory.times.shape[0]
    assert res.config == SMALL
    assert res.params.viscosity == SMALL.viscosity
    assert res.params.resistivity == SMALL.resistivity
    assert res.trajectory.states.dye.shape == (n_saved, *SMALL.shape)
    assert res.trajectory.states.mhd.omega.shape == (n_saved, *SMALL.shape)
    assert res.entropy.shape == (n_saved,)
    # The initial state is not saved; the first sample is one stride in.
    assert float(res.trajectory.times[0]) == pytest.approx(SMALL.save_every * SMALL.dt)
    assert n_saved == SMALL.steps // SMALL.save_every
    assert float(res.trajectory.times[-1]) == pytest.approx(SMALL.t_end)
    assert bool(jnp.all(jnp.diff(res.trajectory.times) > 0))
    drift = jnp.max(jnp.abs(res.trajectory.states.dye[0] - res.initial_state.dye))
    assert 0.0 < float(drift) < 1.0e-2
    np.testing.assert_allclose(res.final_state.dye, res.trajectory.states.dye[-1])
    assert res.final_state.dye.shape == SMALL.shape
    assert bool(jnp.all(jnp.isfinite(res.trajectory.states.dye)))
    assert bool(jnp.all(jnp.isfinite(res.trajectory.states.mhd.omega)))
    assert bool(jnp.all(jnp.isfinite(res.entropy)))


def test_run_conserves_dye_mean_and_history_matches(small_run) -> None:
    res = small_run
    means = jnp.mean(res.trajectory.states.dye, axis=(1, 2))
    np.testing.assert_allclose(means, means[0], atol=1e-10)
    np.testing.assert_allclose(
        dye_entropy_history(res.trajectory, res.grid), res.entropy, rtol=1e-12
    )
    initial_entropy = float(dye_entropy(res.initial_state.dye, res.grid))
    assert float(res.entropy[0]) == pytest.approx(initial_entropy, rel=1e-2)


def test_run_default_config_argument_validates() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        run_kelvin_helmholtz_dye(dataclasses.replace(SMALL, dt=0.0))


def test_objective_matches_run_and_default_config(small_run) -> None:
    value = kelvin_helmholtz_entropy_objective(jnp.asarray(SMALL.perturbation_amplitude), SMALL)
    assert float(value) == pytest.approx(float(small_run.entropy[-1]), rel=1e-12)
    default = kelvin_helmholtz_entropy_objective(jnp.asarray(1.0e-2))
    assert bool(jnp.isfinite(default))
    assert default.shape == ()


def _tiny_config() -> KelvinHelmholtzConfig:
    return KelvinHelmholtzConfig(shape=(8, 16), dt=1.0e-3, t_end=4.0e-3, save_every=4)


def test_value_and_grad_matches_finite_difference() -> None:
    cfg = _tiny_config()
    amp = 0.05
    value, grad = kelvin_helmholtz_entropy_value_and_grad(amp, cfg)
    assert float(value) == pytest.approx(
        float(kelvin_helmholtz_entropy_objective(jnp.asarray(amp), cfg)), rel=1e-12
    )
    h = 1.0e-4
    plus = kelvin_helmholtz_entropy_objective(jnp.asarray(amp + h), cfg)
    minus = kelvin_helmholtz_entropy_objective(jnp.asarray(amp - h), cfg)
    fd = float(plus - minus) / (2.0 * h)
    assert bool(jnp.isfinite(grad))
    assert float(grad) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_jvp_matches_gradient_times_tangent_and_finite_difference() -> None:
    cfg = _tiny_config()
    amp, tangent = 0.05, 2.0
    value, dvalue = kelvin_helmholtz_entropy_jvp(amp, tangent, cfg)
    ref_value, grad = kelvin_helmholtz_entropy_value_and_grad(amp, cfg)
    assert float(value) == pytest.approx(float(ref_value), rel=1e-12)
    assert float(dvalue) == pytest.approx(tangent * float(grad), rel=1e-8, abs=1e-12)
    h = 1.0e-4
    plus = kelvin_helmholtz_entropy_objective(jnp.asarray(amp + h * tangent), cfg)
    minus = kelvin_helmholtz_entropy_objective(jnp.asarray(amp - h * tangent), cfg)
    assert float(dvalue) == pytest.approx(float(plus - minus) / (2.0 * h), rel=1e-4, abs=1e-9)


def test_value_and_grad_uses_default_config() -> None:
    value, grad = kelvin_helmholtz_entropy_value_and_grad(1.0e-2)
    assert bool(jnp.isfinite(value)) and bool(jnp.isfinite(grad))
    value_j, dvalue = kelvin_helmholtz_entropy_jvp(1.0e-2, 1.0)
    assert float(value_j) == pytest.approx(float(value), rel=1e-12)
    assert float(dvalue) == pytest.approx(float(grad), rel=1e-8, abs=1e-12)
