from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import solvax

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import MeshConfig, NumericsConfig, PhysicsConfig, RunConfig, TimeConfig
from mhx.numerics import (
    MatrixFreeOperator,
    as_solvax_operator,
    complex_linear_extension,
    spectral_diffusion_preconditioner,
)
from mhx.state import ReducedMHDParams, ReducedMHDState
from mhx.time_integrators import backward_euler_step, evolve_backward_euler, evolve_rk4

jax.config.update("jax_enable_x64", True)


def test_mhx_operator_and_gmres_adapters() -> None:
    operator = MatrixFreeOperator(
        shape=(6,),
        matvec=lambda vector: 2.0 * vector,
        name="twice",
    )
    adapted = as_solvax_operator(operator)
    rhs = jnp.arange(1.0, 7.0)

    solution = solvax.gmres(adapted, rhs, restart=4, max_restarts=2)

    assert adapted.shape == (6, 6)
    assert bool(solution.converged)
    assert jnp.allclose(solution.x, 0.5 * rhs)

    with pytest.raises(ValueError, match="flat MHX vector"):
        as_solvax_operator(MatrixFreeOperator(shape=(2, 3), matvec=lambda vector: vector))


def test_complex_linear_extension_applies_real_jvp_twice() -> None:
    extended = complex_linear_extension(lambda vector: 3.0 * vector)
    vector = jnp.asarray([1.0 + 2.0j, -3.0 + 4.0j])

    assert jnp.allclose(extended(vector), 3.0 * vector)


def test_spectral_diffusion_preconditioner_inverts_principal_part() -> None:
    shape = (12, 10)
    x = 2.0 * jnp.pi * jnp.arange(shape[0]) / shape[0]
    y = 2.0 * jnp.pi * jnp.arange(shape[1]) / shape[1]
    psi = jnp.sin(2.0 * x[:, None]) * jnp.ones((1, shape[1]))
    omega = jnp.ones((shape[0], 1)) * jnp.cos(3.0 * y[None, :])
    state = ReducedMHDState(psi=psi, omega=omega)
    params = ReducedMHDParams(resistivity=0.2, viscosity=0.3)
    dt = 0.1
    precondition = spectral_diffusion_preconditioner(
        params,
        lengths=(2.0 * jnp.pi, 2.0 * jnp.pi),
        dt=dt,
    )

    result = precondition(state)

    assert jnp.allclose(result.psi, psi / (1.0 + dt * params.resistivity * 4.0))
    assert jnp.allclose(result.omega, omega / (1.0 + dt * params.viscosity * 9.0))

    complex_result = precondition(
        ReducedMHDState(
            psi=psi.astype(jnp.complex128) * (1.0 + 1.0j),
            omega=omega.astype(jnp.complex128) * (1.0 - 1.0j),
        )
    )
    assert jnp.iscomplexobj(complex_result.psi)
    assert jnp.iscomplexobj(complex_result.omega)

    with pytest.raises(ValueError, match="dt"):
        spectral_diffusion_preconditioner(params, lengths=(1.0, 1.0), dt=0.0)
    with pytest.raises(ValueError, match="lengths"):
        spectral_diffusion_preconditioner(params, lengths=(1.0,), dt=0.1)
    with pytest.raises(ValueError, match="lengths"):
        spectral_diffusion_preconditioner(params, lengths=(1.0, -1.0), dt=0.1)


def test_backward_euler_matches_linear_decay_and_saves_tail() -> None:
    state0 = ReducedMHDState(
        psi=jnp.asarray([1.0, 2.0]),
        omega=jnp.asarray([-1.0, 3.0]),
    )

    def rhs(state: ReducedMHDState) -> ReducedMHDState:
        return ReducedMHDState(psi=-2.0 * state.psi, omega=-3.0 * state.omega)

    step = backward_euler_step(state0, rhs, 0.1)
    trajectory = evolve_backward_euler(
        state0,
        rhs,
        dt=0.1,
        steps=3,
        save_every=2,
        t0=1.0,
    )

    assert bool(step.converged)
    assert bool(step.linear_converged)
    assert jnp.allclose(step.x.psi, state0.psi / 1.2)
    assert jnp.allclose(step.x.omega, state0.omega / 1.3)
    assert jnp.allclose(trajectory.trajectory.times, jnp.asarray([1.2, 1.3]))
    assert bool(jnp.all(trajectory.converged))
    assert bool(jnp.all(trajectory.linear_converged))


def test_fixed_and_implicit_integrator_validation() -> None:
    state = ReducedMHDState(psi=jnp.ones(2), omega=jnp.ones(2))

    def rhs(value: ReducedMHDState) -> ReducedMHDState:
        return value

    with pytest.raises(ValueError, match="dt"):
        backward_euler_step(state, rhs, 0.0)
    for kwargs, match in (
        ({"steps": 0, "dt": 0.1}, "steps"),
        ({"steps": 1, "dt": 0.0}, "dt"),
        ({"steps": 1, "dt": 0.1, "save_every": 0}, "save_every"),
    ):
        with pytest.raises(ValueError, match=match):
            evolve_backward_euler(state, rhs, **kwargs)
        with pytest.raises(ValueError, match=match):
            evolve_rk4(state, rhs, **kwargs)


def test_run_config_selects_implicit_solvax_backend() -> None:
    config = RunConfig(
        mesh=MeshConfig(shape=(8, 8)),
        time=TimeConfig(t0=0.5, t1=0.52, dt=0.01),
        physics=PhysicsConfig(resistivity=0.02, viscosity=0.02),
        numerics=NumericsConfig(
            time_integrator="backward_euler",
            nonlinear_max_steps=5,
            linear_restart=12,
            linear_max_restarts=4,
        ),
    )

    trajectory, diagnostics = run_linear_tearing_smoke(config)

    assert trajectory.times[-1] == 0.52
    assert diagnostics["final_time"] == 0.52
    assert diagnostics["dealiasing"] == "two_thirds"
    assert diagnostics["time_integrator"] == "backward_euler"
    assert diagnostics["implicit_converged"] is True
    assert diagnostics["implicit_linear_converged"] is True
