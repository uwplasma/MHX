from __future__ import annotations

import jax.numpy as jnp

from mhx.solver.plugins import (
    LinearDragTerm,
    HyperResistivityTerm,
    HallTerm,
    AnisotropicPressureTerm,
    apply_terms,
    validate_term,
)


def test_plugin_apply_shapes():
    v_hat = jnp.zeros((3, 4, 4, 1))
    B_hat = jnp.zeros((3, 4, 4, 1))
    kx = jnp.zeros((4, 4, 1))
    ky = jnp.zeros((4, 4, 1))
    kz = jnp.zeros((4, 4, 1))
    k2 = jnp.ones((4, 4, 1))
    mask = jnp.ones((4, 4, 1), dtype=bool)

    terms = [LinearDragTerm(mu=0.1), HyperResistivityTerm(eta4=0.01)]
    dv, dB = apply_terms(
        terms,
        t=0.0,
        v_hat=v_hat,
        B_hat=B_hat,
        kx=kx,
        ky=ky,
        kz=kz,
        k2=k2,
        mask_dealias=mask,
    )
    assert dv.shape == v_hat.shape
    assert dB.shape == B_hat.shape


def test_hall_nonzero():
    v_hat = jnp.zeros((3, 4, 4, 1))
    B_hat = jnp.ones((3, 4, 4, 1))
    kx = jnp.ones((4, 4, 1))
    ky = 2.0 * jnp.ones((4, 4, 1))
    kz = 3.0 * jnp.ones((4, 4, 1))
    k2 = kx**2 + ky**2 + kz**2
    mask = jnp.ones((4, 4, 1), dtype=bool)

    dv, dB = apply_terms(
        [HallTerm(d_h=1e-2)],
        t=0.0,
        v_hat=v_hat,
        B_hat=B_hat,
        kx=kx,
        ky=ky,
        kz=kz,
        k2=k2,
        mask_dealias=mask,
    )
    assert dv.shape == v_hat.shape
    assert dB.shape == B_hat.shape
    assert jnp.any(jnp.abs(dB) > 0.0)


def test_anisotropic_pressure_shapes():
    v_hat = jnp.ones((3, 4, 4, 1))
    B_hat = jnp.zeros((3, 4, 4, 1))
    kx = jnp.zeros((4, 4, 1))
    ky = jnp.zeros((4, 4, 1))
    kz = jnp.ones((4, 4, 1))
    k2 = kx**2 + ky**2 + kz**2
    mask = jnp.ones((4, 4, 1), dtype=bool)

    dv, dB = apply_terms(
        [AnisotropicPressureTerm(chi=0.1)],
        t=0.0,
        v_hat=v_hat,
        B_hat=B_hat,
        kx=kx,
        ky=ky,
        kz=kz,
        k2=k2,
        mask_dealias=mask,
    )
    assert dv.shape == v_hat.shape
    assert dB.shape == B_hat.shape


def test_plugin_validate():
    errors = validate_term(LinearDragTerm(mu=0.1))
    assert errors == []
