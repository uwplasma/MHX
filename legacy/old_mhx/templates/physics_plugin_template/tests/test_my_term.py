from __future__ import annotations

import jax.numpy as jnp

from templates.physics_plugin_template.my_term import MyTerm


def test_my_term_shapes():
    v_hat = jnp.zeros((3, 4, 4, 1))
    B_hat = jnp.zeros((3, 4, 4, 1))
    kx = jnp.zeros((4, 4, 1))
    ky = jnp.zeros((4, 4, 1))
    kz = jnp.zeros((4, 4, 1))
    k2 = jnp.ones((4, 4, 1))
    mask = jnp.ones((4, 4, 1), dtype=bool)

    term = MyTerm()
    dv, dB = term.rhs_additions(
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
