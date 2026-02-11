from __future__ import annotations

from typing import Dict, Any

import jax
import jax.numpy as jnp
import optax
import diffrax as dfx

Array = jnp.ndarray


def init_mlp(in_dim: int, hidden_dim: int, out_dim: int, key: jax.random.PRNGKey) -> Dict[str, Array]:
    k1, k2 = jax.random.split(key, 2)
    W1 = jax.random.normal(k1, (in_dim, hidden_dim)) * 0.1
    b1 = jnp.zeros((hidden_dim,))
    W2 = jax.random.normal(k2, (hidden_dim, out_dim)) * 0.1
    b2 = jnp.zeros((out_dim,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def mlp_apply(params: Dict[str, Array], x: Array) -> Array:
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def fit_latent_ode(
    ts: Array,
    y: Array,
    *,
    latent_dim: int = 2,
    steps: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit a simple latent ODE to time series y(t)."""
    key = jax.random.PRNGKey(seed)
    key_z, key_latent, key_dec = jax.random.split(key, 3)

    z0 = jax.random.normal(key_z, (latent_dim,))
    latent_params = init_mlp(latent_dim, 32, latent_dim, key_latent)
    decoder_params = init_mlp(latent_dim, 32, y.shape[-1], key_dec)

    params = {"latent": latent_params, "decoder": decoder_params, "z0": z0}
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def latent_rhs(t, z, args):
        _ = t
        return mlp_apply(args["latent"], z)

    def decode(z, decoder):
        return mlp_apply(decoder, z)

    def solve(z0, latent):
        term = dfx.ODETerm(latent_rhs)
        solver = dfx.Dopri5()
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=z0,
            args={"latent": latent},
            saveat=dfx.SaveAt(ts=ts),
        )
        return sol.ys

    def loss_fn(params):
        zs = solve(params["z0"], params["latent"])
        y_hat = jax.vmap(lambda z: decode(z, params["decoder"]))(zs)
        return jnp.mean((y_hat - y) ** 2)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    history = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))

    return {"params": params, "history": history}
