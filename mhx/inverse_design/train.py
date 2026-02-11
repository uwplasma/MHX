from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from mhx.config import Objective, objective_preset, dump_config_yaml, ModelConfig
from mhx.io.paths import RunPaths, create_run_dir
from mhx.io.npz import savez
from mhx.solver.plugins import build_terms, PhysicsTerm
from mhx.solver.tearing import (
    _run_tearing_simulation_and_diagnostics,
    TearingMetrics,
)


@dataclass
class InverseDesignConfig:
    # Grid and box
    Nx: int = 64
    Ny: int = 64
    Nz: int = 1
    Lx: float = 2.0 * math.pi
    Ly: float = 2.0 * math.pi
    Lz: float = 2.0 * math.pi

    # Fixed physical parameters
    B0: float = 1.0
    B_g: float = 0.2
    a: float = 0.25
    eps_B: float = 1e-3

    # Time integration
    t0: float = 0.0
    t1: float = 60.0
    n_frames: int = 150
    dt0: float = 5e-4
    progress: bool = True
    jit: bool = False
    check_finite: bool = True

    # Equilibrium
    equilibrium_mode: str = "forcefree"

    # Objective
    objective: Objective = field(default_factory=lambda: objective_preset("forcefree"))

    # Physics model / plugin configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Bounds for eta and nu (log10-space)
    log10_eta_min: float = -4.5
    log10_eta_max: float = -2.0
    log10_nu_min: float = -4.5
    log10_nu_max: float = -2.0

    # Neural network + training hyperparameters
    latent_dim: int = 1
    hidden_width: int = 32
    hidden_depth: int = 2
    learning_rate: float = 1e-3
    n_train_steps: int = 25
    print_every: int = 1

    # Latent design value to train at (scalar)
    z_train: float = 0.0

    # Random seed
    seed: int = 1234

    @classmethod
    def default(cls, eq_mode: str = "forcefree") -> "InverseDesignConfig":
        return cls(equilibrium_mode=eq_mode, objective=objective_preset(eq_mode))

    @classmethod
    def fast(cls, eq_mode: str = "forcefree") -> "InverseDesignConfig":
        cfg = cls(equilibrium_mode=eq_mode, objective=objective_preset(eq_mode))
        cfg.Nx = 16
        cfg.Ny = 16
        cfg.Nz = 1
        cfg.t1 = 0.5
        cfg.n_frames = 6
        cfg.dt0 = 5e-4
        cfg.progress = False
        cfg.jit = False
        cfg.check_finite = True
        cfg.hidden_width = 8
        cfg.hidden_depth = 1
        cfg.n_train_steps = 2
        return cfg


class DesignMLP(eqx.Module):
    """MLP mapping latent design z -> (log10_eta, log10_nu)."""

    layers: List[eqx.nn.Linear]
    activation: Any = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_width: int, hidden_depth: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, hidden_depth + 1)
        layers: List[eqx.nn.Linear] = []
        layers.append(eqx.nn.Linear(in_dim, hidden_width, key=keys[0]))
        for i in range(hidden_depth - 1):
            layers.append(eqx.nn.Linear(hidden_width, hidden_width, key=keys[i + 1]))
        layers.append(eqx.nn.Linear(hidden_width, 2, key=keys[-1]))
        self.layers = layers
        self.activation = jax.nn.tanh

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(z, dtype=jnp.float64)
        if x.ndim == 0:
            x = jnp.expand_dims(x, 0)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


def _squash_to_interval(raw: jnp.ndarray, xmin: float, xmax: float) -> jnp.ndarray:
    center = 0.5 * (xmin + xmax)
    half_width = 0.5 * (xmax - xmin)
    return center + half_width * jnp.tanh(raw)


def _simulate_metrics(
    eta: jnp.ndarray,
    nu: jnp.ndarray,
    cfg: InverseDesignConfig,
    terms: list[PhysicsTerm] | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    if terms is None and cfg.model is not None:
        terms = build_terms(cfg.model.rhs_terms, cfg.model.term_params)
    eq_mode = cfg.model.equilibrium_mode or cfg.equilibrium_mode
    res = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        Lx=cfg.Lx,
        Ly=cfg.Ly,
        Lz=cfg.Lz,
        nu=nu,
        eta=eta,
        B0=cfg.B0,
        a=cfg.a,
        B_g=cfg.B_g,
        eps_B=cfg.eps_B,
        t0=cfg.t0,
        t1=cfg.t1,
        n_frames=cfg.n_frames,
        dt0=cfg.dt0,
        equilibrium_mode=eq_mode,
        terms=terms,
        progress=cfg.progress,
        jit=cfg.jit,
        check_finite=cfg.check_finite,
    )

    metrics = TearingMetrics.from_result(res)
    f_kin = jnp.asarray(metrics.f_kin)
    complexity = jnp.asarray(metrics.complexity)
    gamma_fit = jnp.asarray(metrics.gamma_fit)
    return f_kin, complexity, gamma_fit, res


def make_loss_fn(cfg: InverseDesignConfig, terms: list[PhysicsTerm] | None):
    target = jnp.array([cfg.objective.target_f_kin, cfg.objective.target_complexity], dtype=jnp.float64)
    z_train = jnp.array(cfg.z_train, dtype=jnp.float64)

    def loss_fn(model: DesignMLP, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        raw_log10_eta, raw_log10_nu = model(z_train)
        log10_eta = _squash_to_interval(raw_log10_eta, cfg.log10_eta_min, cfg.log10_eta_max)
        log10_nu = _squash_to_interval(raw_log10_nu, cfg.log10_nu_min, cfg.log10_nu_max)

        eta = 10.0 ** log10_eta
        nu = 10.0 ** log10_nu

        f_kin, complexity, gamma_fit, res = _simulate_metrics(eta, nu, cfg, terms=terms)
        _ = (gamma_fit, res)
        diff_f = f_kin - target[0]
        diff_c = complexity - target[1]
        loss = diff_f**2 + cfg.objective.lambda_complexity * diff_c**2

        aux = {
            "log10_eta": log10_eta,
            "log10_nu": log10_nu,
            "eta": eta,
            "nu": nu,
            "f_kin": f_kin,
            "complexity": complexity,
        }
        return loss, aux

    return loss_fn


def build_training_step(
    cfg: InverseDesignConfig,
    optimizer: optax.GradientTransformation,
    static_model: DesignMLP,
    terms: list[PhysicsTerm] | None,
):
    loss_fn = make_loss_fn(cfg, terms)

    def loss_from_params(params, key):
        model = eqx.combine(params, static_model)
        loss, aux = loss_fn(model, key)
        return loss, aux

    grad_fn = jax.value_and_grad(loss_from_params, argnums=0, has_aux=True)

    @jax.jit
    def step(params, opt_state, key):
        (loss_val, aux), grads = grad_fn(params, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux

    return step


def run_inverse_design(
    cfg: InverseDesignConfig,
    *,
    run_paths: RunPaths | None = None,
) -> Tuple[RunPaths, Dict[str, List[float]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if run_paths is None:
        run_paths = create_run_dir(tag=f"inverse_{cfg.equilibrium_mode}")

    config_payload = {"sim": {
        "Nx": cfg.Nx, "Ny": cfg.Ny, "Nz": cfg.Nz,
        "Lx": cfg.Lx, "Ly": cfg.Ly, "Lz": cfg.Lz,
                "B0": cfg.B0, "a": cfg.a, "B_g": cfg.B_g, "eps_B": cfg.eps_B,
        "t0": cfg.t0, "t1": cfg.t1, "n_frames": cfg.n_frames, "dt0": cfg.dt0,
        "progress": cfg.progress, "jit": cfg.jit, "check_finite": cfg.check_finite,
        "equilibrium_mode": cfg.equilibrium_mode,
    }, "objective": cfg.objective.as_dict(), "training": {
        "latent_dim": cfg.latent_dim, "hidden_width": cfg.hidden_width, "hidden_depth": cfg.hidden_depth,
        "learning_rate": cfg.learning_rate, "n_train_steps": cfg.n_train_steps, "seed": cfg.seed,
    }}
    if cfg.model is not None:
        config_payload["model"] = cfg.model.as_dict()
    dump_config_yaml(run_paths.config_yaml, config_payload)

    # Initialize MLP and optimizer
    key = jax.random.PRNGKey(cfg.seed)
    key_model, key_train = jax.random.split(key)

    model = DesignMLP(
        in_dim=cfg.latent_dim,
        hidden_width=cfg.hidden_width,
        hidden_depth=cfg.hidden_depth,
        key=key_model,
    )

    params, static_model = eqx.partition(model, eqx.is_array)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.learning_rate))
    opt_state = optimizer.init(params)
    terms = build_terms(cfg.model.rhs_terms, cfg.model.term_params) if cfg.model is not None else None
    training_step = build_training_step(cfg, optimizer, static_model, terms)

    # Baseline simulation
    log10_eta0 = 0.5 * (cfg.log10_eta_min + cfg.log10_eta_max)
    log10_nu0 = 0.5 * (cfg.log10_nu_min + cfg.log10_nu_max)
    eta0 = 10.0**log10_eta0
    nu0 = 10.0**log10_nu0
    _, _, _, res_init = _simulate_metrics(
        jnp.array(eta0, dtype=jnp.float64),
        jnp.array(nu0, dtype=jnp.float64),
        cfg,
        terms=terms,
    )
    savez(run_paths.solution_initial_npz, res_init)

    history: Dict[str, List[float]] = {
        "loss": [],
        "log10_eta": [],
        "log10_nu": [],
        "eta": [],
        "nu": [],
        "f_kin": [],
        "complexity": [],
        "target_f_kin": [float(cfg.objective.target_f_kin)],
        "target_complexity": [float(cfg.objective.target_complexity)],
        "lambda_complexity": [float(cfg.objective.lambda_complexity)],
    }

    best_loss = float("inf")
    best_params = params
    best_aux = None

    for step in range(cfg.n_train_steps):
        key_train, key_step = jax.random.split(key_train)
        params, opt_state, loss_val, aux = training_step(params, opt_state, key_step)

        loss_float = float(loss_val)
        history["loss"].append(loss_float)
        history["log10_eta"].append(float(aux["log10_eta"]))
        history["log10_nu"].append(float(aux["log10_nu"]))
        history["eta"].append(float(aux["eta"]))
        history["nu"].append(float(aux["nu"]))
        history["f_kin"].append(float(aux["f_kin"]))
        history["complexity"].append(float(aux["complexity"]))

        if loss_float < best_loss:
            best_loss = loss_float
            best_params = params
            best_aux = aux

    assert best_aux is not None
    params = best_params

    # Mid-training run
    mid_index = len(history["eta"]) // 2
    eta_mid = history["eta"][mid_index]
    nu_mid = history["nu"][mid_index]
    _, _, _, res_mid = _simulate_metrics(
        jnp.array(eta_mid, dtype=jnp.float64),
        jnp.array(nu_mid, dtype=jnp.float64),
        cfg,
        terms=terms,
    )
    savez(run_paths.solution_mid_npz, res_mid)

    # Final run
    eta_final = float(best_aux["eta"])
    nu_final = float(best_aux["nu"])
    _, _, _, res_final = _simulate_metrics(
        jnp.array(eta_final, dtype=jnp.float64),
        jnp.array(nu_final, dtype=jnp.float64),
        cfg,
        terms=terms,
    )
    savez(run_paths.solution_final_npz, res_final)

    # Save history
    savez(run_paths.history_npz, history)

    return run_paths, history, res_init, res_mid, res_final
