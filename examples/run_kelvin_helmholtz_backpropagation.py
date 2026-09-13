"""Backpropagation through a FAST Kelvin--Helmholtz simulation.

Differentiates a small smooth Kelvin--Helmholtz passive-dye run with respect
to the perturbation amplitude.  Demonstrates reverse-mode gradients,
forward-mode JVPs, a finite-difference check, and a tiny optimization loop.

The default grid/time choices are intentionally small so this can run on a
laptop.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.kelvin_helmholtz import (
    KelvinHelmholtzConfig,
    kelvin_helmholtz_entropy_jvp,
    kelvin_helmholtz_entropy_objective,
    kelvin_helmholtz_entropy_value_and_grad,
)
from mhx.runtime import configure_jax


def main() -> None:
    configure_jax(enable_x64=True)

    output_root = Path(os.environ.get("MHX_EXAMPLE_OUTDIR_ROOT", "outputs/examples"))
    outdir = output_root / "kelvin_helmholtz_backpropagation"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {outdir.resolve()}")

    # Differentiable objective: final passive-dye entropy after a very short
    # KH run.  Reverse mode is appropriate because the output is scalar; JVP
    # is useful for cheap single-direction sensitivity checks.
    config = KelvinHelmholtzConfig(
        shape=(16, 32),
        dt=1.0e-3,
        t_end=8.0e-3,
        save_every=8,
        viscosity=1.0e-3,
    )

    amplitude0 = 1.0e-2
    value, gradient = kelvin_helmholtz_entropy_value_and_grad(amplitude0, config)
    jvp_value, jvp_tangent = kelvin_helmholtz_entropy_jvp(amplitude0, 1.0e-3, config)

    print("final entropy:", float(value))
    print("reverse-mode dS/dA:", float(gradient))
    print("JVP value:", float(jvp_value))
    print("JVP tangent for dA=1e-3:", float(jvp_tangent))
    print("JVP/grad consistency:", float(jvp_tangent / (gradient * 1.0e-3)))

    # Finite-difference check.  Sanity check for the tutorial-scale run;
    # not a substitute for a full adjoint verification suite at production
    # resolution.
    epsilon = 1.0e-4
    f_plus = kelvin_helmholtz_entropy_objective(jnp.asarray(amplitude0 + epsilon), config)
    f_minus = kelvin_helmholtz_entropy_objective(jnp.asarray(amplitude0 - epsilon), config)
    finite_difference = (f_plus - f_minus) / (2.0 * epsilon)

    print("finite-difference dS/dA:", float(finite_difference))
    relative_error = abs(finite_difference - gradient) / max(abs(float(gradient)), 1.0e-14)
    print("relative error:", float(relative_error))

    # Tiny optimization loop.  Optimize the perturbation amplitude toward a
    # target final entropy.  Pedagogical; the target is chosen close to the
    # FAST run so the loop remains numerically tame.
    target_entropy = jnp.asarray(float(value) + 1.0e-7)
    learning_rate = 5.0e5
    amplitude = jnp.asarray(amplitude0)
    history = []

    def loss(active_amplitude: jax.Array) -> jax.Array:
        prediction = kelvin_helmholtz_entropy_objective(active_amplitude, config)
        return (prediction - target_entropy) ** 2

    for step in range(8):
        loss_value, loss_gradient = jax.value_and_grad(loss)(amplitude)
        entropy_value = kelvin_helmholtz_entropy_objective(amplitude, config)
        history.append(
            (step, float(amplitude), float(entropy_value), float(loss_value), float(loss_gradient))
        )
        amplitude = amplitude - learning_rate * loss_gradient

    for row in history:
        print(
            f"step={row[0]:02d} amplitude={row[1]:.8f} "
            f"entropy={row[2]:.12f} loss={row[3]:.3e} grad={row[4]:.3e}"
        )

    # Plot optimization history.
    history_array = np.asarray(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    axes[0].plot(history_array[:, 0], history_array[:, 1], marker="o")
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("perturbation amplitude")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(history_array[:, 0], history_array[:, 3], marker="o")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.3)
    fig.savefig(outdir / "kh_backpropagation_history.png", dpi=180)
    plt.close(fig)
    print("wrote", outdir / "kh_backpropagation_history.png")


if __name__ == "__main__":
    main()
