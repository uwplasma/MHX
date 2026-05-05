"""Runtime configuration helpers."""

from __future__ import annotations


def configure_jax(*, enable_x64: bool | None = None) -> bool:
    """Configure JAX process-wide precision and return the active x64 flag."""
    import jax

    if enable_x64 is not None:
        jax.config.update("jax_enable_x64", bool(enable_x64))
    return bool(jax.config.jax_enable_x64)
