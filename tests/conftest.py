"""Shared pytest configuration."""

from __future__ import annotations

import os

# Four logical CPU devices so sharded code paths run, and count toward
# coverage, inside the main test process. Must be set before JAX loads.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax

jax.config.update("jax_enable_x64", True)
