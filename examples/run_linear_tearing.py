"""Minimal Python driver for the linear tearing smoke config.

Loads a TOML config, runs the linear tearing smoke benchmark and prints its
diagnostics. It takes seconds. Edit the config file (mesh, time, physics) to
scale up. Run from the repository root:

    python examples/run_linear_tearing.py
"""

import time

from mhx.benchmarks import run_linear_tearing_smoke
from mhx.config import load_config

# --- Input parameters ------------------------------------------------------
CONFIG = "examples/linear_tearing.toml"  # config to run, relative to the current directory

# --- Run -------------------------------------------------------------------
print(f"Running the linear tearing smoke benchmark from {CONFIG}", flush=True)
cfg = load_config(CONFIG)
start = time.perf_counter()
_, diagnostics = run_linear_tearing_smoke(cfg)
print(f"Run time (including compile): {time.perf_counter() - start:.2f} s")

# --- Results ---------------------------------------------------------------
print(diagnostics)
