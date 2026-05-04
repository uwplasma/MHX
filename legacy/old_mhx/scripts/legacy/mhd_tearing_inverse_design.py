#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper for inverse-design training.

This script now delegates to `mhx.inverse_design.train` and preserves
CLI usage while keeping the full training logic in the package.
"""

from __future__ import annotations

import os
import dataclasses

from mhx.config import InverseDesignConfig
from mhx.inverse_design.train import run_inverse_design


def main() -> None:
    cfg = InverseDesignConfig.default()

    # Optional overrides via environment (used by CLI wrapper)
    eq_env = os.environ.get("MHX_ID_EQ_MODE")
    steps_env = os.environ.get("MHX_ID_STEPS")
    fast_env = os.environ.get("MHX_ID_FAST")
    if eq_env:
        cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, equilibrium_mode=eq_env))
    if fast_env == "1":
        cfg = InverseDesignConfig.fast(cfg.sim.equilibrium_mode)
    if steps_env:
        cfg = dataclasses.replace(cfg, n_train_steps=int(steps_env))

    run_inverse_design(cfg)


if __name__ == "__main__":
    main()
