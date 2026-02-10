#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility wrapper for inverse-design training.

This script now delegates to `mhx.inverse_design.train` and preserves
CLI usage while keeping the full training logic in the package.
"""

from __future__ import annotations

import os

from mhx.inverse_design.train import InverseDesignConfig, run_inverse_design


def main() -> None:
    cfg = InverseDesignConfig()

    # Optional overrides via environment (used by CLI wrapper)
    eq_env = os.environ.get("MHX_ID_EQ_MODE")
    steps_env = os.environ.get("MHX_ID_STEPS")
    fast_env = os.environ.get("MHX_ID_FAST")
    if eq_env:
        cfg.equilibrium_mode = eq_env
    if steps_env:
        cfg.n_train_steps = int(steps_env)
    if fast_env == "1":
        cfg = InverseDesignConfig.fast(cfg.equilibrium_mode)

    run_inverse_design(cfg)


if __name__ == "__main__":
    main()
