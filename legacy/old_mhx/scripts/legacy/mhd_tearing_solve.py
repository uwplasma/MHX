#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility shim.

The core tearing solver has moved to `mhx.solver.tearing` as part of the
package refactor. This module re-exports the public API so existing research
scripts that import `mhd_tearing_solve` keep working.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from mhx.solver import *  # noqa: F401,F403


if __name__ == "__main__":
    from mhx.solver.tearing import main

    main()
