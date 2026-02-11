from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from mhx.solver.plugins import API_VERSION

Array = jnp.ndarray


@dataclass(frozen=True)
class MyTerm:
    name: str = "my_term"
    api_version: str = API_VERSION

    def rhs_additions(
        self,
        *,
        t: float,
        v_hat: Array,
        B_hat: Array,
        kx: Array,
        ky: Array,
        kz: Array,
        k2: Array,
        mask_dealias: Array,
    ) -> Tuple[Array, Array]:
        _ = (t, kx, ky, kz, k2, mask_dealias)
        dv = jnp.zeros_like(v_hat)
        dB = jnp.zeros_like(B_hat)
        return dv, dB
