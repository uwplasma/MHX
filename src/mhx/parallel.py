"""Device sharding for periodic reduced-MHD fields.

MHX splits the first grid axis across JAX devices. JAX then partitions the
compiled Fourier operators and time integrator with its SPMD compiler.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mhx.state import ReducedMHDState


@dataclass(frozen=True)
class SpatialSharding:
    """A one-dimensional device mesh for two-dimensional fields.

    Attributes:
        devices: JAX devices used by the mesh.
        mesh: Named JAX device mesh.
        fields: Sharding rule for arrays shaped ``(nx, ny)``.
    """

    devices: tuple[jax.Device, ...]
    mesh: Mesh
    fields: NamedSharding

    @property
    def device_count(self) -> int:
        """Return the number of devices in this mesh."""
        return len(self.devices)


def make_spatial_sharding(
    shape: tuple[int, int],
    device_count: int,
    *,
    platform: str | None = None,
) -> SpatialSharding:
    """Split the first grid axis across local JAX devices.

    Args:
        shape: Global two-dimensional field shape.
        device_count: Number of local devices to use.
        platform: Optional JAX platform name, such as ``"cpu"`` or ``"gpu"``.

    Returns:
        A sharding plan for all reduced-MHD fields.

    Raises:
        ValueError: If the device count is invalid or cannot divide ``shape[0]``.
    """
    if device_count < 1:
        raise ValueError("device_count must be at least 1")
    if shape[0] % device_count:
        raise ValueError(
            f"shape[0] must be divisible by device_count; got {shape[0]} and {device_count}"
        )
    devices = tuple(jax.devices(platform))
    if device_count > len(devices):
        raise ValueError(
            f"requested {device_count} {platform or 'local'} devices, but JAX found {len(devices)}"
        )
    selected = devices[:device_count]
    mesh = Mesh(np.asarray(selected), ("device",))
    fields = NamedSharding(mesh, PartitionSpec("device", None))
    return SpatialSharding(devices=selected, mesh=mesh, fields=fields)


def shard_state(
    state: ReducedMHDState,
    sharding: SpatialSharding,
) -> ReducedMHDState:
    """Place both reduced-MHD fields on a spatial device mesh."""
    return jax.tree.map(lambda field: jax.device_put(field, sharding.fields), state)


def available_devices(platform: str | None = None) -> tuple[jax.Device, ...]:
    """Return local JAX devices for a platform."""
    return tuple(jax.devices(platform))
