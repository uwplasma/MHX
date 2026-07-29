"""Device sharding for periodic reduced-MHD fields.

MHX splits the first grid axis across JAX devices. JAX then partitions the
compiled Fourier operators and time integrator with its SPMD compiler.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import jax
import numpy as np
import solvax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mhx.state import ReducedMHDState

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


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


def make_device_mesh(
    device_count: int,
    *,
    platform: str | None = None,
) -> Mesh:
    """Return a one-dimensional mesh over the requested JAX devices."""
    if device_count < 1:
        raise ValueError("device_count must be at least 1")
    devices = tuple(jax.devices(platform))
    if device_count > len(devices):
        raise ValueError(
            f"requested {device_count} {platform or 'JAX'} devices, but JAX found {len(devices)}"
        )
    return Mesh(np.asarray(devices[:device_count]), ("device",))


def make_spatial_sharding(
    shape: tuple[int, int],
    device_count: int,
    *,
    platform: str | None = None,
) -> SpatialSharding:
    """Split the first grid axis across JAX devices.

    Args:
        shape: Global two-dimensional field shape.
        device_count: Number of devices to use.
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
    mesh = make_device_mesh(device_count, platform=platform)
    selected = tuple(mesh.devices.flat)
    fields = NamedSharding(mesh, PartitionSpec("device", None))
    return SpatialSharding(devices=selected, mesh=mesh, fields=fields)


def shard_state(
    state: ReducedMHDState,
    sharding: SpatialSharding,
) -> ReducedMHDState:
    """Place both reduced-MHD fields on a spatial device mesh."""
    return jax.tree.map(lambda field: jax.device_put(field, sharding.fields), state)


def available_devices(platform: str | None = None) -> tuple[jax.Device, ...]:
    """Return JAX devices, including remote devices after initialization."""
    return tuple(jax.devices(platform))


def initialize_distributed(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | None = None,
) -> None:
    """Connect JAX processes before any code queries devices.

    Slurm, Open MPI, Kubernetes, and supported TPU environments can usually
    call this function without arguments. A manual launch must give the same
    coordinator address and process count to every process, plus its distinct
    process ID.
    """
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )


def shard_batch(
    local_function: Callable[[InputT], OutputT],
    *,
    mesh: Mesh,
    input_rank: int,
    output_rank: int,
    output_batch_axis: int = 0,
) -> Callable[[InputT], OutputT]:
    """Map an independent local batch explicitly over a device mesh.

    SOLVAX owns this numerical parallelism. The short fallback keeps MHX
    compatible with the current SOLVAX release while the same helper proceeds
    through SOLVAX review.
    """
    solvax_shard_batch = getattr(solvax, "shard_batch", None)
    if solvax_shard_batch is not None:
        return solvax_shard_batch(
            local_function,
            mesh=mesh,
            input_rank=input_rank,
            output_rank=output_rank,
            output_batch_axis=output_batch_axis,
        )

    input_spec = PartitionSpec("device", *([None] * (input_rank - 1)))
    output_axes = [None] * output_rank
    output_axes[output_batch_axis % output_rank] = "device"
    return jax.shard_map(
        local_function,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=PartitionSpec(*output_axes),
    )
