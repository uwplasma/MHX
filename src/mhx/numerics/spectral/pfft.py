"""Distributed real-to-complex 3D transforms with a slab decomposition.

``pfft3`` and ``pifft3`` follow the ``jnp.fft.rfftn``/``irfftn`` convention
on the last three axes. Without a mesh, they fall through to ``jnp.fft``
unchanged. With a device mesh, the physical field is sharded along its
first spatial axis and the spectral field along its second spatial axis:
the two local transforms surround one ``all_to_all`` transpose, so no
field-sized gather ever occurs. Reverse-mode differentiation stays
distributed because every piece is a local transform or an ``all_to_all``,
both of which JAX transposes shard-by-shard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array


def spectral_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the half-spectrum shape for a real field of ``shape``."""
    return (shape[0], shape[1], shape[2] // 2 + 1)


def _forward_local(block: Array) -> Array:
    """Transform the two unsharded axes of one x-slab."""
    partial = jnp.fft.rfft(block, axis=-1)
    return jnp.fft.fft(partial, axis=-2)


def _forward_finish(block: Array) -> Array:
    """Transform the x axis after the transpose."""
    return jnp.fft.fft(block, axis=-3)


def pfft3(field: Array, *, mesh: Mesh | None = None) -> Array:
    """Real-to-complex 3D transform over the last three axes.

    Args:
        field: Real array whose last three axes are the periodic directions.
            With a mesh, axis ``-3`` must divide by the mesh size.
        mesh: One-axis device mesh, or ``None`` for a single device.

    Returns:
        Complex half-spectrum array. With a mesh, the result is sharded
        along axis ``-2``.
    """
    if mesh is None or mesh.size == 1:
        return jnp.fft.rfftn(field, axes=(-3, -2, -1))

    axis = mesh.axis_names[0]
    batch = (None,) * (field.ndim - 3)

    def transform(block: Array) -> Array:
        partial = _forward_local(block)
        transposed = jax.lax.all_to_all(
            partial,
            axis,
            split_axis=field.ndim - 2,
            concat_axis=field.ndim - 3,
            tiled=True,
        )
        return _forward_finish(transposed)

    return jax.shard_map(
        transform,
        mesh=mesh,
        in_specs=PartitionSpec(*batch, axis, None, None),
        out_specs=PartitionSpec(*batch, None, axis, None),
    )(field)


def pifft3(
    field_hat: Array,
    *,
    shape: tuple[int, int, int],
    mesh: Mesh | None = None,
) -> Array:
    """Complex-to-real inverse of :func:`pfft3`.

    Args:
        field_hat: Half-spectrum array from :func:`pfft3`.
        shape: Global real-space shape of the last three axes.
        mesh: The same mesh handed to :func:`pfft3`, or ``None``.

    Returns:
        Real array with the last three axes equal to ``shape``. With a
        mesh, the result is sharded along axis ``-3``.
    """
    if mesh is None or mesh.size == 1:
        return jnp.fft.irfftn(field_hat, s=shape, axes=(-3, -2, -1))

    axis = mesh.axis_names[0]
    batch = (None,) * (field_hat.ndim - 3)

    def transform(block: Array) -> Array:
        undone = jnp.fft.ifft(block, axis=-3)
        transposed = jax.lax.all_to_all(
            undone,
            axis,
            split_axis=field_hat.ndim - 3,
            concat_axis=field_hat.ndim - 2,
            tiled=True,
        )
        partial = jnp.fft.ifft(transposed, axis=-2)
        return jnp.fft.irfft(partial, n=shape[2], axis=-1)

    return jax.shard_map(
        transform,
        mesh=mesh,
        in_specs=PartitionSpec(*batch, None, axis, None),
        out_specs=PartitionSpec(*batch, axis, None, None),
    )(field_hat)


def shard_physical(field: Array, mesh: Mesh | None) -> Array:
    """Place a real field on the mesh, sharded along axis ``-3``."""
    if mesh is None or mesh.size == 1:
        return field
    batch = (None,) * (field.ndim - 3)
    spec = PartitionSpec(*batch, mesh.axis_names[0], None, None)
    return jax.device_put(field, NamedSharding(mesh, spec))


def shard_spectral(field_hat: Array, mesh: Mesh | None) -> Array:
    """Place a spectral field on the mesh, sharded along axis ``-2``."""
    if mesh is None or mesh.size == 1:
        return field_hat
    batch = (None,) * (field_hat.ndim - 3)
    spec = PartitionSpec(*batch, None, mesh.axis_names[0], None)
    return jax.device_put(field_hat, NamedSharding(mesh, spec))
