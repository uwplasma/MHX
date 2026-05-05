"""MHX: differentiable JAX tools for reconnection and magnetohydrodynamics."""

from mhx._version import __version__
from mhx.versioning import MHX_PUBLIC_API_VERSION, api_version_info

__all__ = ["MHX_PUBLIC_API_VERSION", "__version__", "api_version_info"]
