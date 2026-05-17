"""MHX: differentiable JAX tools for reconnection and magnetohydrodynamics."""

from pathlib import Path

from mhx._version import __version__
from mhx.config import RunConfig, load_config
from mhx.versioning import MHX_PUBLIC_API_VERSION, api_version_info


def run(config: str | Path, *, outdir: str | Path | None = None) -> Path:
    """Run a v1 reduced-MHD TOML configuration and return ``manifest.json``."""
    from mhx.cli.main import _run_config

    return _run_config(
        Path(config),
        outdir=None if outdir is None else Path(outdir),
    )


__all__ = [
    "MHX_PUBLIC_API_VERSION",
    "RunConfig",
    "__version__",
    "api_version_info",
    "load_config",
    "run",
]
