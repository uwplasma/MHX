"""Versioned public API and artifact-schema helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from mhx._version import __version__

MHX_PUBLIC_API_VERSION = "v1"
MHX_API_VERSION_ENV = "MHX_API_VERSION"
SUPPORTED_API_VERSIONS = (MHX_PUBLIC_API_VERSION,)

PHYSICS_API_VERSION = "mhx.physics.v1"
DIAGNOSTICS_API_VERSION = "mhx.diagnostics.v1"

REDUCED_MHD_TRAJECTORY_SCHEMA = "mhx.reduced_mhd.trajectory.v1"
MANIFEST_SCHEMA = "mhx.manifest.v1"
ARTIFACT_MANIFEST_SCHEMA = "mhx.artifacts.v1"
VALIDATION_SUITE_SCHEMA = "mhx.validation.suite.v1"


@dataclass(frozen=True)
class APIVersionInfo:
    """Serializable summary of the active MHX public API surface."""

    package_version: str
    public_api_version: str
    supported_api_versions: tuple[str, ...]
    physics_api_version: str
    diagnostics_api_version: str
    trajectory_schema: str
    manifest_schema: str
    artifact_manifest_schema: str
    validation_suite_schema: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible API metadata."""
        return {
            "package_version": self.package_version,
            "public_api_version": self.public_api_version,
            "supported_api_versions": list(self.supported_api_versions),
            "physics_api_version": self.physics_api_version,
            "diagnostics_api_version": self.diagnostics_api_version,
            "trajectory_schema": self.trajectory_schema,
            "manifest_schema": self.manifest_schema,
            "artifact_manifest_schema": self.artifact_manifest_schema,
            "validation_suite_schema": self.validation_suite_schema,
        }


def active_api_version() -> str:
    """Return the requested public API version.

    ``MHX_API_VERSION`` is an explicit reproducibility override. It currently
    accepts only ``v1``; setting any other value fails early instead of silently
    reading or writing artifacts with incompatible assumptions.
    """
    return os.environ.get(MHX_API_VERSION_ENV, MHX_PUBLIC_API_VERSION)


def require_supported_api_version(version: str | None = None, *, context: str = "MHX") -> str:
    """Validate and return a supported public API version."""
    requested = active_api_version() if version is None else version
    if requested not in SUPPORTED_API_VERSIONS:
        supported = ", ".join(SUPPORTED_API_VERSIONS)
        raise ValueError(
            f"{context} requires unsupported API version {requested!r}; "
            f"supported versions: {supported}"
        )
    return requested


def api_version_info() -> APIVersionInfo:
    """Return the active package/API/schema compatibility summary."""
    return APIVersionInfo(
        package_version=__version__,
        public_api_version=require_supported_api_version(context=MHX_API_VERSION_ENV),
        supported_api_versions=SUPPORTED_API_VERSIONS,
        physics_api_version=PHYSICS_API_VERSION,
        diagnostics_api_version=DIAGNOSTICS_API_VERSION,
        trajectory_schema=REDUCED_MHD_TRAJECTORY_SCHEMA,
        manifest_schema=MANIFEST_SCHEMA,
        artifact_manifest_schema=ARTIFACT_MANIFEST_SCHEMA,
        validation_suite_schema=VALIDATION_SUITE_SCHEMA,
    )
