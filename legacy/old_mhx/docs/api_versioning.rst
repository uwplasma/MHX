API Versioning
==============

MHX exposes explicit version identifiers for stable public interfaces:

- ``TearingSimConfig.API_VERSION``
- ``InverseDesignConfig.API_VERSION``
- ``PhysicsAPI.version``
- ``NPZ_SCHEMA_VERSION``

These are defined in ``mhx/version.py`` and reflected in docs.

Compatibility policy
--------------------

- ``v1`` interfaces are intended to remain stable across minor releases.
- Breaking changes bump the corresponding API version and are called out in
  ``CHANGELOG.md``.
- All run artifacts include ``schema_version`` and ``mhx_version`` for traceability.

Enforcing versions
------------------

Set ``MHX_API_VERSION`` to enforce expected versions when loading configs/NPZs.
Examples:

- ``MHX_API_VERSION=1`` (apply to all)
- ``MHX_API_VERSION=sim=1,inverse=1,physics=1,npz=1``

When ``MHX_API_VERSION`` is set, mismatches raise an error; otherwise they emit
a warning.
