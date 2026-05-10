# API compatibility policy

The rebuilt MHX package separates three versioned surfaces:

| Surface | Current value | Stability intent |
| --- | --- | --- |
| Package version | `0.1.0a0` | Changes with every release. |
| Public API version | `v1` | Compatibility contract for config loaders, plugin interfaces, and artifact readers. |
| Artifact schemas | `mhx.*.v1` | File-format contracts for generated outputs. |
| Claim levels | `smoke`, `validation`, `production_template`, `production` | Reviewer-facing boundary for what an artifact can support. |

Inspect the active values:

```bash
mhx api status
mhx api status --json
```

## Reproducibility override

Set `MHX_API_VERSION` to force loaders and writers to validate the requested
public API before doing work:

```bash
MHX_API_VERSION=v1 mhx validate all --outdir outputs/validation_suite
```

Any unsupported value fails early. This is intentionally strict: a reviewer or
workflow runner should not silently read an artifact produced under an
unrecognized API contract.

## Stable v1 interfaces

The following names are part of the rebuilt v1 public surface:

- `mhx.config.RunConfig`, `MeshConfig`, `TimeConfig`, `PhysicsConfig`,
  `NumericsConfig`, and `DiagnosticsConfig`.
- `mhx.physics.PhysicsTerm`, `PhysicsRegistry`,
  `PHYSICS_API_VERSION = "mhx.physics.v1"`, and
  `PHYSICS_ENTRY_POINT_GROUP = "mhx.physics"`.
- `mhx.diagnostics.DiagnosticSpec`, `DiagnosticsRegistry`, and
  `DIAGNOSTICS_ENTRY_POINT_GROUP = "mhx.diagnostics"`.
- `mhx.io.read_reduced_mhd_trajectory_npz` and
  `mhx.io.write_reduced_mhd_trajectory_npz` for
  `mhx.reduced_mhd.trajectory.v1`.
- Manifest `claim_level` values: `unspecified`, `smoke`, `validation`,
  `production_template`, and `production`.
- `mhx validate all`, `mhx benchmark ...`, `mhx figures`, `mhx report`, and
  `mhx artifact-manifest` command families.

## Compatibility rules

- Patch releases may add optional fields to JSON/NPZ metadata but must keep
  existing v1 keys readable.
- Minor pre-releases may add new diagnostics, benchmarks, plugins, and CLI
  options.
- Breaking changes require either a new public API version or a documented
  deprecation window.
- Active source files must not import archived legacy modules. The CI command
  `python tools/check_legacy_imports.py` enforces this.

## Source links

- [Versioning helpers](https://github.com/uwplasma/MHX/blob/main/src/mhx/versioning.py)
- [Trajectory schema loader](https://github.com/uwplasma/MHX/blob/main/src/mhx/io/trajectory.py)
- [Legacy import guard](https://github.com/uwplasma/MHX/blob/main/tools/check_legacy_imports.py)
