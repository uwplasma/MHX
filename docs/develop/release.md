# Release readiness

This page records the practical release gates for making MHX public and
reviewer-auditable.

## Required local checks

```bash
python -m ruff check src tests examples tools
python tools/check_legacy_imports.py
python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95
sphinx-build -W -b html docs docs/_build/html
mhx validate all --outdir outputs/release/validation_suite
mhx validate readiness --suite outputs/release/validation_suite --outdir outputs/release/readiness
mhx validate release-candidate --outdir outputs/release/release_candidate \
  --readiness outputs/release/readiness --require-readiness
mhx validate paper-pipeline --outdir outputs/release/paper_pipeline
mhx benchmark catalog --outdir outputs/release/catalog
mhx artifact-manifest outputs/release
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Release artifacts

A release candidate should include:

- `validation_suite.json` and `validation_suite.md`;
- `readiness/readiness.json` and `release_candidate/release_candidate.json`;
- `paper_pipeline/paper_pipeline.json`, `paper_pipeline/validation.json`, and
  recursive artifact hashes;
- all validation figures under `validation_suite/*/figures/`;
- `artifact_manifest.json` with SHA-256 hashes;
- the exact commit SHA;
- the `MHX_API_VERSION` value used for the run;
- a changelog entry and citation metadata.

## Current maturity boundary

The current rebuilt repository is suitable for:

- demonstrating the package architecture;
- validating spectral operators, exact diffusion, matrix-free JVPs, and
- Harris-sheet outer-region $\Delta'$ matching, direct tearing eigenvalue
  gates, nonlinear energy budgets, Orszag--Tang/turbulence media, and
  neural-ODE FAST reproducibility workflows;
- showing how to add reduced-state physics and diagnostic plugins.

The current rebuilt repository should not yet be described as a calibrated
nonlinear plasmoid solver or as a production neural-ODE inverse-design tool.
Those claims require the next validation stages described in the audit and
roadmap.

## Fast static release gate

Use the static release-candidate gate after the dynamic validation suite has
produced a readiness report:

```bash
mhx validate release-candidate \
  --outdir outputs/release/release_candidate \
  --readiness outputs/release/readiness \
  --require-readiness
```

This command checks packaging metadata, CI workflow versions, ReadTheDocs
configuration, citation/version consistency, required docs and examples, active
legacy-import hygiene, README claim boundaries, and documentation figure hashes.
It deliberately does not promote nonlinear physics claims; production claims
remain controlled by the campaign promotion reports documented in
[campaign_runner.md](../project/campaign_runner.md).

## Source links

- [Release checklist](https://github.com/uwplasma/MHX/blob/main/RELEASE.md)
- [Release-candidate gate](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/release_candidate.py)
- [Changelog](https://github.com/uwplasma/MHX/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/uwplasma/MHX/blob/main/CITATION.cff)

## Migration from legacy scripts

The original exploratory scripts are preserved under `legacy/old_mhx/`. They are
not imported by the rebuilt package and are not part of the public API.

Use the active CLI instead:

| Legacy workflow | Active replacement |
| --- | --- |
| `run_MHD.py` or `run_MHD_box.py` | `mhx run examples/linear_tearing.toml --outdir outputs/smoke` |
| `mhd_tearing_solve.py` | `mhx benchmark run --config examples/linear_tearing.toml --outdir outputs/benchmarks/linear_tearing_fast` |
| `mhd_tearing_postprocess.py` | `mhx figures <run_dir> --gif` and `mhx report <run_dir>` |
| `mhd_linear_benchmarks.py` | `mhx benchmark decay`, `mhx benchmark linearized-rhs`, `mhx benchmark reduced-mhd-eigenmode` |
| `mhd_tearing_scan.py` | Roadmap: TOML-driven scan command after the v1 validation core is complete. |
| `mhd_tearing_inverse_design.py` | Roadmap: differentiable inverse-design command after calibrated tearing eigenvalue validation. |
| `mhd_tearing_ml.py` and `mhd_tearing_ml_v2.py` | `mhx neural-ode dataset --outdir outputs/neural_ode/seed_qi_fast` and `mhx neural-ode train --outdir outputs/neural_ode/latent_ode_fast` |

### Why the old scripts are archived

The old scripts were valuable exploratory tooling, but they mixed solver code,
plotting, hard-coded parameters, objective functions, and output paths. The new
package keeps these concerns separate:

- configs live in TOML and are saved as `config_effective.json`;
- diagnostics are registry entries with stable output keys;
- physics terms are versioned plugins;
- artifacts are schema-versioned and checksumed;
- validation commands have explicit pass/fail gates.

### Enforcement

Run the same check used in CI:

```bash
python tools/check_legacy_imports.py
```

This fails if active Python files import `legacy.old_mhx` or any archived
top-level script module such as `mhd_tearing_solve`.

## API compatibility policy

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

### Reproducibility override

Set `MHX_API_VERSION` to force loaders and writers to validate the requested
public API before doing work:

```bash
MHX_API_VERSION=v1 mhx validate all --outdir outputs/validation_suite
```

Any unsupported value fails early. This is intentionally strict: a reviewer or
workflow runner should not silently read an artifact produced under an
unrecognized API contract.

TOML configs may also include `api_version = "v1"`. `mhx.config.load_config`
rejects unsupported config API values, and also rejects an unsupported
`MHX_API_VERSION` override even when the config omits the field. Configs without
`api_version` are treated as v1 while v1 is the only supported public API.

### Stable v1 interfaces

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
- Public CLI families documented for v1: `mhx api`, `mhx campaign`,
  `mhx neural-ode`, `mhx physics`, `mhx diagnostics`, and
  `mhx validate readiness`. New subcommands may be added in minor pre-releases;
  removed or renamed commands require a migration note.

### Compatibility rules

- Patch releases may add optional fields to JSON/NPZ metadata but must keep
  existing v1 keys readable.
- Minor pre-releases may add new diagnostics, benchmarks, plugins, and CLI
  options.
- Breaking changes require either a new public API version or a documented
  deprecation window.
- Active source files must not import archived legacy modules. The CI command
  `python tools/check_legacy_imports.py` enforces this.

### Source links

- [Versioning helpers](https://github.com/uwplasma/MHX/blob/main/src/mhx/versioning.py)
- [Trajectory schema loader](https://github.com/uwplasma/MHX/blob/main/src/mhx/io/trajectory.py)
- [Legacy import guard](https://github.com/uwplasma/MHX/blob/main/tools/check_legacy_imports.py)
