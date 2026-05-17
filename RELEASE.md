# Release process

MHX is currently a pre-alpha rebuild. Public releases should be cut only when
the validation suite, documentation build, and artifact pipeline are green.

## Versioning

- Package versions follow SemVer pre-release tags while the solver matures, for
  example `0.1.0a0`, `0.1.0a1`, and later `0.1.0`.
- Public API compatibility is tracked separately by `MHX_PUBLIC_API_VERSION`.
  The current rebuilt API is `v1`.
- Artifact schemas are versioned independently, for example
  `mhx.reduced_mhd.trajectory.v1`.

## Deprecation policy

- Legacy scripts remain archived under `legacy/old_mhx/` for one pre-release
  cycle after the replacement CLI command exists.
- Active code under `src/`, `tests/`, and `examples/` must not import archived
  modules. CI enforces this with `python tools/check_legacy_imports.py`.
- Deprecated workflows must have an old-to-new mapping in `docs/migration.md`.
- Removing a legacy shim requires updating `CHANGELOG.md`, `docs/migration.md`,
  and any docs that mention the removed path.

## Release checklist

```bash
python -m ruff check src tests examples tools
python tools/check_legacy_imports.py
python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=95
sphinx-build -W -b html docs docs/_build/html
mhx validate all --outdir outputs/release/validation_suite
mhx benchmark catalog --outdir outputs/release/catalog
mhx artifact-manifest outputs/release
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Before tagging:

1. Update `src/mhx/_version.py`.
2. Update `CHANGELOG.md`.
3. Regenerate validation/documentation figures if any plotted results changed.
4. Confirm GitHub Actions is green on the release commit.
5. Tag and push:

```bash
git tag v0.1.0a1
git push origin v0.1.0a1
```
