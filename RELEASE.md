# Release process

MHX is currently a pre-alpha rebuild. Public releases should be cut only when
the tests, the documentation build and the lint check are green.

## Versioning

- Package versions follow SemVer pre-release tags while the solver matures, for
  example `0.1.0a0`, `0.1.0a1`, and later `0.1.0`.
- Public API compatibility is tracked separately by `MHX_PUBLIC_API_VERSION`.
  The current rebuilt API is `v1`.
- Artifact schemas are versioned independently, for example
  `mhx.reduced_mhd.trajectory.v1`.

## Release checklist

```bash
python -m ruff check src tests examples tools
python -m pytest --cov=mhx --cov-report=term-missing --cov-fail-under=85
sphinx-build -W -b html docs docs/_build/html
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Before tagging:

1. Update `src/mhx/_version.py`.
2. Update `CHANGELOG.md`.
3. Confirm GitHub Actions is green on the release commit.
4. Tag and push:

```bash
git tag v0.1.0a1
git push origin v0.1.0a1
```
