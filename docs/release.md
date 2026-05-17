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
mhx benchmark catalog --outdir outputs/release/catalog
mhx artifact-manifest outputs/release
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Release artifacts

A release candidate should include:

- `validation_suite.json` and `validation_suite.md`;
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

## Source links

- [Release checklist](https://github.com/uwplasma/MHX/blob/main/RELEASE.md)
- [Changelog](https://github.com/uwplasma/MHX/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/uwplasma/MHX/blob/main/CITATION.cff)
