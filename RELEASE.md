Release Process
===============

Deprecation policy
------------------

- Deprecated APIs remain supported for **two minor releases**.
- Deprecations must emit a warning and link to the migration guide.
- Removal requires a changelog entry and a doc update.

1. Update version in `pyproject.toml` and `mhx/version.py`.
2. Update `CHANGELOG.md` with release notes.
3. Run:

```bash
python -m ruff check mhx tests examples
python -m pytest -q
python examples/reproduce_figures.py
```

4. Tag and push:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push --tags
```

5. (Optional) Build and publish:

```bash
python -m build
twine upload dist/*
```
