Physics Plugin Template
=======================

This folder is a minimal template for a third‑party physics plugin. Copy it
into a new repo and rename as needed.

Contents
--------

- `my_term.py`: a minimal `PhysicsTerm` implementation.
- `tests/test_my_term.py`: a basic shape test.
- `docs/index.md`: short documentation stub.

Metadata
--------

Make sure each term defines:

- `name`
- `api_version` (set to `mhx.solver.plugins.API_VERSION`)
- `rhs_additions(...)` with the required keyword arguments.

Validate via:

```
mhx plugin lint
```

Suggested structure
-------------------

```
my_plugin/
  pyproject.toml
  my_plugin/
    __init__.py
    my_term.py
  tests/
    test_my_term.py
  docs/
    index.md
```
