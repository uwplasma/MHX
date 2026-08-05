# Installation

Install the current source in an isolated environment:

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
mhx version
```

The base install includes SOLVAX, Matplotlib, and the command-line tools. The
`dev` extra adds test, documentation, and lint tools.

For GPU work, install the correct JAX wheel from the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
before you install MHX.

Verify the install:

```bash
mhx api status
mhx benchmark decay --outdir outputs/install_check/resistive_decay
.venv/bin/sphinx-build -W -b html docs docs/_build/html
```

Expected files:

- `outputs/install_check/resistive_decay/diagnostics.json`
- `outputs/install_check/resistive_decay/validation.json`
- `outputs/install_check/resistive_decay/manifest.json`

Set `MHX_API_VERSION=v1` when an artifact reader must reject a future API.
