# Installation

MHX is currently a pre-alpha source install. Use an isolated environment so
JAX, Sphinx, plotting, and test dependencies remain reproducible.

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
mhx version
```

For GPU or TPU work, install the platform-specific JAX wheel from the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
before installing MHX. The validation suite and CI use CPU-friendly FAST
settings.

Verify the install:

```bash
mhx api status
mhx benchmark decay --outdir outputs/install_check/resistive_decay
sphinx-build -W -b html docs docs/_build/html
```

Expected files:

- `outputs/install_check/resistive_decay/diagnostics.json`
- `outputs/install_check/resistive_decay/validation.json`
- `outputs/install_check/resistive_decay/manifest.json`

Use `MHX_API_VERSION=v1` in reproducibility scripts when you want artifact
loaders and writers to fail early on incompatible future API versions.
