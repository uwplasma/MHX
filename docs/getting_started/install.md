# Installation

MHX needs Python 3.10 or newer. Install the current source in an isolated
environment:

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m venv .venv
source .venv/bin/activate
python -m pip install .
mhx version
```

The base install includes JAX for CPU, SOLVAX, Matplotlib, and the `mhx`
command-line tools.

## GPU install

Install the accelerator-specific JAX wheel before MHX, following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
For a CUDA machine:

```bash
python -m pip install -U "jax[cuda13]"
python -m pip install .
```

MHX never pins `jaxlib`. The JAX wheel you install decides the backend.

## Precision

JAX computes in float32 by default. Validation work and gradient checks
should enable float64:

```bash
export JAX_ENABLE_X64=1
```

The [differentiability page](../physics/differentiability.md) explains why
this matters for gradient validation.

## Developer install

The `dev` extra adds the test, lint, and documentation tools:

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not slow"
```

## Verify the install

```bash
mhx api status
mhx benchmark decay --outdir outputs/install_check/resistive_decay
```

The decay command runs an exact physics check and writes
`diagnostics.json`, `validation.json`, and `manifest.json` under
`outputs/install_check/resistive_decay/`. If it passes, the solver, the
spectral operators, and the output stack all work.

If anything fails, check [troubleshooting](troubleshooting.md).
