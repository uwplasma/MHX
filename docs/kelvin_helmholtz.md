# Kelvin--Helmholtz tutorials

MHX includes clean Kelvin--Helmholtz (KH) notebooks for three purposes:

1. a FAST incompressible/reduced-MHD passive-dye run;
2. a FAST backpropagation example through a complete simulation;
3. a smooth, low-Mach compressible-MHD tutorial.

The original imported notebook remains available at
[`examples/KelvinHelmholtz.ipynb`](https://github.com/uwplasma/MHX/blob/main/examples/KelvinHelmholtz.ipynb).
The cleaned notebooks are new examples layered on reusable package code:

- [`src/mhx/benchmarks/kelvin_helmholtz.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/kelvin_helmholtz.py)
- [`src/mhx/equations/compressible_mhd.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/equations/compressible_mhd.py)
- [`src/mhx/state/compressible_mhd.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/state/compressible_mhd.py)

## Run the notebooks

```bash
jupyter notebook examples/run_kelvin_helmholtz_incompressible.ipynb
jupyter notebook examples/run_kelvin_helmholtz_backpropagation.ipynb
jupyter notebook examples/run_kelvin_helmholtz_compressible_mhd.ipynb
```

All three notebooks honor:

```bash
export MHX_EXAMPLE_OUTDIR_ROOT=/path/to/outputs
```

Expected FAST output files are:

- `kelvin_helmholtz_incompressible/kh_incompressible_snapshots.png`
- `kelvin_helmholtz_incompressible/kh_incompressible_entropy.png`
- `kelvin_helmholtz_backpropagation/kh_backpropagation_history.png`
- `kelvin_helmholtz_compressible_mhd/kh_compressible_mhd_snapshots.png`
- `kelvin_helmholtz_compressible_mhd/kh_compressible_mhd_entropy.png`

## Publication-style validation example

The draft PR also includes a standalone validation script that mirrors the
repository's publication-example conventions: user-editable all-caps
parameters at the top, no hidden `main()` function, deterministic output paths,
JSON gates, NPZ histories, manifest hashes, figures, and a compact GIF.

Source links:

- [`examples/publication_kelvin_helmholtz_validation.py`](https://github.com/uwplasma/MHX/blob/main/examples/publication_kelvin_helmholtz_validation.py)
- [`src/mhx/benchmarks/kelvin_helmholtz.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/kelvin_helmholtz.py)
- [`tests/test_kelvin_helmholtz.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_kelvin_helmholtz.py)

Run the CI-sized version:

```bash
MHX_EXAMPLE_FAST=1 \
MHX_EXAMPLE_OUTDIR_ROOT=outputs/examples/publication \
python examples/publication_kelvin_helmholtz_validation.py
```

Run the default validation version, which uses `64×128`, `t_end=2.0`, a
`32×64` resolution-comparison run, and a smooth low-Mach compressible-MHD
positivity check:

```bash
MHX_EXAMPLE_OUTDIR_ROOT=outputs/examples/publication \
python examples/publication_kelvin_helmholtz_validation.py
```

Expected files under
`outputs/examples/publication/kelvin_helmholtz_validation/`:

- `manifest.json`
- `diagnostics.json`
- `validation.json`
- `kelvin_helmholtz_incompressible.npz`
- `kelvin_helmholtz_resolution_comparison.npz`
- `kelvin_helmholtz_compressible_mhd.npz`
- `figures/publication_kelvin_helmholtz_summary.png`
- `figures/kelvin_helmholtz_entropy.png`
- `figures/kelvin_helmholtz_snapshots.png`
- `figures/kelvin_helmholtz_compressible_minima.png`
- `figures/kelvin_helmholtz_dye.gif`

The validation gates require finite histories, a positive passive-dye entropy
response, bounded spectral overshoot of the dye field, consistency of final
entropy between the two resolutions, and positive density/pressure in the
smooth compressible-MHD tutorial. The manifest records `claim_level =
"validation"` by design. Passing this example is evidence that the examples,
IO schema, AD-friendly RK4 path, and smooth tutorial equations are wired
correctly; it is not a high-Reynolds-number KH convergence result and not a
shock-capturing compressible-MHD claim.

```{figure} _static/validation/kelvin_helmholtz/publication_kelvin_helmholtz_summary.png
:alt: Kelvin--Helmholtz validation bundle summary
:width: 95%

Validation summary for the default Kelvin--Helmholtz example. The panels show
entropy growth, compressible-MHD positivity checks, validation metadata, and
dye snapshots with vorticity contours.
```

```{figure} _static/validation/kelvin_helmholtz/kelvin_helmholtz_dye.gif
:alt: Kelvin--Helmholtz passive dye animation
:width: 52%

Compact passive-dye GIF from the default validation run. The animation is meant
for documentation and reviewer orientation; the quantitative gates are in
`validation.json` and `manifest.json`.
```

## Incompressible passive-dye model

The incompressible notebook uses the reduced-MHD hydrodynamic limit
($\psi=0$) with a passive dye $c$. The vorticity equation is

$$
\partial_t \omega + [\phi,\omega] = \nu \nabla^2 \omega,
\qquad
\nabla^2 \phi = \omega,
$$

where $[a,b]=a_x b_y-a_y b_x$. The dye equation is

$$
\partial_t c + [\phi,c] = \nu_c \nabla^2 c.
$$

The smooth double-shear initial condition follows the style of the validated
nonlinear KH benchmark of
[Lecoanet et al.](https://doi.org/10.1093/mnras/stv2564). The FAST notebook
defaults to `32×64`, `t_end=0.2`, and is intentionally early-time. It
demonstrates the setup, diagnostics, and output path; it is not a nonlinear
roll-up convergence claim.

## Differentiating the simulation

The backpropagation notebook defines a scalar map

$$
J(A)=S(c(T;A)),
\qquad
S(c)=\int -c\log(c)\,dA,
$$

where $A$ is the KH perturbation amplitude. It then computes:

- reverse-mode sensitivity with `jax.value_and_grad`;
- forward-mode sensitivity with `jax.jvp`;
- a centered finite-difference check;
- a short gradient-descent update on $A$.

This works because MHX's fixed-step RK4 integrator is a pure PyTree program
implemented with `jax.lax.scan` in
[`src/mhx/time_integrators/fixed_step.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/time_integrators/fixed_step.py).
For scalar objectives and small tutorial runs, unrolled reverse-mode AD is the
simplest and clearest API. For one-parameter sensitivity checks, `jax.jvp` is
cheap and directly comparable to finite differences.

For longer simulations, memory becomes the central issue. The next production
step is checkpoint/rematerialization with
[JAX checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html).
For implicit, steady-state, or fixed-point solvers, use implicit
differentiation patterns like the
[JAXopt implicit-differentiation API](https://jaxopt.github.io/stable/implicit_diff.html).
The spectral-adjoint direction is also informed by
[Skene & Burns, 2025](https://arxiv.org/abs/2506.14792), which emphasizes
constructing fast automated adjoints for spectral PDE solvers rather than
blindly storing every primal time step.

## Smooth compressible-MHD tutorial

The compressible notebook uses conservative ideal-MHD fluxes:

$$
\partial_t \rho + \nabla\cdot(\rho \mathbf{v}) = 0,
$$

$$
\partial_t(\rho\mathbf{v}) + \nabla\cdot
\left[
\rho\mathbf{v}\mathbf{v}
+\left(p+\frac{|\mathbf{B}|^2}{2}\right)\mathbf{I}
-\mathbf{B}\mathbf{B}
\right]=0,
$$

$$
\partial_t E+\nabla\cdot
\left[
\left(E+p+\frac{|\mathbf{B}|^2}{2}\right)\mathbf{v}
-(\mathbf{v}\cdot\mathbf{B})\mathbf{B}
\right]=0,
$$

with the 2D induction equation in conservative form. The tutorial uses
periodic Fourier derivatives, a low-Mach smooth state, and short times. It is
useful for learning the MHX state/RHS pattern and for early autodiff
experiments.

It is **not** a shock-capturing production MHD solver. Production compressible
MHD requires additional gates:

- shock-tube/Riemann benchmarks;
- divergence-control validation, such as constrained transport or
  hyperbolic/parabolic cleaning following
  [Dedner et al.](https://doi.org/10.1006/jcph.2001.6961);
- resolution and timestep convergence;
- positivity-preserving controls;
- documented regimes where spectral derivatives remain appropriate.

## Tests

The KH examples are covered by:

```bash
python -m pytest tests/test_kelvin_helmholtz.py tests/test_compressible_mhd.py -q
python -m pytest tests/test_kelvin_helmholtz_notebooks.py -q
MHX_EXAMPLE_FAST=1 python examples/publication_kelvin_helmholtz_validation.py
```

The notebook execution test is marked `slow` because it compiles and executes
all clean notebook code cells. The non-slow unit tests still cover the reusable
API, finite outputs, primitive/conservative roundtrips, uniform-state RHS
zero, reverse-mode/JVP consistency, finite-difference gradient agreement, and
the validation manifest/NPZ schema.
