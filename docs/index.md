# MHX

MHX runs differentiable MHD models in JAX: two-dimensional reduced MHD,
full three-dimensional incompressible MHD, and subsonic compressible MHD. It builds the plasma equations
and diagnostics. [SOLVAX](https://github.com/uwplasma/SOLVAX)
supplies the linear, Krylov, and nonlinear solvers.

Use MHX to study periodic current sheets, tearing modes, magnetic
reconnection, MHD turbulence, and dynamos. MHX does not solve
compressible MHD with shocks.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
```{video} _static/movies/double_harris_reconnection.mp4
:autoplay:
:loop:
:muted:
:nocontrols:
:width: 100%
```
**Magnetic reconnection.** A many-plasmoid double-Harris demonstration
shows magnetic flux with black flux contours. It is solver-generated
morphology, not a converged plasmoid-scaling result.
:::

:::{grid-item}
```{video} _static/movies/decaying_mhd_turbulence_current_256.mp4
:autoplay:
:loop:
:muted:
:nocontrols:
:width: 100%
```
**MHD turbulence.** Current density in decaying reduced-MHD turbulence at
256 x 256. Vortical structures merge and stretch into the thin current
sheets where the energy dissipates.
:::

:::{grid-item}
```{video} _static/movies/orszag_tang_current.mp4
:autoplay:
:loop:
:muted:
:nocontrols:
:width: 100%
```
**Orszag--Tang vortex.** The standard 2-D reduced-MHD benchmark develops
interacting current sheets and transfers magnetic energy to smaller scales.
:::

::::

The turbulence movie replays a gate-passing validation run. The reconnection
movie is a high-resolution solver demonstration. Both are paced to be easy to
follow. The [gallery](gallery.md) documents every movie with its command and
claim boundary.

## Install and run

```bash
git clone https://github.com/uwplasma/MHX.git
cd MHX
python -m pip install .
```

```python
import mhx

simulation = mhx.Simulation(
    shape=(64, 64),
    equilibrium=mhx.PeriodicDoubleHarrisEquilibrium(
        perturbation_amplitude=4.0e-3,
        perturbation_mode=(2, 1),
    ),
    resistivity=5.0e-3,
    viscosity=5.0e-3,
    dt=2.0e-2,
    t_end=40.0,
)
result = simulation.run()
result.print_summary()
result.plot("summary.png")
```

The run prints grid, timing, energy, and divergence data, then writes a
four-panel figure. [Run your first model](getting_started/first_run.md) walks
through every line.

## Find your path

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: getting_started/install
:link-type: doc
Install MHX and run a first reconnection model in minutes.
:::

:::{grid-item-card} Physics
:link: physics/reduced_mhd
:link-type: doc
Read the reduced-MHD equations, assumptions, and model limits.
:::

:::{grid-item-card} Validation
:link: validation/index
:link-type: doc
Check every physics gate, tolerance, and claim boundary.
:::

:::{grid-item-card} API reference
:link: reference/api
:link-type: doc
Look up every public class, function, and CLI command.
:::

::::

## Cite

Use the release tag or commit SHA when you cite a run. Citation metadata is in
[`CITATION.cff`](https://github.com/uwplasma/MHX/blob/main/CITATION.cff).

```{toctree}
:hidden:
:caption: Get started

getting_started/install
getting_started/first_run
getting_started/first_movie
getting_started/troubleshooting
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/01_tearing_mode
tutorials/02_reconnection
tutorials/03_gradients
tutorials/04_inverse_problem
tutorials/05_ensembles_and_devices
tutorials/06_implicit_stepping
```

```{toctree}
:hidden:
:caption: Examples

examples/autodiff_gradients
examples/orszag_tang_vencels
examples/orszag_tang_ic_optimization
examples/turbulence_2d
examples/orszag_tang_3d
examples/turbulence_3d
examples/sweet_parker_collapse
```

```{toctree}
:hidden:
:caption: How-to guides

how_to/run_from_toml
how_to/run_on_gpus
how_to/extend
```

```{toctree}
:hidden:
:caption: Physics

physics/reduced_mhd
physics/mhd3d
physics/compressible
physics/numerics
```

```{toctree}
:hidden:
:caption: Gallery

gallery
```

```{toctree}
:hidden:
:caption: Validation

validation/index
validation/nonlinear
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/cli
reference/config_schema
reference/output_schema
reference/literature
reference/bibliography
```
