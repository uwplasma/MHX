# Gallery

Every movie here comes from a recorded MHX run. The caption under each one
states what it shows, the command that made it, and the claim boundary. The
[media inventory](project/media_inventory.md) holds the full provenance,
settings, and quality checks for each source.

## Harris-sheet reconnection

```{video} _static/movies/double_harris_reconnection.mp4
:loop:
:muted:
:width: 100%
```

Magnetic flux and contours from the high-resolution many-plasmoid
double-Harris demonstration. It is real solver output, but it is not a
converged Sweet--Parker or plasmoid-scaling result.

```{video} _static/movies/double_harris_current_sheet.mp4
:loop:
:muted:
:width: 100%
```

Companion current-density view of the same many-plasmoid demonstration.

```{video} _static/movies/double_harris_island_64.mp4
:loop:
:muted:
:width: 60%
```

The island movie from [make your first movie](getting_started/first_movie.md)
at its exact documented settings: 64 x 64, $t_{\mathrm{end}}=40$, $S=200$.
Solver output without validation gates, shown so readers can check their own
result.

## Orszag--Tang vortex

```{video} _static/movies/orszag_tang_3d_current.mp4
:loop:
:muted:
:width: 100%
```

Full 3D incompressible MHD: the {cite}`politano1995` Orszag--Tang vortex
at 192 x 192 x 192 to $t=4$, on one office GPU. The panels show the
current magnitude as a midplane slice and a maximum projection:

```bash
python - # mhx.Simulation(shape=(192,)*3, equations="mhd3d", ...) per docs/physics/mhd3d.md
```

Campaign-scale solver output on the road to gate G7. Not yet a gated
validation artifact: the dissipation-peak tolerance waits on the
normalization audit in `plan_3d.md`.

```{video} _static/movies/orszag_tang_current.mp4
:loop:
:muted:
:width: 100%
```

Current-density filament formation in the reduced-MHD Orszag--Tang vortex
{cite}`orszag1979` at 96 x 96 to $t=10$. Generate it with:

```bash
mhx benchmark orszag-tang --outdir outputs/orszag_tang --nx 96 --ny 96 --t-end 10 --movies
```

```{video} _static/movies/orszag_tang_vorticity.mp4
:loop:
:muted:
:width: 100%
```

Vorticity roll-up from the same run.

```{video} _static/movies/orszag_tang_flux.mp4
:loop:
:muted:
:width: 100%
```

Flux-function deformation and dissipative mixing from the same run. All
three are nonlinear reduced-MHD validation media, not compressible
shock-capturing results.

## Turbulence

```{video} _static/movies/decaying_mhd_turbulence_current_256.mp4
:loop:
:muted:
:width: 100%
```

Current density in decaying reduced-MHD turbulence {cite}`biskamp2003` at
256 x 256 to $t=10$, from a gate-passing validation run. Its evidence is
committed under
[`_static/validation/decaying_turbulence_256_t10/`](https://github.com/uwplasma/MHX/tree/main/docs/_static/validation/decaying_turbulence_256_t10):

```bash
mhx benchmark decaying-turbulence --outdir outputs/turbulence --nx 256 --ny 256 --t-end 10 --dt 0.004 --movies
```

```{video} _static/movies/forced_2d_turbulence.mp4
:loop:
:muted:
:width: 100%
```

Continually forced 2-D turbulence at 256 x 256 to $t=50$. The movie combines
current density, the live spectrum, and a compensated-spectrum constant fit.
It is a morphology and spectral diagnostic, not a converged inertial-range
measurement.

```{video} _static/movies/forced_3d_turbulence_current.mp4
:loop:
:muted:
:width: 100%
```

Forced 3-D incompressible turbulence at 96 cubed to $t=200$. The panels show
midplane and maximum-projection current magnitude. This is demonstration
media, not a converged turbulence-statistics result.

## Kelvin--Helmholtz instability

```{video} _static/movies/kelvin_helmholtz.mp4
:loop:
:muted:
:width: 100%
```

Smooth periodic Kelvin--Helmholtz roll-up in the hydrodynamic limit at
128 x 256. It demonstrates the ported example initialization and nonlinear
evolution, not a quantitative instability-growth benchmark.

## Linear theory

```{video} _static/movies/harris_layer_sweep.mp4
:loop:
:muted:
:width: 100%
```

The direct Harris-sheet eigenproblem swept over Lundquist number: growth
decreases with $S$ while the resonant layer narrows, the classical tearing
localization of {cite}`furth1963`. From `mhx benchmark linear-tearing-layer`.

## Reproduce or extend

Render every available final source bundle into staging with:

```bash
python examples/media/run_all.py status --preset final
python examples/media/run_all.py simulate --preset final --allow-expensive
python examples/media/run_all.py render --preset final
```

Missing requested sources are errors. Existing committed files are never treated
as substitutes for missing inputs. The per-case commands and output contract are documented in
`examples/media/README.md`. The [generation-instructions page](how_to/generate_media.md)
provides one placeholder section per curated case. [Make your first movie](getting_started/first_movie.md) shows how to
build your own.
