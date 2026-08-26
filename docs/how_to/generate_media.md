# Generate documentation media

MHX comes with a series of media showing some of the capabilities available in
the codebase. Instructions for reproducing these examples are provided below.

Final simulations write source bundles beneath `outputs/media-campaign/final/`.
Rendering reads those bundles and writes GIFs, MP4s, posters, and provenance
records beneath `outputs/media-preview/`. Rendering does not rerun a simulation.

Use `--preset preview` before starting a final run when testing changes to an
example or its plotting code.

## Decaying turbulence

**What this demonstrates**

This simulation shows an decaying turbulence figure and spectrum, showing
MHX's ability to capture turbulent processes, and show energy dissipation.

**Run a preview**

```bash
python examples/media/decaying_turbulence.py simulate --preset preview
python examples/media/decaying_turbulence.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/decaying_turbulence.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/decaying_turbulence.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/readme/decaying_mhd_turbulence_current_256.gif`
- `outputs/media-preview/movies/decaying_mhd_turbulence_current_256.mp4`
- `outputs/media-preview/posters/decaying_mhd_turbulence_current_256.png`
- `outputs/media-preview/records/decaying_mhd_turbulence_current_256.json`

**What to check**

Make sure you observe initial peaks of color (j_z) that quickly fade away to a near uniform background. Another thing to check is the spectrum, which should be initialized with energy in the first 4 wavenumbers, which then becomes a quite smooth downward facing curve that decays.

## Double-Harris current sheet and reconnection

**What this demonstrates**

These plots show the evolution of a double harris sheet, initialized like that for periodicity, and are meant to highlight MHX's ability to show reconnection and plasmoid formation. We plot J_z (in double_harris_current_sheet.mp4) and the magnetic flux psi (in double_harris_reconnection.mp4)

**Run a preview**

```bash
python examples/media/double_harris_reconnection.py simulate --preset preview
python examples/media/double_harris_reconnection.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/double_harris_reconnection.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/double_harris_reconnection.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/readme/double_harris_reconnection.gif`
- `outputs/media-preview/readme/double_harris_current_sheet.gif`
- `outputs/media-preview/movies/double_harris_reconnection.mp4`
- `outputs/media-preview/movies/double_harris_current_sheet.mp4`
- Corresponding posters and provenance records under `outputs/media-preview/`

**What to check**

When observing double_harris_current_sheet.mp4, you should be see a flash of the broadband j_z noise initialized in order to trigger an instability, and should then see clear island forming as the current sheet thins and breaks. In double_harris_reconnection, you should be able to see the same process, but with more visible plasmoid groups via the magnetic flux.

## Double-Harris island tutorial

**What this demonstrates**

This is the small 64 × 64 double-Harris tutorial used to demonstrate the basic
Python movie-generation workflow. It evolves one deliberately seeded magnetic
island and shows how to save frames and assemble a GIF. It is not the retired
high-resolution island campaign and should not be interpreted as a converged
plasmoid or reconnection-rate calculation.

**Run the example**

Follow the complete Python example in
[`docs/getting_started/first_movie.md`](../getting_started/first_movie.md).

The tutorial uses a periodic double-Harris equilibrium with width `0.4`, a
`4.0e-3` perturbation in mode `(2, 1)`, resistivity and viscosity of `5.0e-3`,
`dt=2.0e-2`, `t_end=40.0`, and one saved state every 100 steps.

**Expected outputs**

- `outputs/movies/island/island_flux.gif`
- Individual frame PNGs under `outputs/movies/island/`

The committed documentation version is:

- `docs/_static/readme/double_harris_island_64.gif`
- `docs/_static/movies/double_harris_island_64.mp4`
- `docs/_static/posters/double_harris_island_64.png`

**What to check**

The seeded island component should grow visibly while the red--blue color scale
remains fixed across every frame. Black contours of the total magnetic flux
should stay aligned with the evolving island topology and provide the
double-Harris equilibrium context. The overall pattern and final state should
agree with the movie embedded in the tutorial; a changing per-frame color scale
or contours detached from the image indicates a rendering error.

## Forced two-dimensional turbulence

**What this demonstrates**

This continually forced reduced-MHD calculation sustains two-dimensional
magnetic and velocity fluctuations instead of allowing them simply to decay.
One animation follows the evolving current density; a companion animation
shows the live shell-integrated magnetic spectrum and the compensated quantity
`k^(5/3) E_B(k)`. A constant is fitted to the compensated spectrum from `k=5`
through the 2/3 cutoff, making the degree of agreement with a `k^-5/3` interval
visible without claiming a converged inertial-range measurement.

**Run a preview**

```bash
python examples/media/forced_turbulence_2d.py simulate --preset preview
python examples/media/forced_turbulence_2d.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/forced_turbulence_2d.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/forced_turbulence_2d.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/readme/forced_2d_turbulence.gif`
- `outputs/media-preview/readme/forced_2d_turbulence_spectrum.gif`
- `outputs/media-preview/movies/forced_2d_turbulence.mp4`
- `outputs/media-preview/posters/forced_2d_turbulence.png`
- `outputs/media-preview/posters/forced_2d_turbulence_final_spectrum.png`
- Corresponding provenance records under `outputs/media-preview/records/`

**What to check**

The current-density view should develop persistent, fine filamentary structure
rather than uniformly fading, which confirms that the large-scale forcing is
continuing to supply activity. In the spectrum, power should extend from the
forced large scales through an intermediate range and decrease before the
marked 2/3 cutoff without an artificial pile-up at the highest resolved modes.
The compensated curve should be approximately level over the fitted interval
when a `k^-5/3` range is present. Its normalized root-mean-square error (NRMSE)
measures fractional scatter about the fitted constant: lower values are flatter,
while a large value means the apparent scaling is weak and should not be read as
evidence for a clean power law.

The main poster is selected from the saved timestep with the lowest finite
constant-fit NRMSE. The selected time and NRMSE are recorded in the provenance
JSON.

## Forced three-dimensional turbulence

**What this demonstrates**

This example runs continually forced, incompressible three-dimensional MHD and
visualizes the magnitude of the electric current, `|J|`. The left panel is a
central `z=L_z/2` slice, while the right panel takes the maximum along `z` at
every `(x, y)` location. Together they contrast structure on one plane with the
strongest current encountered anywhere through the volume.

**Run a preview**

```bash
python examples/media/turbulence_3d.py simulate --preset preview
python examples/media/turbulence_3d.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/turbulence_3d.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/turbulence_3d.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/movies/forced_3d_turbulence_current.mp4`
- `outputs/media-preview/posters/forced_3d_turbulence_current.png`
- `outputs/media-preview/records/forced_3d_turbulence_current.json`

**What to check**

Current should organize into evolving sheets, ribbons, and intermittent compact
structures rather than remaining close to the initial pattern. Due to the 3D computational
requirements, it is not run at high enough resolutions to prevent some blocky features

## Kelvin--Helmholtz instability

**What this demonstrates**

This example initializes two smooth periodic shear layers with a small
single-mode transverse perturbation and follows their Kelvin--Helmholtz
instability. A passive dye marks the layers while white vorticity contours show
the rolling vortical flow. The magnetic flux and resistivity are zero, so this
is the viscous hydrodynamic limit of the reduced-MHD evolution rather than a
magnetized Kelvin--Helmholtz calculation.

**Run a preview**

```bash
python examples/media/kelvin_helmholtz.py simulate --preset preview
python examples/media/kelvin_helmholtz.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/kelvin_helmholtz.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/kelvin_helmholtz.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/readme/kelvin_helmholtz.gif`
- `outputs/media-preview/movies/kelvin_helmholtz.mp4`
- `outputs/media-preview/posters/kelvin_helmholtz.png`
- `outputs/media-preview/records/kelvin_helmholtz.json`

**What to check**

The two dye streams mix and show vorticity, and through this process you can
observe MHX's ability to simulate hydrodynamic flows. Compare to literature examples
of the KH instability, like from simulations in dedalus.

## Orszag--Tang vortex in two dimensions

**What this demonstrates**

This is the periodic, two-dimensional reduced-MHD adaptation of the
Orszag--Tang vortex. Its initially smooth velocity and magnetic modes interact
nonlinearly, exercising advection, magnetic tension, spectral transfer, and
resistive-viscous dissipation as a network of thin current sheets forms and
interacts. It is an incompressible reduced-MHD benchmark, not the compressible
shock-capturing Orszag--Tang test.

**Run a preview**

```bash
python examples/media/orszag_tang_2d.py simulate --preset preview
python examples/media/orszag_tang_2d.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/orszag_tang_2d.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/orszag_tang_2d.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/readme/orszag_tang_current.gif`
- `outputs/media-preview/movies/orszag_tang_current.mp4`
- `outputs/media-preview/posters/orszag_tang_current.png`
- `outputs/media-preview/records/orszag_tang_current.json`

**What to check**

The initially broad current pattern should sharpen into interacting positive
and negative current sheets and then weaken as dissipation acts. Every frame
should retain physical `x` and `y` labels, a `j_z` colorbar in code units, and
the displayed simulation time. The symmetric color limits are fixed from the
rendered trajectory, so apparent growth and decay are comparable across time.
The associated validation data should show finite, non-increasing total energy
with net dissipation and increased current and vorticity content at high
wavenumber, rather than unresolved energy accumulating at the grid cutoff.

## Orszag--Tang vortex in three dimensions

**What this demonstrates**

This example evolves the full three-dimensional incompressible-MHD
Orszag--Tang equilibrium and derives the current magnitude `|J|` from the saved
magnetic field. The two panels show a central `z=L_z/2` slice and the maximum
current along `z`, exposing both the current sheets that cross one plane and the
strongest structures present anywhere in the volume.

**Run a preview**

```bash
python examples/media/orszag_tang_3d.py simulate --preset preview
python examples/media/orszag_tang_3d.py render --preset preview
```

**Run the final simulation**

```bash
python examples/media/orszag_tang_3d.py simulate --preset final
```

**Render the final media**

```bash
python examples/media/orszag_tang_3d.py render --preset final
```

**Expected outputs**

- `outputs/media-preview/movies/orszag_tang_3d_current.mp4`
- `outputs/media-preview/posters/orszag_tang_3d_current.png`
- `outputs/media-preview/records/orszag_tang_3d_current.json`

**What to check**

Smooth initial structures should steepen into evolving three-dimensional
current sheets. Features crossing the central plane should appear in both
panels at the same `(x, y)` positions, while the maximum projection may contain
additional or stronger features located away from that plane. With equal
positive viscosity and resistivity, small-scale structure should be regularized
rather than producing persistent grid-scale noise, and the run should exhibit
dissipative evolution. This movie is qualitative morphology evidence from one
resolution: it is not a convergence study, a current-sheet-statistics result,
or the compressible shock-forming Orszag--Tang benchmark.

## Run the complete campaign

To run every currently curated simulation:

```bash
python examples/media/run_all.py simulate --preset final --allow-expensive
```

To render every available final source bundle:

```bash
python examples/media/run_all.py render --preset final
```

To inspect source availability without running anything:

```bash
python examples/media/run_all.py status --preset final
```

Before promoting regenerated files into `docs/_static/`, inspect the media and
run the committed-media checks:

```bash
python tools/render_all_media.py --check-only
```