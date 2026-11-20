# Examples

Every example is a script you read from top to bottom and run with
`python <path>` from the repository root. They follow one template:

1. A docstring that says what the example computes, what it writes and how to
   scale it up.
2. **Input parameters** as UPPER_CASE constants. Edit them to define a new run.
3. **Run**, printing the setup, timings and progress.
4. **Results**, printing the key numbers and saving figures and data.

The scripts take no command-line arguments or environment variables and define no
`main()`. To resize a run on a GPU, `modal run modal_run.py --script <path>
--overrides "NAME=value ..."` replaces the constants and runs the script on
Modal.

## Start here

`gallery/` holds fourteen short scripts that show the public Python API:
reconnection, tearing, implicit stepping, multi-device and multi-process runs,
gradients, 3D MHD, compressible MHD, turbulence, Taylor--Green, Alfven-wave
collisions and Kelvin--Helmholtz. See `gallery/README.md`.

```bash
python examples/gallery/01_reconnection.py
```

## By topic

| Folder | What it holds |
| --- | --- |
| `orszag_tang/` | Comparison with the Vencels kinetic result, autodiff fits of resistivity, viscosity and initial conditions, an inviscid energy check, and 3D tracking |
| `turbulence/` | Forced 2D turbulence, its spectrum, and the render and analysis scripts that read the saved run |
| `sweet_parker/` | The large-`Delta'` collapse check, its scaling plot and a Perlmutter batch script |
| `plasmoid_chain/` | The many-plasmoid double-Harris run and a hybrid forced-turbulent-reconnection run |
| `current_sheet/` | An inverse problem that recovers an effective resistivity from sparse probes |
| `toml/` | Complete inputs for `mhx run`, for 2D, 3D and compressible runs |
| `plugin_template/` | A package template for an external physics or diagnostics plugin |

`run_orszag_tang.py` and `run_linear_tearing.py` are small single-purpose runs.
`make_validation_media.py` regenerates the validation figures under
`docs/_static/validation`.

## Recorded config runs

The TOML files support command-line runs with an explicit config record:

```bash
mhx run examples/toml/orszag_tang_3d.toml
```

The same file drives the Python API with
`mhx.Simulation.from_config(mhx.load_config(path)).run()`. See the
[configuration reference](../docs/reference/config_schema.md) for every model and file.
