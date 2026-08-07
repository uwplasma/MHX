# Example gallery

Each script follows the same five steps:

1. Set the output path.
2. Create `mhx.Simulation`.
3. Run the simulation.
4. Print and plot the result.
5. Save the fields and metadata.

Start with `01_reconnection.py`. The first three scripts run on a normal CPU
install. `04_cpu_parallel.py` creates four logical CPU devices before JAX
starts. `05_gpu_parallel.py` uses every visible GPU. `06_strong_scaling.py`
holds a four-case reconnection ensemble fixed, records run time after
compilation, and writes the values beside its plot. `07_multi_process.py` uses
the same ensemble API after JAX connects processes launched by Slurm or Open
MPI. `08_gradient.py` differentiates a solve with respect to resistivity and
proves the gradient against central differences under float64.
`09_orszag_tang_3d.py` runs full 3D incompressible MHD through the same
five steps, at a size a laptop finishes in under a minute.

Run a script from the repository root:

```bash
python examples/gallery/01_reconnection.py
```

The scripts contain their settings near the top. Edit those values to define a
new run. They do not parse command-line arguments or hide work inside a
`main()` function.
