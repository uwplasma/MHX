# Performance

MHX reports compilation and execution as separate times. Compare execution
times only after JAX has compiled the program.

## Strong-scaling test

Run the checked example:

```bash
python examples/gallery/06_strong_scaling.py
```

The script keeps four reconnection cases and every grid fixed. It divides only
the case axis, runs three samples for each device count, and records the median
execution time. This is strong scaling: the total numerical workload does not
grow with the device count.

```{image} _static/readme/strong_scaling.png
:alt: MHX CPU and GPU reconnection-ensemble strong scaling
:width: 900px
```

The checked data records these runs:

| Hardware | Cases x grid | Steps | Devices | Median run time | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple M4 host | 4 x 256 x 256 | 20 | 1 | 0.668 s | 1.00x |
| Apple M4 host | 4 x 256 x 256 | 20 | 2 | 0.401 s | 1.67x |
| Apple M4 host | 4 x 256 x 256 | 20 | 4 | 0.281 s | 2.38x |
| NVIDIA RTX A4000 | 4 x 1024 x 1024 | 50 | 1 | 2.089 s | 1.00x |
| Two RTX A4000 GPUs | 4 x 1024 x 1024 | 50 | 2 | 1.070 s | 1.95x |

These measurements date from July 29, 2026. The CPU test used forced logical
JAX devices on one processor. The GPU pair crosses PCIe host bridges in one
NUMA node and has no NVLink connection.

`Simulation.run_ensemble` uses an explicit SOLVAX shard map. Each device runs
complete local cases. There are no collectives inside the time loop. Ordinary
named sharding was not enough here: the compiler replicated the batched FFT
work in a local audit, while the explicit map divided it.

Read the full settings:

- [`cpu_ensemble_strong_scaling.json`](_static/performance/cpu_ensemble_strong_scaling.json)
- [`gpu_ensemble_strong_scaling.json`](_static/performance/gpu_ensemble_strong_scaling.json)

Regenerate the README plot after a measurement update:

```bash
python tools/plot_strong_scaling.py
```

## Time a simulation

`SimulationResult` contains both timing values:

```python
result = simulation.run()
print(result.compile_seconds)
print(result.run_seconds)
```

Call `jax.block_until_ready` before you stop a custom JAX timer. Device work is
asynchronous.

## Choose the parallel axis

Use `Simulation.run()` for one trajectory. MHX keeps RK4 states in Fourier
space, batches repeated transforms, and can shard a field when memory requires
it. A two-dimensional FFT still needs device-to-device communication, so one
trajectory does not necessarily run faster on a PCIe-connected GPU pair.

Use `Simulation.run_ensemble()` when cases are independent:

```python
equilibria = tuple(
    mhx.PeriodicDoubleHarrisEquilibrium(
        perturbation_amplitude=1.0e-3 * (1.0 + 0.05 * case)
    )
    for case in range(4)
)

result = mhx.Simulation(
    shape=(1024, 1024),
    device_count=2,
).run_ensemble(equilibria)
```

Seed studies, parameter scans, and uncertainty ensembles have no numerical
reason to communicate during a step. Divide that axis before splitting one
FFT-heavy trajectory.

## Several processes or hosts

JAX must connect the processes before any device query or computation:

```python
from mhx.parallel import initialize_distributed

initialize_distributed()
```

Slurm and Open MPI provide settings that JAX detects. Run the same program in
every process:

```bash
mpirun -np 2 python examples/gallery/07_multi_process.py
```

For a manual launch, pass `coordinator_address`, `num_processes`, and the
distinct `process_id` to `initialize_distributed`. A distributed ensemble uses
every global device. Its case count must divide evenly over devices and
processes. Every process must enter array-gathering calls such as
`EnsembleResult.plot` and `EnsembleResult.save`; only process 0 writes files.

The benchmark command writes a larger timing record:

```bash
mhx benchmark timing \
  --outdir outputs/benchmarks/timing \
  --repeats 3 \
  --warmups 1
```

It writes raw samples, summary values, environment data, a figure, and a
manifest.

## Reduce run time

Use these controls in order:

1. Start with the default RK4 integrator.
2. Use the two-thirds filter for nonlinear production runs.
3. Increase `save_every` when you need fewer retained fields.
4. Use 32-bit arrays only after diagnostics pass at that precision.
5. Test backward Euler when stiffness requires very small RK4 steps.
6. Test sharding with the target grid and hardware.

Backward Euler adds Newton and GMRES work. It can still reduce total time when
one stable implicit step replaces many explicit steps.

## Control memory

MHX stores only the states selected by `save_every`. A retained reduced-MHD
state contains magnetic flux and vorticity.

Estimate the raw field storage with:

```text
saved states x nx x ny x 2 fields x bytes per value
```

JAX also needs temporary arrays for RK4 stages, Fourier transforms, and solver
work. Those arrays can exceed the retained history.

When several programs share a GPU, set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Long reverse-mode runs need checkpointing or a custom adjoint. Saving fewer
output states does not remove the reverse-mode tape.

## Source and tests

- [`src/mhx/benchmarks/timing.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/timing.py).
- [`tests/test_timing_benchmark.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_timing_benchmark.py).
- [`.github/workflows/ci.yml`](https://github.com/uwplasma/MHX/blob/main/.github/workflows/ci.yml).
