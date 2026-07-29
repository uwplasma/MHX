# Performance

MHX reports compilation and execution as separate times. Compare execution
times only after JAX has compiled the program.

## Spatial sharding test

Run the checked example:

```bash
python examples/gallery/06_strong_scaling.py
```

The script keeps the global grid fixed. It runs three samples for each device
count and records the median execution time.

```{image} _static/readme/strong_scaling.png
:alt: MHX CPU and GPU spatial-sharding measurements
:width: 900px
```

The checked data records these runs:

| Hardware | Grid | Steps | Devices | Median run time |
| --- | ---: | ---: | ---: | ---: |
| Apple M4 host | 256 x 256 | 20 | 1 | 0.293 s |
| Apple M4 host | 256 x 256 | 20 | 2 | 0.678 s |
| Apple M4 host | 256 x 256 | 20 | 4 | 1.093 s |
| NVIDIA RTX A4000 | 1024 x 1024 | 50 | 1 | 1.629 s |
| Two RTX A4000 GPUs | 1024 x 1024 | 50 | 2 | 4.362 s |

These measurements date from July 29, 2026. The CPU test used forced logical
JAX devices on one processor. The GPU pair shares a PCIe NUMA node and has no
NVLink connection.

Spatial sharding ran on both platforms. It did not reduce run time on this
hardware. The distributed Fourier communication cost exceeded the local work.
Use one device for these grid sizes on these machines.

Read the full settings:

- [`cpu_spatial_sharding.json`](_static/performance/cpu_spatial_sharding.json)
- [`gpu_spatial_sharding.json`](_static/performance/gpu_spatial_sharding.json)

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
