# Performance guide

MHX performance reporting is designed for reproducible engineering comparisons,
not for hardware-independent pass/fail claims. The active benchmark matrix is
small enough for CI and produces artifacts that can be downloaded and compared
between commits.

## Run timing artifacts

```bash
mhx benchmark timing --outdir outputs/benchmarks/timing --repeats 3 --warmups 1
```

Expected files:

- `outputs/benchmarks/timing/timing.json`
- `outputs/benchmarks/timing/timing.md`
- `outputs/benchmarks/timing/figures/timing_summary.png`
- `outputs/benchmarks/timing/manifest.json`

The JSON schema is `mhx.benchmark.timing.v1`. Each case records raw repeat
durations, median/min/max wall time, peak Python allocations from
`tracemalloc`, and environment metadata including Python, JAX, NumPy, and the
selected JAX backend.

```{image} _static/performance/timing_summary.png
:alt: MHX FAST benchmark timing summary
:width: 780px
```

## Current cases

| Case | What it exercises |
| --- | --- |
| `linear_tearing_fast` | Config loading, periodic spectral derivatives, RK4 stepping, diagnostics, and reduced-MHD RHS evaluation. |
| `resistive_decay_fast` | Exact Fourier-mode resistive diffusion gate with numerical error diagnostics. |
| `reconnection_scaling` | Analytic FKR, Sweet-Parker plasmoid, and ideal-tearing scaling scaffolds. |

## Interpreting the numbers

- Compare timings only on the same machine class or the same GitHub Actions
  runner type.
- CI verifies finite positive timings and required artifact files; it does not
  enforce absolute runtime thresholds.
- `tracemalloc` reports Python allocations, not GPU/TPU device memory. Future
  accelerator benchmarks should add backend-specific memory probes.
- JAX compilation and caching can dominate small cases. Use `--warmups` to
  remove first-call overhead when comparing local changes.

## Performance knobs

The active TOML config exposes the first controls users should tune:

```toml
[mesh]
shape = [32, 32]

[time]
t1 = 0.1
dt = 0.01
save_every = 1

[numerics]
enable_x64 = true
enable_jit = true
```

Larger `mesh.shape` values increase spectral FFT cost and trajectory storage.
Smaller `dt` improves temporal resolution but increases the number of RHS
evaluations. Larger `save_every` reduces IO and plotting memory. X64 is used in
physics validation gates; exploratory performance runs may use X32 after a
regression check confirms diagnostics remain stable.

## Source links

- [Timing implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/timing.py)
- [Timing tests](https://github.com/uwplasma/MHX/blob/main/tests/test_timing_benchmark.py)
- [CI artifact workflow](https://github.com/uwplasma/MHX/blob/main/.github/workflows/ci.yml)
