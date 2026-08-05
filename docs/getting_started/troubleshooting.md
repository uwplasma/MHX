# Troubleshooting

Short answers to the failures new users hit first.

## JAX installs but only sees the CPU

Install the accelerator wheel for your platform, for example
`pip install -U "jax[cuda13]"`, then confirm with:

```python
import jax
print(jax.devices())
```

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
lists the wheel for each CUDA and platform combination. MHX uses whatever
backend JAX reports.

## The first run pauses before stepping

That pause is JAX compilation. MHX reports compile time and run time
separately, so a line such as `Done 100 steps in 0.065 s after 0.276 s
compilation` is normal. Compilation repeats when the grid shape, step count,
or integrator changes, not when only parameter values change.

## `shape[0] must divide evenly` errors on parallel runs

Field sharding splits the first grid axis across devices. Pick `shape[0]` as
a multiple of `device_count`. For ensembles, the case count must divide by
the device count instead.

## `t_end must be an integer multiple of dt`

MHX runs a fixed number of equal steps. Choose `t_end` and `dt` so that
`t_end / dt` is a whole number.

## Results differ in the last digits between runs

Float32 reductions on parallel hardware reorder operations. Enable float64
with `JAX_ENABLE_X64=1` for validation and gradient work. The
[precision note](install.md#precision) has the context.

## Implicit runs report `implicit_converged: False`

Check precision first. The Newton tolerances sit near $10^{-9}$, which
float32 cannot reach, so implicit runs need `JAX_ENABLE_X64=1`. If the flag
still fails under float64, reduce `dt` and rerun. Do not use a non-converged
trajectory as evidence. The [time integration page](../physics/time_integration.md)
explains the solver.

## Something else

Open an issue at
[github.com/uwplasma/MHX/issues](https://github.com/uwplasma/MHX/issues) with
the command, the full output, and `mhx version`.
