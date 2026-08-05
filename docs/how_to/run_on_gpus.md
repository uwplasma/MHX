# Run on several devices

MHX parallelizes in two ways. Field sharding splits one large simulation
across devices. Ensemble parallelism gives each device complete, independent
cases. Choose by workload:

| Workload | Use | Why |
| --- | --- | --- |
| one trajectory too large for one device | field sharding | the FFTs communicate, everything else is local |
| parameter scans, seed studies | ensembles | no communication inside the time loop |

## Field sharding

Set `device_count` on the simulation. MHX builds a one-dimensional JAX device
mesh and splits the first grid axis:

```python
result = mhx.Simulation(
    shape=(1024, 1024),
    device_count=2,
).run()
```

`shape[0]` must divide evenly by `device_count`. JAX compiles one SPMD
program, and the distributed two-dimensional FFT exchanges data between
devices at each derivative evaluation.

## Ensembles

Pass one equilibrium per case to `run_ensemble`:

```python
equilibria = tuple(
    mhx.PeriodicDoubleHarrisEquilibrium(
        perturbation_amplitude=1.0e-3 * (1.0 + 0.05 * case)
    )
    for case in range(4)
)

result = mhx.Simulation(
    shape=(256, 256),
    device_count=4,
).run_ensemble(equilibria)
```

Each device advances its own cases with no communication inside the time
loop, so ensembles scale better than sharding. Prefer them whenever the
cases are independent.

## CPU devices for practice

JAX can split one host CPU into logical devices, which makes the parallel
path testable on a laptop:

```bash
python examples/gallery/04_cpu_parallel.py
```

The script sets `XLA_FLAGS=--xla_force_host_platform_device_count=4` before
JAX starts, then runs a four-device sharded field.

## GPUs

Install the CUDA wheel ([installation](../getting_started/install.md#gpu-install)),
then run on every visible GPU:

```bash
JAX_PLATFORM_NAME=gpu python examples/gallery/05_gpu_parallel.py
```

MHX reports compile and run times separately. Measured strong-scaling
numbers, with their exact settings and hardware, are on the
[performance page](../reference/performance.md).

## Several processes

On a cluster or an MPI installation, run one copy of the script per process:

```bash
mpirun -np 2 python examples/gallery/07_multi_process.py
```

Every process runs the same script. `mhx.initialize_distributed()` must
execute before any code queries JAX devices. Each process then writes its
own shard of the ensemble output, and the manifest records the process
layout.
