# Diagnostics registry

MHX diagnostics are centralized in `mhx.diagnostics` so runs, reports,
benchmarks, and future scan/inverse-design workflows use the same definitions.
TOML configs select diagnostics through `[diagnostics].quantities`:

```toml
[diagnostics]
quantities = ["energy", "mode_growth", "divergence_error"]
mode = [1, 1]
fit_time_window = [0.02, 0.1]
```

Inspect registered diagnostics from the CLI:

```bash
mhx diagnostics list
```

## Built-in diagnostics

| Name | Output keys |
| --- | --- |
| `energy` | `initial_total_energy`, `final_total_energy`, `final_magnetic_energy`, `final_kinetic_energy` |
| `mode_growth` | `diagnostic_mode`, `fit_time_window`, `fit_sample_count`, `initial_mode_amplitude`, `final_mode_amplitude`, `gamma_fit` |
| `divergence_error` | `final_magnetic_divergence_linf` |

## Energy definitions

For the current 2D reduced-MHD state variables $\psi$ and $\omega=\nabla^2\phi$,
MHX computes

$$
E_B = \frac{1}{2}\langle |\nabla\psi|^2\rangle,
\qquad
E_K = \frac{1}{2}\langle |\nabla\phi|^2\rangle,
\qquad
E = E_B + E_K.
$$

The trajectory diagnostic reports final magnetic/kinetic energy and initial/final
total energy. The benchmark validation report uses these fields to gate
unphysical total-energy growth in the current dissipative FAST benchmark.

## Mode growth

For a configured Fourier mode $(k_x,k_y)$, MHX stores

$$
A_k(t) = |\hat\psi_{k_x,k_y}(t)|.
$$

The fitted growth/decay rate is the least-squares slope of
$\log A_k(t)$ against time over the configured inclusive window:

$$
\gamma_{\mathrm{fit}}
=
\frac{\sum_i (t_i-\bar t)(\log A_i-\overline{\log A})}
       {\sum_i (t_i-\bar t)^2}.
$$

This scalar is a reproducibility diagnostic for the FAST smoke run. It is not
yet a calibrated FKR eigenmode growth-rate claim; that validation layer remains
on the roadmap.

## Divergence error

The reduced-MHD perpendicular magnetic field is represented as

$$
B_\perp = (\partial_y\psi,\,-\partial_x\psi),
$$

so analytically $\nabla\cdot B_\perp=0$. MHX reports the final spectral
consistency check

$$
\|\nabla\cdot B_\perp\|_\infty
=
\|\partial_x\partial_y\psi - \partial_y\partial_x\psi\|_\infty.
$$

For smooth periodic fields this should be near roundoff. The unit tests include
a spectral-zero gate for this diagnostic.

## Python extension API

Use `DiagnosticSpec` and `DiagnosticsRegistry` for new diagnostics:

```python
from mhx.diagnostics import DiagnosticSpec, default_diagnostics_registry

def compute_my_metric(context):
    return {"my_metric": 0.0}

registry = default_diagnostics_registry()
registry.register(
    DiagnosticSpec(
        name="my_metric",
        description="Example user metric.",
        output_keys=("my_metric",),
        compute=compute_my_metric,
    )
)
```

The callable receives a `DiagnosticContext` with the saved trajectory, initial
state, domain lengths, diagnostic Fourier mode, and fit-time window.

## Source links

- [Diagnostics implementation](https://github.com/uwplasma/MHX/blob/main/src/mhx/diagnostics/reduced_mhd.py)
- [Diagnostics tests](https://github.com/uwplasma/MHX/blob/main/tests/test_reduced_mhd.py)
- [Run integration](https://github.com/uwplasma/MHX/blob/main/src/mhx/benchmarks/tearing.py)
