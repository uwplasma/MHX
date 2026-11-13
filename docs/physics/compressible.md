# The compressible model

MHX solves subsonic compressible MHD beside the incompressible models,
under the same `Simulation` call. The scope is deliberate and stated:
smooth flows at Mach numbers below about one half, pseudo-spectral,
periodic, with no shock capturing. That class has its own research
lineage: {cite}`dahlburg1990` ran exactly this formulation, and modern
subsonic compressible spectral turbulence follows it {cite}`brodiano2021`.

## The equations

MHX evolves the log density, velocity, and magnetic field with the
isothermal closure $p = c_s^2\rho$:

$$
\partial_t \ln\rho = -\mathbf{u}\cdot\nabla\ln\rho - \nabla\cdot\mathbf{u},
$$

$$
\partial_t \mathbf{u} = -\boldsymbol{\omega}\times\mathbf{u}
  - \nabla\tfrac{u^2}{2} - c_s^2\nabla\ln\rho
  + \frac{\mathbf{j}\times\mathbf{B}}{\rho}
  + \nu\left(\nabla^2\mathbf{u}
  + \tfrac{1}{3}\nabla\nabla\cdot\mathbf{u}\right)
  + \nu_b\,\nabla\nabla\cdot\mathbf{u},
$$

$$
\partial_t \mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B})
  + \eta\nabla^2\mathbf{B}.
$$

Log density is a design decision, not a convenience. It guarantees
positive density without clipping floors. Floors would destroy
differentiability exactly where compressible physics lives. The bulk
viscosity $\nu_b$ damps the dilatational component and gives the linear
gates an independent observable. At low Mach the density fluctuations are
pseudosound of order Mach squared {cite}`ghosh1992`, which gate C2
measures directly.

## Validation so far

| Gate | Statement | Anchor |
| --- | --- | --- |
| C1 | damped sound decays at the Stokes--Kirchhoff rate $\gamma = (\tfrac{4}{3}\nu + \nu_b)k^2/2$, at $10^{-3}$ | Landau--Lifshitz section 79 |
| C1 | oblique fast-magnetosonic frequency and damping against the eigenvalues of the exact per-mode block, at $2\times 10^{-3}$ | exact linear algebra |
| C2 | density fluctuations scale as Mach squared | {cite}`ghosh1992` |
| C3 | the solenoidal velocity converges onto the incompressible module as Mach falls | nearly incompressible theory |
| C4 | a circularly polarized Alfvén pump decays at the Goldstein--Derby rate, within ten percent of the dispersion-relation root; the same pump in incompressible MHD only decays resistively | {cite}`goldstein1978,derby1978` |

Gate C4 is the cross-model discriminator. Parametric decay exists only
in compressible MHD, so one experiment validates the new physics and
the model boundary at once. The remaining ladder entries are the {cite}`dahlburg1989` subsonic
Orszag--Tang comparison and the subsonic turbulence campaign.
[`plan_3d.md`](https://github.com/uwplasma/MHX/blob/main/plan_3d.md)
section 14 tracks both.

## Using it

```python
import mhx

result = mhx.Simulation(
    shape=(64, 64, 4),
    equations="compressible",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    sound_speed=5.0,
    viscosity=5.0e-3,
    bulk_viscosity=5.0e-3,
    resistivity=5.0e-3,
    dt=1.0e-3,
    t_end=1.0,
).run()
result.print_summary()
result.plot("compressible_ot.png")
```

A thin box such as `(64, 64, 4)` gives two-dimensional physics. A cubic
shape gives full 3D. `sound_speed` sets the Mach number for a given flow
amplitude. Do not push toward Mach one: the model states its validity
boundary, and supersonic MHD needs a shock-capturing code
{cite}`picone1991`.
