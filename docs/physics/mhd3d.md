# The 3D incompressible MHD model

MHX solves full three-dimensional incompressible visco-resistive MHD in a
periodic box, beside the [2D reduced model](reduced_mhd.md). The same
`mhx.Simulation` call runs both: a three-entry `shape` with
`equations="mhd3d"` selects this model. The program plan behind it, with
the full benchmark ladder and delivery state, is
[`plan_3d.md`](https://github.com/uwplasma/MHX/blob/main/plan_3d.md).

## The equations

MHX evolves the velocity and magnetic field in Alfvén units:

$$
\partial_t \mathbf{v} = \mathcal{P}\left[
  \mathbf{v}\times\boldsymbol{\omega} + \mathbf{j}\times\mathbf{B}
\right] + \nu\nabla^2\mathbf{v},
$$

$$
\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
  + \eta\nabla^2\mathbf{B},
$$

with $\boldsymbol{\omega} = \nabla\times\mathbf{v}$,
$\mathbf{j} = \nabla\times\mathbf{B}$, and the spectral Leray projector

$$
\mathcal{P}_{ij}(\mathbf{k}) = \delta_{ij} - \frac{k_i k_j}{k^2},
$$

which eliminates the pressure and keeps
$\nabla\cdot\mathbf{v} = \nabla\cdot\mathbf{B} = 0$ at round-off. The
rotational form of the advection term conserves energy under truncation.
An optional uniform guide field $\mathbf{B}_0$ enters through the
real-space products. That one addition reproduces the
$\mathbf{j}\times\mathbf{B}_0$ force and the $\mathbf{B}_0$ advection in
the induction equation, with no separate terms.

## Where the equations come from

Start from compressible visco-resistive MHD and take the constant-density
limit: sound waves leave the system and $\nabla\cdot\mathbf{v} = 0$
becomes a constraint rather than an evolution. Measure the field in
Alfvén-speed units, $\mathbf{B}/\sqrt{\mu_0\rho_0} \to \mathbf{B}$, so
the momentum and induction equations read

$$
\partial_t \mathbf{v} + (\mathbf{v}\cdot\nabla)\mathbf{v}
  = -\nabla p + (\nabla\times\mathbf{B})\times\mathbf{B}
  + \nu\nabla^2\mathbf{v},
\qquad
\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
  + \eta\nabla^2\mathbf{B}.
$$

Two identities give the solved form. The identity
$(\mathbf{v}\cdot\nabla)\mathbf{v} =
\boldsymbol{\omega}\times\mathbf{v} + \nabla(v^2/2)$ moves the advection
into rotational form. Its gradient part folds into the pressure, which
then enforces incompressibility alone, so applying $\mathcal{P}$ removes
it exactly. In Fourier space the projected equations close over
$(\hat{\mathbf{v}}, \hat{\mathbf{B}})$ with no elliptic solve. The curl form of the induction term keeps
$\mathbf{k}\cdot\hat{\mathbf{B}} = 0$ invariant, because a curl is
orthogonal to $\mathbf{k}$ mode by mode.

The ideal system conserves the energy
$E = \tfrac{1}{2}\langle v^2 + B^2 \rangle$, the cross helicity
$H_C = \tfrac{1}{2}\langle \mathbf{v}\cdot\mathbf{B} \rangle$, and the
magnetic helicity $H_M = \langle \mathbf{A}\cdot\mathbf{B} \rangle$
{cite}`frisch1975`. In Elsässer variables
$\mathbf{z}^\pm = \mathbf{v} \pm \mathbf{B}$ the nonlinearity couples
only counter-propagating fields, which is the structural fact behind
Alfvénic turbulence phenomenology {cite}`biskamp2003`. The
[2D reduced model](reduced_mhd.md) follows from this system in the
strong-guide-field limit {cite}`strauss1976`, and gate G12 of the
program plan measures that limit directly.

## Numerics

- **Space**: Fourier pseudo-spectral on the periodic box, half-spectrum
  real transforms, and the two-thirds dealiasing rule {cite}`orszag1971`.
  On a device mesh, the transforms run as a slab decomposition with one
  all-to-all transpose, and the compiled program contains no field-sized
  gather. The [parallel contract tests](https://github.com/uwplasma/MHX/blob/main/tests/test_mhd3d_parallel.py)
  pin sharded-versus-single parity for values and gradients.
- **Time**: Williamson two-register RK3 on the integrating-factor
  transformed variable. The diffusive terms integrate exactly, so only
  the advective and Alfvén scales limit the step. A Cox--Matthews ETDRK4
  stepper {cite}`coxmatthews2002`, with coefficients evaluated through
  the Kassam--Trefethen contour mean {cite}`kassam2005`, cross-checks
  every wave gate at fourth order.
- **Constraints**: kinetic and magnetic energy, cross helicity, and
  magnetic helicity are first-class diagnostics with Parseval-exact
  definitions, and the ideal-limit gates bound their drift.

## Validation so far

The gate ladder and its literature anchors live in `plan_3d.md`. The
gates passing in CI today:

| Gate | Statement | Anchor |
| --- | --- | --- |
| G1 | single-mode resistive decay exact at $10^{-12}$; divergence at round-off | exact solution |
| G2 | damped oblique Alfvén dispersion at $10^{-3}$ from the exact eigenvector | exact solution |
| G3 | large-amplitude circularly polarized Alfvén wave, third-order convergence | exact Walén state |
| G4 | ideal invariants drift below $10^{-6}$ and converge with the step | {cite}`frisch1975` |
| G5 | ABC kinematic dynamo windows at $R_m = 1/\eta$ | {cite}`galloway1986,bouya2013` |
| strong $B_0$ | exact dispersion at $B_0/b = 33$ | exact solution |
| gradients | $dE/d\nu$ against finite differences at $10^{-6}$ under x64 | code contract |

The first campaign-scale nonlinear run is the 3D Orszag--Tang vortex of
{cite}`politano1995` at $128^3$. It shows current-sheet formation, a
dissipation peak inside the window reported by {cite}`mininni2006`, and
an internally closed energy budget. Its promotion into gate G7 follows
the normalization audit recorded in the plan.

```{video} ../_static/movies/orszag_tang_3d_current.mp4
:loop:
:muted:
:width: 100%
```

The movie shows $|\mathbf{j}|$ from that run on one fixed color scale:
the midplane slice on the left, the maximum-intensity projection along
$z$ on the right. Current sheets form, roll up, and fragment through the
dissipation peak near $t = 2.8$.

## Using it

```python
import mhx

result = mhx.Simulation(
    shape=(128, 128, 128),
    equations="mhd3d",
    equilibrium=mhx.OrszagTang3DEquilibrium(beta=0.8),
    viscosity=2.0e-3,
    resistivity=2.0e-3,
    dt=1.0e-3,
    t_end=4.0,
    save_every=200,
).run()
result.print_summary()
result.plot("ot3d.png")
```

`result.plot` writes midplane slices of $|\mathbf{j}|$ and $|\mathbf{v}|$
with the energy and cross-helicity histories. Built-in equilibria:
`OrszagTang3DEquilibrium`, `TaylorGreenEquilibrium`,
`ABCFlowEquilibrium`, `CircularlyPolarizedAlfvenEquilibrium`, and
`SingleModeEquilibrium`. All formulas match the cited papers exactly, and
the tests pin their exact mean energies.
