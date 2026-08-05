# The reduced-MHD model

MHX evolves two scalar fields on a periodic two-dimensional domain: the
magnetic flux function $\psi(x,y,t)$ and the fluid vorticity $\omega(x,y,t)$.
These two fields describe an incompressible, magnetized plasma with a strong,
uniform guide field. The model captures tearing modes, magnetic reconnection,
and reduced-MHD turbulence at a small fraction of the cost of full MHD.

This page states the equations exactly as the code solves them, then derives
them from resistive MHD. It ends with the built-in equilibria, the optional
physics terms, and the limits of the model.

## The equations

MHX solves

$$
\partial_t \psi + [\phi,\psi] = \eta \nabla^2 \psi + S_\psi,
$$

$$
\partial_t \omega + [\phi,\omega] = [\psi,\nabla^2\psi]
  + \nu \nabla^2 \omega + S_\omega,
$$

with the stream function defined by

$$
\nabla^2 \phi = \omega,
$$

on a doubly periodic domain. The two-dimensional Poisson bracket is

$$
[a,b] = \partial_x a\,\partial_y b - \partial_y a\,\partial_x b.
$$

The optional sources $S_\psi$ and $S_\omega$ come from
[physics term plugins](#optional-physics-terms) and default to zero.
The implementation is
[`reduced_mhd_rhs`](https://github.com/uwplasma/MHX/blob/main/src/mhx/equations/reduced_mhd.py).

| Symbol | Meaning | Code name |
| --- | --- | --- |
| $\psi$ | magnetic flux function | `psi` |
| $\omega$ | out-of-plane vorticity | `omega` |
| $\phi$ | stream function, solved from $\omega$ | internal |
| $\eta$ | resistivity, inverse Lundquist number | `resistivity` |
| $\nu$ | viscosity, inverse Reynolds number | `viscosity` |
| $j_z = -\nabla^2\psi$ | out-of-plane current density | `current_density` |

The in-plane fields follow from the two potentials:

$$
\mathbf{B}_\perp = \nabla\psi\times\hat{z}, \qquad
\mathbf{v} = \hat{z}\times\nabla\phi .
$$

Both are divergence-free by construction, so the model is exactly
incompressible and $\nabla\cdot\mathbf{B}=0$ holds to machine precision.

## Where the equations come from

Start from incompressible, visco-resistive MHD in Alfvén units:

$$
\partial_t \mathbf{v} + \mathbf{v}\cdot\nabla\mathbf{v}
  = -\nabla p + \mathbf{j}\times\mathbf{B} + \nu\nabla^2\mathbf{v},
$$

$$
\partial_t \mathbf{B} = \nabla\times(\mathbf{v}\times\mathbf{B})
  + \eta\nabla^2\mathbf{B}, \qquad \nabla\cdot\mathbf{v}=0 .
$$

Impose two-dimensional symmetry, $\partial_z = 0$, with a uniform guide field
$B_0\hat{z}$. Write the in-plane fields through the potentials above. Three
identities then reduce the vector system to two scalar equations:

1. Advection becomes a bracket: $\mathbf{v}\cdot\nabla f = [\phi,f]$.
2. Parallel gradients become a bracket: $\mathbf{B}_\perp\cdot\nabla g = -[\psi,g]$.
3. The curl of the momentum equation removes the pressure gradient.

The $z$ component of the induction equation gives the flux equation. The
$z$ component of the curled momentum equation gives the vorticity equation,
where the Lorentz force appears as
$\mathbf{B}_\perp\cdot\nabla j_z = [\psi,\nabla^2\psi]$.

These are the two-dimensional limit of the reduced equations that
{cite}`strauss1976` derived for tokamak plasmas with a strong guide field.
{cite}`biskamp2000` gives a textbook derivation and the reconnection context.
The guide field $B_0$ drops out of the two-dimensional equations. It matters
physically because it justifies incompressibility and the ordering
$|\mathbf{B}_\perp| \ll B_0$.

## Normalization

The equations are dimensionless. Lengths are measured in a reference scale
$L_0$, in-plane magnetic fields in a reference value $B_\perp$, and velocities
in the corresponding Alfvén speed $v_A = B_\perp/\sqrt{\mu_0\rho_0}$. Time is
measured in Alfvén times $\tau_A = L_0/v_A$.

With this choice the dimensionless coefficients are inverse quality factors:

$$
\eta = \frac{1}{S}, \qquad
\nu = \frac{1}{\mathrm{Re}}, \qquad
\mathrm{Pm} = \frac{\nu}{\eta},
$$

where $S$ is the Lundquist number, $\mathrm{Re}$ the Reynolds number, and
$\mathrm{Pm}$ the magnetic Prandtl number. A run with `resistivity=5.0e-3`
therefore models $S = 200$. Tearing-mode theory organizes its predictions by
these numbers {cite}`furth1963`, so every validation page states them.

## Conservation laws

In the ideal limit $\eta = \nu = 0$ the model conserves the energy

$$
E = \frac{1}{2}\left\langle |\nabla\psi|^2 + |\nabla\phi|^2 \right\rangle,
$$

the cross helicity $\langle \nabla\psi\cdot\nabla\phi \rangle$, and the mean
square flux $\langle \psi^2 \rangle$. Angle brackets denote domain averages.
With dissipation the exact energy balance is

$$
\frac{dE}{dt} = -\eta\,\langle j_z^2 \rangle - \nu\,\langle \omega^2 \rangle
  \le 0,
$$

where the two terms are Ohmic and viscous dissipation. The
[nonlinear energy-budget gate](../validation/nonlinear.md) checks this
identity numerically at every release. Faster decay of energy than of mean
square flux is a classical property of two-dimensional MHD turbulence
{cite}`biskamp2003`.

## Built-in equilibria

Every simulation starts from an equilibrium object that builds the initial
fields. All start from rest, $\omega_0 = 0$.

### Periodic double Harris sheet

`PeriodicDoubleHarrisEquilibrium` places two oppositely signed
{cite}`harris1962` current sheets at $x_L = L_x/4$ and $x_R = 3L_x/4$:

$$
\psi_{\mathrm{eq}}(x) = A\,a\left[
  \ln\cosh\frac{x-x_L}{a} - \ln\cosh\frac{x-x_R}{a}
\right] - A\,x,
$$

so the reconnecting field $B_y = -\partial_x\psi$ jumps between $+A$ and $-A$
across each sheet of half-width $a$. Two sheets make the configuration
periodic without boundary layers. An optional seed perturbation

$$
\delta\psi = \varepsilon
  \cos\frac{2\pi m x}{L_x}\,
  \cos\frac{2\pi n y}{L_y}
$$

with amplitude $\varepsilon$ and mode numbers $(m,n)$ starts reconnection at
a controlled wavelength. This is the default initial condition of
`mhx.Simulation` with $a=0.4$, $A=1$, and $\varepsilon=10^{-3}$.

### Cosine current sheet

`CosineTearingEquilibrium` uses the smooth periodic sheet

$$
\psi_{\mathrm{eq}} = \cos\frac{2\pi y}{L_y}, \qquad
\delta\psi = \varepsilon
  \cos\frac{2\pi x}{L_x}\,
  \cos\frac{2\pi y}{L_y}.
$$

It is inexpensive and fully smooth, so the linear-algebra and
differentiability gates use it. It is not a Harris sheet, and its growth
rates need their own eigenvalue reference before comparison.

### Zero equilibrium

`ZeroEquilibrium` returns $\psi_0 = \omega_0 = 0$. Unit tests and plugin
demonstrations use it.

## Optional physics terms

Extra right-hand-side terms enter through the plugin registry and stay off by
default. Each adds its contribution to $(S_\psi, S_\omega)$:

| Term | Adds | Status |
| --- | --- | --- |
| `hyper_resistivity` | $-\eta_4\nabla^4\psi$ and $-\nu_4\nabla^4\omega$ | scale filter for marginally resolved runs |
| `vorticity_drag` | $-\alpha\,\omega$ | simple momentum sink |
| `toy_hall_ohm` | $d_i\,[j_z,\psi]$ in the flux equation | toy model, not gated Hall physics |
| `electron_pressure_tensor` | anisotropic current smoothing | toy closure, not gated physics |

The toy terms exist to exercise the plugin interface. Do not attach physical
claims to them. [Extend physics](../how_to/extend_physics.md) shows how to
register your own term.

## Assumptions and limits

The model assumes, in decreasing order of importance:

1. Two-dimensional dynamics with an ignorable $z$ coordinate.
2. A strong, uniform guide field with $|\mathbf{B}_\perp| \ll B_0$.
3. Incompressible flow with constant density.
4. Doubly periodic boundaries.
5. Constant scalar $\eta$ and $\nu$.

MHX therefore cannot represent compressible dynamics, shocks, sound waves,
three-dimensional instabilities, kinetic scales, or line-tied boundaries.
Sweet--Parker sheets {cite}`sweet1958,parker1957`, plasmoid chains
{cite}`loureiro2007`, and ideal-tearing onset {cite}`pucci2014` remain within
scope at the resolutions the [validation pages](../validation/index.md)
document. Claims outside these limits need a different code or a future MHX
equation module.
