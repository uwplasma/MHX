# Reduced-MHD spectral smoke benchmark

The first physical workflow in the rebuilt MHX is a deliberately small periodic
2D reduced-MHD benchmark. It is intended to validate package architecture,
spectral operators, time integration, diagnostics, and automatic
differentiation. It is not yet a calibrated FKR tearing-growth benchmark.

## Equations

MHX evolves magnetic flux $\psi(x,y,t)$ and vorticity $\omega(x,y,t)$:

$$
\partial_t \psi + [\phi,\psi] = \eta \nabla^2 \psi,
$$

$$
\partial_t \omega + [\phi,\omega] =
[\psi,\nabla^2\psi] + \nu \nabla^2 \omega,
$$

with

$$
\nabla^2 \phi = \omega,
$$

and periodic boundary conditions. The 2D Poisson bracket is

$$
[a,b] = \partial_x a\,\partial_y b - \partial_y a\,\partial_x b.
$$

The implementation uses FFT derivatives and an inverse Laplacian with the zero
Fourier mode fixed to zero:

$$
\widehat{\nabla^{-2} f}_{\mathbf{k}} =
-\frac{\hat f_{\mathbf{k}}}{|\mathbf{k}|^2}, \qquad
\widehat{\nabla^{-2} f}_{\mathbf{0}} = 0.
$$

## FAST initial condition

The current smoke initial condition is

$$
\psi_0 = \cos(y) + \epsilon \cos(x)\cos(y), \qquad \omega_0 = 0.
$$

It is chosen because it is periodic, deterministic, smooth, and inexpensive. A
future benchmark will replace this with a documented tearing eigenfunction and
growth-rate comparison.

## Diagnostics

The active diagnostics are

$$
E_B = \frac{1}{2}\langle |\nabla\psi|^2\rangle,
\qquad
E_K = \frac{1}{2}\langle |\nabla\phi|^2\rangle.
$$

`mhx run examples/linear_tearing.toml --outdir outputs/smoke` writes a JSON
manifest and scalar energy diagnostics for this benchmark.

## Differentiability check

The test suite includes a finite-difference check of

$$
\frac{\partial E_B(t_f)}{\partial \eta}
$$

through a three-step RK4 solve using `jax.grad`. This is intentionally tiny but
establishes the contract that future inverse-design objectives must remain
differentiable unless explicitly documented otherwise.

## Mode amplitude and growth rate

The first tearing-oriented diagnostic tracks the normalized Fourier amplitude

$$
A_{k_x,k_y}(t) = |\hat{\psi}_{k_x,k_y}(t)|
$$

for a configured mode, currently `(1, 1)` in the FAST smoke benchmark. MHX fits

$$
A(t) \approx A_0 e^{\gamma t}
$$

by least squares on $\log A(t)$. This `gamma_fit` is useful for plumbing and
regression tests, but it should not be interpreted as an FKR tearing rate until
the eigenfunction, equilibrium, fit window, and parameter regime are validated.
