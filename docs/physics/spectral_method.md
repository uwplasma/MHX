# Spectral method

MHX discretizes space with a Fourier pseudo-spectral method on a uniform,
doubly periodic grid. Derivatives are exact in the resolved band, dissipation
is purely physical, and the divergence constraint holds to machine precision.
This page defines the discrete operators and the dealiasing rule.

## Fourier representation

A field $f(x,y)$ on an $N_x \times N_y$ grid with lengths $(L_x, L_y)$ has the
discrete transform

$$
f(x,y) = \sum_{\mathbf{k}} \hat f_{\mathbf{k}}\,
  e^{i(k_x x + k_y y)}, \qquad
k_x = \frac{2\pi m}{L_x},\quad
k_y = \frac{2\pi n}{L_y},
$$

with integer mode numbers $m$ and $n$. MHX computes transforms with
`jax.numpy.fft` and keeps both evolving fields in Fourier space between
steps. Derivatives are diagonal there:

$$
\widehat{\partial_x f}_{\mathbf{k}} = i k_x \hat f_{\mathbf{k}}, \qquad
\widehat{\nabla^2 f}_{\mathbf{k}} = -|\mathbf{k}|^2 \hat f_{\mathbf{k}} .
$$

For smooth periodic fields the truncation error decays faster than any power
of the grid spacing {cite}`canuto1988`. In practice the resolution question
is physical: the current-layer width must fit in the resolved band. The
[convergence evidence](../validation/reconnection_campaigns.md) measures this
for the reconnection benchmarks.

## Inverse Laplacian and gauge

The stream function solves $\nabla^2\phi = \omega$:

$$
\hat\phi_{\mathbf{k}} = -\frac{\hat\omega_{\mathbf{k}}}{|\mathbf{k}|^2},
\qquad \hat\phi_{\mathbf{0}} = 0 .
$$

Fixing the zero mode removes the free constant in $\phi$ and $\psi$. Only
gradients of the potentials are physical, so this gauge choice does not
affect any observable.

## Dealiasing

The Poisson brackets multiply fields in real space. On a discrete grid the
product of two resolved modes can fold back onto a wrong resolved mode. MHX
removes this aliasing with the two-thirds rule of {cite}`orszag1971`: before
each product it zeros every mode with $|m| > N_x/3$ or $|n| > N_y/3$.

Runs should set `dealiasing="two_thirds"`, and `mhx.Simulation` does so by
default. The `"none"` option exists for operator unit tests, where exactness
matters and no products occur.

## Batched transforms

One right-hand-side evaluation needs eight real-space derivative fields for
the three brackets. MHX computes them with one batched inverse transform and
returns the bracket results with one batched forward transform. On a device
mesh this issues one large collective instead of many small ones, which is
the main cost lever for [sharded runs](../how_to/run_on_gpus.md).

The operators live in
[`numerics/spectral/operators.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/numerics/spectral/operators.py).
The equation assembly lives in
[`equations/reduced_mhd.py`](https://github.com/uwplasma/MHX/blob/main/src/mhx/equations/reduced_mhd.py).
