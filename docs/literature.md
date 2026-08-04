# Literature and code context

MHX is positioned between validated plasma/MHD solvers and differentiable JAX
research workflows.

## Differentiable JAX ecosystem

- [JAX documentation](https://docs.jax.dev/) motivates pure array programs that
  compose with `jit`, `vmap`, `grad`, and accelerator execution.
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/) document
  checkpointed discrete adjoints and continuous/backsolve adjoints for
  differentiable time integration.
- [Lineax](https://docs.kidger.site/lineax/) provides JAX-native matrix-free
  linear solves, useful for implicit diffusion, projection, and elliptic pieces.
- [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) is a current reference for
  differentiable CFD implementation and performance discipline in JAX.

## Spectral and parallel implementation

- [Orszag (1971)](https://doi.org/10.1175/1520-0469(1971)028%3C1074:OTEOAI%3E2.0.CO;2)
  gives the high-wavenumber filtering rule used to remove quadratic aliasing.
  MHX filters nonlinear inputs once in Fourier space and filters each bracket
  output. Differentiation commutes with that mask.
- [JAX-CFD's spectral equations](https://github.com/google/jax-cfd/blob/main/jax_cfd/spectral/equations.py)
  keep the evolving vorticity in Fourier space and form all physical
  derivatives from those coefficients. MHX uses the same transform-reuse
  principle for magnetic flux and vorticity.
- [JAX `shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
  specifies the program seen by each device and exposes collectives such as
  `all_to_all`. MHX uses an explicit map for independent cases because named
  array placement alone did not prevent replicated batched FFT work.
- [JAX multi-process documentation](https://docs.jax.dev/en/latest/multi_process.html)
  defines global devices, process-spanning arrays, initialization order, and
  the requirement that every process enter collectives in the same order.
- [jaxDecomp](https://doi.org/10.21105/joss.08852) implements slab and pencil
  decompositions with local FFTs and global transposes. Its strong-scaling
  results use large three-dimensional grids on H100 GPUs.
- [heFFTe](https://netlib.org/utk/people/JackDongarra/PAPERS/heffte.pdf)
  shows why distributed FFT performance depends on the decomposition,
  communication backend, message latency, and interconnect.
- [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/#multiple-gpu-cufft-transforms)
  notes that a multi-GPU transform is not guaranteed to beat one GPU and that
  NVLink or GPUs under the same PCIe switch give the best results.

These sources lead to two separate parallel paths. Field sharding helps when
one state does not fit on one device, but a two-dimensional FFT must exchange
data. Case sharding is collective-free inside a step and is the measured
strong-scaling path for scans and seed ensembles.

## Plasma and MHD validation targets

The first validation sequence will cover FKR/Coppi tearing growth, plasmoid
instability, ideal tearing, GEM-style Hall reconnection, and generalized Ohm's
law terms. Extended-MHD examples will be added only with explicit assumptions,
equations, tests, and limitations.

For the active tearing validation gates:

- [Furth, Killeen & Rosenbluth (1963), finite-resistivity sheet-pinch instabilities](https://cir.nii.ac.jp/crid/1363107370207531008)
  is the classical constant-$\psi$ resistive tearing reference.
- [MacTaggart (2019), The tearing instability of resistive magnetohydrodynamics](https://eprints.gla.ac.uk/191898/)
  gives the 1D reduced-MHD normal-mode equations and reference growth-rate
  values used by `mhx benchmark linear-tearing-eigenvalue`.
- [MacTaggart & Stewart (2017), Optimal energy growth in current sheets](https://www.maths.gla.ac.uk/~dmactaggart/papers/dmac17c.pdf)
  discusses the discrete generalized eigenproblem, the unique unstable tearing
  eigenvalue near $0.0131$ for $S=1000$, $k=0.5$, and the non-normal spectrum.
- [Rutherford (1973), nonlinear growth of the tearing mode](https://doi.org/10.1063/1.1694232)
  is the nonlinear island-growth reference behind the MHX island-width proxy and
  duration audit; MHX does not yet claim to reproduce this regime with the PDE
  solver.
- [McClements et al. (2022), triggering tearing in a forming current sheet](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/triggering-tearing-in-a-forming-current-sheet-with-the-mirror-instability/38550B29006F97E9EF9E3AA460083BF8)
  gives a modern discussion of FKR versus Coppi regime separation and
  hyper-resistive tearing scalings.
- [Loureiro, Schekochihin & Cowley (2007), instability of current sheets and
  formation of plasmoid chains](https://arxiv.org/abs/astro-ph/0703631)
  is the Sweet-Parker plasmoid-scaling target used in the README schematic:
  $\gamma_{\max}\tau_A\sim S^{1/4}$ and $k_{\max}L\sim S^{3/8}$.
- [Pucci & Velli (2014), reconnection of quasi-singular current sheets](https://doi.org/10.1088/2041-8205/780/2/L19)
  motivates the ideal-tearing aspect-ratio scaling used in the analytic
  validation roadmap.
- [Orszag & Tang (1979), small-scale structure of two-dimensional MHD
  turbulence](https://doi.org/10.1017/S002211207900210X) is the classic vortex
  test adapted in MHX as an incompressible reduced-MHD nonlinear media and
  cascade gate.

For generalized Ohm's law and collisionless/two-fluid reconnection context:

- [Birn et al. (2001), GEM magnetic reconnection challenge](https://www.mendeley.com/catalogue/92e8f29f-a6d8-3c8d-a0fa-b24bf4cb8c88/)
  compares resistive tearing, anisotropic pressure, and Hall effects in a
  common Harris-sheet setup.
- [Shay et al. (2001), Alfvénic collisionless reconnection and the Hall term](https://ftp.bartol.udel.edu/whm/GEM/GEM-reconnection/shayEA-JGR-106-3759-2001.pdf)
  is a standard reference for Hall-mediated fast reconnection in the GEM
  challenge family.
- [Rogers et al. (2001), Role of dispersive waves in collisionless reconnection](https://terpconnect.umd.edu/~drake/publications/reconnection/rogers01.pdf)
  connects Hall/two-fluid terms with whistler/kinetic-Alfvén dispersive physics.
- [Liu et al. (2024), Ohm's law and reconnection rate](https://arxiv.org/abs/2406.00875)
  provides a modern review of the generalized Ohm's-law terms that break
  frozen-in flux in collisionless reconnection.

## External comparison codes

MHX will document comparison workflows against public or widely used codes:

- [Athena++](https://www.athena-astro.app/)
- [PLUTO](https://ui.adsabs.harvard.edu/abs/2007ApJS..170..228M/abstract)
- [MPI-AMRVAC](https://amrvac.org/)
- [FLASH](https://flash.rochester.edu/site/flashcode/)
- [OpenMHD](https://sci.nao.ac.jp/MEMBER/zenitani/openmhd-e.html)
- [Dedalus](https://dedalus-project.org/)
- [Gkeyll](https://gkeyll.readthedocs.io/)
