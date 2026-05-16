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
