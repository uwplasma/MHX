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

## External comparison codes

MHX will document comparison workflows against public or widely used codes:

- [Athena++](https://www.athena-astro.app/)
- [PLUTO](https://ui.adsabs.harvard.edu/abs/2007ApJS..170..228M/abstract)
- [MPI-AMRVAC](https://amrvac.org/)
- [FLASH](https://flash.rochester.edu/site/flashcode/)
- [OpenMHD](https://sci.nao.ac.jp/MEMBER/zenitani/openmhd-e.html)
- [Dedalus](https://dedalus-project.org/)
- [Gkeyll](https://gkeyll.readthedocs.io/)

