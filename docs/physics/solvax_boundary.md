# MHX and SOLVAX

MHX and [SOLVAX](https://github.com/uwplasma/SOLVAX) split the work along one
rule. MHX owns every calculation that carries physical meaning. SOLVAX owns
numerical algebra that applies to any model.

| MHX owns | SOLVAX owns |
| --- | --- |
| state variables and their signs | matrix-free operator containers |
| boundary and gauge rules | GMRES and recycled Krylov methods |
| spatial discretization | Newton--Krylov solves |
| time-discrete physical residuals | preconditioner composition |
| physics-based preconditioners | implicit differentiation |
| diagnostics and validation gates | |

The seam is the backward-Euler step. MHX forms the reduced-MHD residual

$$
R(u_{n+1}) = u_{n+1} - u_n - \Delta t\, F(u_{n+1})
$$

and builds the spectral diffusion preconditioner from its own operators. It
passes both to `solvax.newton_krylov`, which iterates Newton updates and
solves each linear system with GMRES. SOLVAX returns the new state plus
convergence diagnostics that MHX stores in `result.diagnostics`.

This split keeps both codes honest. MHX never reimplements Krylov algebra,
and SOLVAX never hardcodes plasma physics. Solver improvements arrive through
the dependency. The integration contract is pinned by
[`tests/test_solvax_integration.py`](https://github.com/uwplasma/MHX/blob/main/tests/test_solvax_integration.py).

The base MHX install always includes SOLVAX.
