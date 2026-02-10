Inverse Design
==============

Inverse design optimizes ``(eta, nu)`` via a small MLP and backpropagates through
the tearing simulation to match a target ``(f_kin, C_plasmoid)``. The objective
is stored with each run and must be used consistently in comparisons.
