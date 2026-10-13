"""Documented parameter scan used to choose the Stage B double barrier.

Criterion: >= 3 resonances in E < V, widths spanning >= 2 orders of magnitude, the
narrowest one narrower than a typical training-grid spacing (~0.06 for N = 160),
reference grid able to resolve it (>= 50 points per FWHM at spacing 1e-4).
"""
import numpy as np

from qscatter.physics.potentials import double_barrier
from qscatter.physics.resonances import find_resonances
from qscatter.physics.smatrix import compute_transmission

CANDIDATES = [(10, 0.5, 2.0), (10, 0.4, 2.0), (8, 0.5, 2.0), (10, 0.6, 2.0), (10, 0.5, 1.5), (6, 0.6, 2.0),
              (10, 0.8, 3.0), (10, 1.0, 3.0), (10, 0.7, 3.0), (10, 0.8, 2.5), (10, 0.9, 2.5)]

if __name__ == "__main__":
    for V, b, w in CANDIDATES:
        pot = double_barrier(V, b, w)
        res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(0.01, 12, 600001))
        desc = ", ".join(f"({r.E_res:.3f}, {r.fwhm:.1e})" for r in res)
        print(f"V={V:<3} b={b:<4} w={w:<4} -> (E_n, Gamma_n): {desc}")
