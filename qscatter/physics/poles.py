"""Exact resonance poles of t(E) in the complex energy plane.

Candidates come from a rational (AAA) fit of t(k) on a dense real-axis scan; each
candidate is then refined by Newton's method on 1/t(E) using the *exact*
analytically continued solver, so the final pole positions do not depend on the
fit. Resonance: E_n = E_r - i Gamma_n / 2 (Gamma_n > 0).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import AAA

from .smatrix import compute_scattering, transmission_complex_energy


def refine_pole(E0: complex, potential, tol: float = 1e-14, maxit: int = 60) -> complex | None:
    f = lambda z: 1.0 / transmission_complex_energy(z, potential)[0]
    z = complex(E0)
    for _ in range(maxit):
        h = 1e-7 * max(abs(z.imag), 1e-6)
        fz = f(z)
        df = (f(z + h) - f(z - h)) / (2 * h)
        step = fz / df
        z -= step
        if abs(step) < tol * max(1.0, abs(z)):
            return z
    return None


def find_poles(potential, E_range, n_scan: int = 20001, chunk: float = 2.0) -> list[complex]:
    """Resonance poles with Re E in E_range (sorted by Re E)."""
    lo, hi = E_range
    found = []
    edges = np.arange(lo, hi + chunk, chunk)
    for a, b in zip(edges[:-1], edges[1:]):
        b = min(b, hi)
        E = np.linspace(max(a - 0.2, 1e-6), b + 0.2, n_scan)
        k = np.sqrt(E)
        r = AAA(k, compute_scattering(E, potential).t, rtol=1e-13)
        p, res = r.poles(), r.residues()
        for kp, rp in zip(p, res):
            if abs(rp) < 1e-14 or kp.imag >= 0:
                continue
            Ep = kp**2
            if not (a <= Ep.real < b) or Ep.imag > -1e-300:
                continue
            z = refine_pole(Ep, potential)
            if z is not None and lo <= z.real <= hi and z.imag < 0 and \
                    all(abs(z - q) > 1e-9 * max(1, abs(z)) for q in found):
                found.append(z)
    return sorted(found, key=lambda z: z.real)
