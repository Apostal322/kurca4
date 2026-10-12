"""Closed-form textbook results used to validate the numerical solver."""
from __future__ import annotations

import numpy as np


def single_barrier_T(E, V0: float, a: float) -> np.ndarray:
    """Textbook transmission probability of a rectangular barrier (hbar^2/2m = 1).

        E < V0:  T = 1 / (1 + V0^2 sinh^2(kappa a) / (4 E (V0 - E))),  kappa = sqrt(V0 - E)
        E = V0:  T = 1 / (1 + V0 a^2 / 4)
        E > V0:  T = 1 / (1 + V0^2 sin^2(q a)   / (4 E (E - V0))),     q = sqrt(E - V0)

    Written in the real form (sinh / sin branches) independently of the complex
    S-matrix algebra in smatrix.py.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    T = np.empty_like(E)
    below = E < V0
    above = E > V0
    at = ~(below | above)
    kap = np.sqrt(V0 - E[below])
    T[below] = 1.0 / (1.0 + V0**2 * np.sinh(kap * a) ** 2 / (4.0 * E[below] * (V0 - E[below])))
    q = np.sqrt(E[above] - V0)
    T[above] = 1.0 / (1.0 + V0**2 * np.sin(q * a) ** 2 / (4.0 * E[above] * (E[above] - V0)))
    if np.any(at):
        T[at] = 1.0 / (1.0 + V0 * a**2 / 4.0)
    return T


def single_barrier_t(E, V0: float, a: float) -> np.ndarray:
    """Complex transmission amplitude for a barrier occupying an interval of width a.

    t = exp(-i k a) / (cos(q a) - i (k^2 + q^2) / (2 k q) sin(q a)),  q = sqrt(E - V0)

    (Global convention psi_R = t e^{ikx}; independent of where the barrier sits.)
    Evaluated with complex q for E < V0. Not valid exactly at E = V0.
    """
    E = np.atleast_1d(np.asarray(E, dtype=float))
    k = np.sqrt(E)
    q = np.sqrt((E - V0).astype(complex))
    return np.exp(-1j * k * a) / (np.cos(q * a) - 1j * (k**2 + q**2) / (2 * k * q) * np.sin(q * a))


def infinite_well_levels(L: float, n_max: int) -> np.ndarray:
    """E_n = (n pi / L)^2 — crude estimate of quasi-bound levels in a double barrier."""
    n = np.arange(1, n_max + 1)
    return (n * np.pi / L) ** 2
