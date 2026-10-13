"""Resonance detection and characterisation for T(E)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks


@dataclass
class Resonance:
    E_res: float  # peak position
    T_peak: float  # peak transmission
    fwhm: float  # full width at half maximum (Gamma)
    E_lo: float  # half-maximum crossings
    E_hi: float

    def to_dict(self):
        return asdict(self)


def find_resonances(
    T_func,
    E_grid: np.ndarray,
    min_height: float = 0.1,
    min_prominence: float = 0.05,
    refine: bool = True,
    xatol: float = 1e-12,
) -> list[Resonance]:
    """Locate peaks of T(E).

    ``T_func`` maps an energy array to T. Peaks are first located on ``E_grid``
    (which must be fine enough to contain at least one point per peak), then refined
    by bounded scalar maximisation and the half-maximum crossings are found with
    Brent's method, so positions/widths do not depend on the grid spacing.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    T = np.asarray(T_func(E_grid))
    idx, _ = find_peaks(T, height=min_height, prominence=min_prominence)
    out = []
    for i in idx:
        lo = E_grid[max(i - 1, 0)]
        hi = E_grid[min(i + 1, len(E_grid) - 1)]
        if refine:
            opt = minimize_scalar(
                lambda e: -float(T_func(np.array([e]))[0]),
                bounds=(lo, hi),
                method="bounded",
                options=dict(xatol=xatol),
            )
            E0, Tp = float(opt.x), float(-opt.fun)
        else:
            E0, Tp = float(E_grid[i]), float(T[i])
        half = Tp / 2

        def f(e):
            return float(T_func(np.array([e]))[0]) - half

        E_lo = _crossing(f, E0, E_grid, direction=-1)
        E_hi = _crossing(f, E0, E_grid, direction=+1)
        width = (E_hi - E_lo) if np.isfinite(E_lo) and np.isfinite(E_hi) else np.nan
        out.append(Resonance(E0, Tp, width, E_lo, E_hi))
    return out


def _crossing(f, E0, E_grid, direction):
    """Find the half-maximum crossing next to the peak E0 in the given direction."""
    step = max(1e-9, 1e-6 * abs(E0))
    lim = E_grid[0] if direction < 0 else E_grid[-1]
    a = E0
    while True:
        b = E0 + direction * step
        if (direction < 0 and b <= lim) or (direction > 0 and b >= lim):
            b = lim
            if f(b) > 0:
                return np.nan
        if f(b) <= 0:
            return brentq(f, min(a, b), max(a, b), xtol=1e-14, rtol=1e-14)
        if b == lim:
            return np.nan
        a = b
        step *= 1.5


def lorentzian(E, E0, gamma, Tp):
    """Breit–Wigner (Lorentzian) line shape with FWHM gamma."""
    return Tp * (gamma / 2) ** 2 / ((E - E0) ** 2 + (gamma / 2) ** 2)
