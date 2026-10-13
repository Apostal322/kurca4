"""Reference scattering solver for piecewise-constant potentials.

Method
------
Each uniform layer j (height V_j, width d_j) is described by its exact
(psi, psi') propagator

    M_j = [[cos(q d),      sin(q d)/q],
           [-q sin(q d),   cos(q d)  ]],     q^2 = E - V_j.

All entries are entire functions of q^2, so the formula is regular at the
classical turning point E = V_j and valid for evanescent layers (E < V_j)
without choosing a branch of the square root.

M_j is converted to a 2x2 scattering matrix of that single layer embedded in
the asymptotic medium (wave number k = sqrt(E)). The layer S-matrices are then
combined with the Redheffer star product. Unlike multiplying transfer matrices
across the whole structure, the star product only ever contains bounded
quantities (|S_ij| <= ~1 for real V), so evanescent growth factors e^{kappa d}
from different layers are never multiplied together. Inside a single layer
cos/sin(q d) can grow like e^{kappa d}/2; they appear only in ratios, which is
accurate until overflow at kappa*d ~ 700 (a validity check raises before that).

Conventions
-----------
Incident wave from the left, common coordinate x:

    psi(x) = exp(i k x) + r exp(-i k x),   x < x_left
    psi(x) = t exp(i k x),                  x > x_right

so r and t are the "global" amplitudes, with T = |t|^2, R = |r|^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potentials import PiecewiseConstantPotential

MAX_KAPPA_D = 600.0  # cosh(600) ~ 1e260, safely below float overflow


@dataclass
class ScatteringResult:
    """Scattering data on an array of energies (all arrays have the same shape)."""

    E: np.ndarray
    k: np.ndarray
    t: np.ndarray  # complex transmission amplitude
    r: np.ndarray  # complex reflection amplitude

    @property
    def T(self) -> np.ndarray:
        return np.abs(self.t) ** 2

    @property
    def R(self) -> np.ndarray:
        return np.abs(self.r) ** 2

    @property
    def unitarity_defect(self) -> np.ndarray:
        return np.abs(self.T + self.R - 1.0)

    def as_dict(self) -> dict:
        return dict(E=self.E, k=self.k, t=self.t, r=self.r, T=self.T, R=self.R)


def _sin_over_q(q, d):
    # sin(q d)/q = d * sinc(q d / pi), regular at q = 0, valid for complex q
    return d * np.sinc(q * d / np.pi)


def _layer_smatrix(E: np.ndarray, k: np.ndarray, V: float, d: float):
    """S-matrix of a single layer embedded in the k medium, local references.

    Returns (r, t, rp, tp): left reflection, left->right transmission,
    right reflection, right->left transmission. Amplitudes are referenced to the
    layer's own left/right edges.
    """
    q = np.sqrt(np.asarray(E - V, dtype=complex))  # any branch: only even functions of q used
    c = np.cos(q * d)
    s_over_q = _sin_over_q(q, d)
    q_s = (q * q) * s_over_q  # q sin(q d)
    m11, m12, m21, m22 = c, s_over_q, -q_s, c
    ik = 1j * k
    # derived from matching (1+r, ik(1-r)) -> M -> (t, ik t); det M = 1
    alpha = ik * m11 - m21
    beta = -(k * k) * m12 - ik * m22
    denom = alpha - beta
    t = 2.0 * ik / denom
    r = -(alpha + beta) / denom
    # incidence from the right: mirror the layer (symmetric single layer) => same values
    return r, t, r.copy(), t.copy()


def _star(A, B):
    """Redheffer star product of two scalar 2-port S-matrices (A on the left)."""
    rA, tA, rpA, tpA = A
    rB, tB, rpB, tpB = B
    inv = 1.0 / (1.0 - rpA * rB)
    t = tB * inv * tA
    r = rA + tpA * rB * inv * tA
    tp = tpA * inv * tpB
    rp = rpB + tB * rpA * inv * tpB
    return r, t, rp, tp


def _check_validity(E, potential: PiecewiseConstantPotential):
    if np.any(~np.isfinite(E)):
        raise ValueError("energies must be finite")
    if np.any(E <= 0):
        raise ValueError("scattering requires E > 0 (asymptotic potential is 0)")
    for V, d in zip(potential.heights, potential.widths):
        kappa = np.sqrt(np.maximum(np.real(V) - E, 0.0))
        if np.any(kappa * d > MAX_KAPPA_D):
            raise ValueError(
                f"layer with V={V}, d={d} is too opaque (kappa*d > {MAX_KAPPA_D}); "
                "transmission underflows double precision"
            )


def compute_scattering(E, potential: PiecewiseConstantPotential) -> ScatteringResult:
    """Exact (up to round-off) t(E), r(E) for a piecewise-constant potential."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    _check_validity(E, potential)
    k = np.sqrt(E)
    S = None
    for V, d in zip(potential.heights, potential.widths):
        Vc = complex(V) if np.iscomplexobj(V) else float(V)  # complex V = absorbing layer (Im V < 0)
        layer = _layer_smatrix(E, k, Vc, float(d))
        S = layer if S is None else _star(S, layer)
    r_loc, t_loc, _, _ = S
    xl, xr = potential.x_left, potential.x_right
    # local -> global phase convention (see module docstring)
    t = t_loc * np.exp(-1j * k * (xr - xl))
    r = r_loc * np.exp(2j * k * xl)
    return ScatteringResult(E=E, k=k, t=t, r=r)


def compute_transmission(E, potential: PiecewiseConstantPotential) -> np.ndarray:
    return compute_scattering(E, potential).T


def compute_energy_scan(energies, potential: PiecewiseConstantPotential) -> ScatteringResult:
    """Alias of :func:`compute_scattering` on a 1D grid (vectorised)."""
    return compute_scattering(np.asarray(energies, dtype=float), potential)


class CountingSolver:
    """Wraps the exact solver and counts the number of energy evaluations.

    Every surrogate experiment must obtain data through an instance of this class,
    so the reported "exact solver evaluation count" is measured, not assumed.
    """

    def __init__(self, potential: PiecewiseConstantPotential):
        self.potential = potential
        self.n_evals = 0

    def __call__(self, E) -> ScatteringResult:
        E = np.atleast_1d(np.asarray(E, dtype=float))
        self.n_evals += E.size
        return compute_scattering(E, self.potential)


def transmission_complex_energy(E, potential: PiecewiseConstantPotential) -> np.ndarray:
    """Analytic continuation t(E) for complex E (k = sqrt(E), principal branch).

    The same layer formulas hold for complex E because they are entire in q^2;
    only the asymptotic k = sqrt(E) carries the branch. With the principal
    branch, the physical sheet near the positive real axis is reached from
    Im E < 0 by continuity of Re k > 0, which is where resonance poles
    E_n = E_r - i Gamma/2 live (Im k < 0). Used only to locate poles exactly.
    """
    E = np.atleast_1d(np.asarray(E, dtype=complex))
    k = np.sqrt(E)
    S = None
    for V, d in zip(potential.heights, potential.widths):
        layer = _layer_smatrix(E, k, complex(V), float(d))
        S = layer if S is None else _star(S, layer)
    return S[1] * np.exp(-1j * k * (potential.x_right - potential.x_left))
