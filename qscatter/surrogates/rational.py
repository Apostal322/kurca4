"""Rational (AAA) approximation of the transmission amplitude.

Physics motivation: for a finite-range potential t(k) is meromorphic in the complex
wavenumber k, and resonances are its poles k_n = k_r - i gamma just below the real
axis (Breit–Wigner form near each one). A rational approximant in k can therefore
represent a narrow resonance with a single pole, where polynomial or kernel
methods need many samples across the peak.

Algorithm: AAA (Nakatsukasa, Sète, Trefethen, SIAM J. Sci. Comput. 40 (2018)
A1494), as implemented in ``scipy.interpolate.AAA``. Adaptive sampling with a
rational surrogate follows the greedy idea in the code of Pradovera, Hiptmair &
Perugia (J. Sci. Comput. 2025; github.com/pradovera/gMRI): sample next where the
barycentric denominator of the current approximant is smallest. Their
minimal-rational-interpolation weights are not identifiable for a scalar output,
so AAA's least-squares weights are used instead (our adaptation).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import AAA


class RationalAAA:
    """fit(E, Y) / predict(E) baseline; Y columns are (Re t, Im t) or (T,).

    For the amplitude target the complex samples t(k_j), k = sqrt(E), are
    approximated as one complex function of k. For the T target T(k) is fitted
    directly (real data, same algorithm), which lets us compare the amplitude
    and the observable as rational-approximation targets.
    """

    name = "aaa"

    def __init__(self, rtol: float = 1e-13, max_terms: int = 100):
        self.rtol = rtol
        self.max_terms = max_terms

    def fit(self, E, Y):
        Y = np.atleast_2d(np.asarray(Y).T).T
        k = np.sqrt(np.asarray(E, float))
        f = Y[:, 0] + 1j * Y[:, 1] if Y.shape[1] == 2 else Y[:, 0].astype(complex)
        self.n_out = Y.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.r = AAA(k, f, rtol=self.rtol, max_terms=min(self.max_terms, len(k)))
        return self

    def predict(self, E):
        z = self.r(np.sqrt(np.asarray(E, float)))
        return np.column_stack([z.real, z.imag]) if self.n_out == 2 else z.real[:, None]

    def denominator(self, E):
        """|D(k)| of the barycentric form N(k)/D(k), D(k) = sum_j w_j / (k - k_j)."""
        k = np.sqrt(np.asarray(E, float))[:, None]
        zj, wj = self.r.support_points, self.r.weights
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.sum(wj / (k - zj), axis=1)
        return np.abs(D)

    def poles(self):
        """Poles in the complex E plane (E = k^2), spurious doublets removed."""
        p, res = self.r.poles(), self.r.residues()
        keep = np.abs(res) > 1e-10 * max(1.0, np.max(np.abs(res)))
        return p[keep] ** 2
