"""Reconstruction-quality metrics evaluated on the dense reference grid."""
from __future__ import annotations

import numpy as np

from .physics.resonances import Resonance, find_resonances

UNPHYSICAL_TOL = 1e-6  # T_hat outside [-tol, 1 + tol] counts as unphysical


def curve_metrics(T_hat, T_ref, t_hat=None, t_ref=None) -> dict:
    err = T_hat - T_ref
    m = dict(
        rmse_T=float(np.sqrt(np.mean(err**2))),
        max_err_T=float(np.max(np.abs(err))),
        rel_l2_T=float(np.linalg.norm(err) / np.linalg.norm(T_ref)),
        frac_unphysical=float(np.mean((T_hat < -UNPHYSICAL_TOL) | (T_hat > 1 + UNPHYSICAL_TOL))),
        min_T_hat=float(np.min(T_hat)),
        max_T_hat=float(np.max(T_hat)),
    )
    if t_hat is not None and t_ref is not None:
        m["rel_l2_t"] = float(np.linalg.norm(t_hat - t_ref) / np.linalg.norm(t_ref))
    return m


def resonance_metrics(E_ref, T_hat, true_res: list[Resonance]) -> dict:
    """Is each true resonance reproduced by the surrogate?

    A true resonance (E0, Gamma) counts as *detected* if the surrogate curve has a
    local maximum with T_hat > 0.5 within |E - E0| < Gamma. For detected
    resonances we report position error / Gamma, relative peak-height error and
    relative width error (surrogate width from its own half-maximum crossings on
    the reference grid).
    """
    interp = lambda e: np.interp(e, E_ref, T_hat)
    try:
        pred = find_resonances(interp, E_ref, min_height=0.5, min_prominence=0.05, refine=False)
    except Exception:
        pred = []
    out = dict(n_true=len(true_res), n_detected=0, n_spurious=0)
    used = set()
    for i, r in enumerate(true_res):
        cand = [(abs(p.E_res - r.E_res), j, p) for j, p in enumerate(pred) if abs(p.E_res - r.E_res) < r.fwhm]
        if cand:
            _, j, p = min(cand, key=lambda c: c[0])
            used.add(j)
            out["n_detected"] += 1
            out[f"res{i}_pos_err_over_gamma"] = abs(p.E_res - r.E_res) / r.fwhm
            out[f"res{i}_height_rel_err"] = abs(p.T_peak - r.T_peak) / r.T_peak
            out[f"res{i}_width_rel_err"] = abs(p.fwhm - r.fwhm) / r.fwhm if np.isfinite(p.fwhm) else np.nan
            out[f"res{i}_detected"] = 1
        else:
            out[f"res{i}_detected"] = 0
            for k in ("pos_err_over_gamma", "height_rel_err", "width_rel_err"):
                out[f"res{i}_{k}"] = np.nan
    out["n_spurious"] = len(pred) - len(used)
    return out


def window_rmse(E_ref, T_hat, T_ref, true_res: list[Resonance], n_gamma: float = 3.0) -> dict:
    """RMSE inside resonance windows |E - E0| < n_gamma * Gamma and outside them."""
    mask = np.zeros_like(E_ref, dtype=bool)
    for r in true_res:
        mask |= np.abs(E_ref - r.E_res) < n_gamma * r.fwhm
    err = T_hat - T_ref
    return dict(
        rmse_T_resonant=float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else np.nan,
        rmse_T_background=float(np.sqrt(np.mean(err[~mask] ** 2))) if (~mask).any() else np.nan,
    )
