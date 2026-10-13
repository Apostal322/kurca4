"""Experiment 3 — robustness of the adaptive-sampling and rational-approximation results.

Scrutinises two claims across many systems instead of one potential:
  (i)  amplitude-driven adaptive sampling (segment (E, t)) detects narrow
       resonances that T-driven sampling misses;
  (ii) rational (AAA) approximation of t(k) reconstructs resonances from few
       uniform samples.

Systems (each a symmetric double barrier):
  width scan      V0 = 10, d = 2.5, barrier width a in A_SCAN (Gamma_1 spans ~6 decades)
  random          20 random (V0, a, d) configurations
  intervals       Stage B with four different energy intervals
  conventions     Stage B with t_X(E) = t(E) exp(i k X) for reference points X
  noise           Stage B with relative complex noise added to every solver output

Reference resonances are exact poles of t(E) (qscatter.physics.poles), and the
reference RMSE is a trapezoid L2 norm on a composite grid (uniform grid plus
dense windows of +-20 Gamma around every pole), so even Gamma ~ 1e-7 is resolved.
Acquisition only sees the oracle's outputs at the requested energies.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..physics.poles import find_poles
from ..physics.potentials import double_barrier
from ..physics.smatrix import compute_scattering
from ..surrogates.rational import RationalAAA
from .adaptive import _candidates, _segment_acquire

CHECKPOINTS = (10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320)
A_SCAN = (0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4)
STRATEGIES = ("uniform", "segment_T", "segment_logT", "segment_t", "aaa_greedy")
RECONS = ("pchip_t", "aaa_t", "aaa_T", "cubicT")


# ------------------------------------------------------------------ systems
@dataclass
class System:
    key: str
    group: str
    V0: float
    a: float
    d: float
    E_range: tuple
    X: float = 0.0  # phase reference point: t_X = t exp(i k X)
    noise: float = 0.0  # relative complex Gaussian noise on each solver output

    def potential(self):
        return double_barrier(self.V0, self.a, self.d)


def build_systems() -> list[System]:
    out = [System(f"width_a{a}", "width", 10.0, a, 2.5, (0.1, 10.0)) for a in A_SCAN]
    rng = np.random.default_rng(12345)
    for i in range(20):
        V0, a, d = rng.uniform(6, 14), rng.uniform(0.4, 1.2), rng.uniform(1.5, 3.5)
        out.append(System(f"random_{i:02d}", "random", round(V0, 3), round(a, 3), round(d, 3), (0.1, round(V0, 3))))
    for rng_ in ((0.1, 10.0), (0.5, 6.0), (2.0, 10.0), (0.1, 20.0)):
        out.append(System(f"interval_{rng_[0]}_{rng_[1]}", "interval", 10.0, 0.8, 2.5, rng_))
    for X in (-2.05, 10.0, 50.0):  # -2.05: amplitude referenced at the left edge; 10, 50: extra winding
        out.append(System(f"convention_X{X}", "convention", 10.0, 0.8, 2.5, (0.1, 10.0), X=X))
    for s in (1e-10, 1e-6, 1e-3):
        out.append(System(f"noise_{s:g}", "noise", 10.0, 0.8, 2.5, (0.1, 10.0), noise=s))
    return out


class Oracle:
    """Counts calls; applies the phase convention and the noise model."""

    def __init__(self, sysm: System, seed: int = 0):
        self.s, self.pot, self.n = sysm, sysm.potential(), 0
        self.rng = np.random.default_rng(seed)

    def __call__(self, E):
        E = np.atleast_1d(E)
        self.n += E.size
        t = compute_scattering(E, self.pot).t * np.exp(1j * np.sqrt(E) * self.s.X)
        if self.s.noise:
            t = t * (1 + self.s.noise * (self.rng.standard_normal(E.size) + 1j * self.rng.standard_normal(E.size)) / np.sqrt(2))
        return t


def reference(sysm: System):
    pot = sysm.potential()
    poles = find_poles(pot, sysm.E_range)
    lo, hi = sysm.E_range
    grids = [np.linspace(lo, hi, 20001)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(lo, z.real - 20 * G), min(hi, z.real + 20 * G), 2001))
    E = np.unique(np.concatenate(grids))
    T = np.abs(compute_scattering(E, pot).t) ** 2
    # resonance = pole whose real-axis peak of T exceeds 0.5 (others are reported, not scored)
    res = []
    for z in poles:
        G = -2 * z.imag
        w = (E > z.real - 3 * G) & (E < z.real + 3 * G)
        if w.any() and T[w].max() > 0.5:
            res.append(dict(E=float(z.real), gamma=float(G), T_peak=float(T[w].max())))
    return E, T, res


def l2_rmse(E, err):
    return float(np.sqrt(np.trapezoid(err**2, E) / (E[-1] - E[0])))


def detected(E, T_hat, res):
    """A resonance counts as found if T_hat > 0.5 somewhere within +-Gamma of E_r."""
    n = 0
    for r in res:
        w = np.abs(E - r["E"]) <= max(r["gamma"], 1e-12)
        n += bool(w.any() and T_hat[w].max() > 0.5)
    return n


# ------------------------------------------------------------------ acquisition
def acq_segment(E, t, n_new, dom, kind):
    if kind == "segment_T":
        Y = (np.abs(t) ** 2)[:, None]
    elif kind == "segment_logT":  # T on a log scale (decades), a fair T-based alternative
        Y = (np.log10(np.abs(t) ** 2 + 1e-300) / 1.0)[:, None]
    else:
        Y = np.column_stack([t.real, t.imag])
    return _segment_acquire(E, Y, n_new, dom)


LOOKAHEAD_TOL = 1e-8  # relative look-ahead error at which the rational greedy declares convergence


def acq_aaa_greedy(E, t, dom, state):
    """Greedy rational sampling (after the gMRI loop of Pradovera et al.).

    Next point = argmin of the barycentric denominator |D| on a candidate grid. The
    approximant's prediction at that point is stored; the caller compares it with the
    exact value (look-ahead error). After 3 consecutive look-ahead errors below
    LOOKAHEAD_TOL the method has certified itself; further budget is spent by
    bisecting the largest gap (exploration), so equal-budget comparisons remain defined.
    """
    if state.get("converged_at") is not None:
        state["pred"] = None
        return _bisect_largest(E, dom)
    r = RationalAAA().fit(E, np.column_stack([t.real, t.imag]))
    cand = _candidates(dom, E, 8000)
    e = cand[int(np.argmin(r.denominator(cand)))]
    y = r.predict(np.array([e]))[0]
    state["pred"] = y[0] + 1j * y[1]
    return np.array([e])


def _bisect_largest(E, dom):
    Es = np.sort(E)
    i = int(np.argmax(np.diff(Es)))
    return np.array([0.5 * (Es[i] + Es[i + 1])])


def reconstruct(name, E_tr, t_tr, E):
    if name == "pchip_t":
        return PchipInterpolator(E_tr, t_tr.real)(E) ** 2 + PchipInterpolator(E_tr, t_tr.imag)(E) ** 2
    if name == "cubicT":
        from scipy.interpolate import CubicSpline

        return CubicSpline(E_tr, np.abs(t_tr) ** 2)(E)
    if name == "aaa_t":
        Y = RationalAAA(rtol=_rtol(t_tr)).fit(E_tr, np.column_stack([t_tr.real, t_tr.imag])).predict(E)
        return Y[:, 0] ** 2 + Y[:, 1] ** 2
    if name == "aaa_T":
        return RationalAAA(rtol=_rtol(t_tr)).fit(E_tr, (np.abs(t_tr) ** 2)[:, None]).predict(E)[:, 0]
    raise ValueError(name)


_NOISE_RTOL = {}


def _rtol(t):
    return _NOISE_RTOL.get("rtol", 1e-13)


def _ref_path(out_dir, sysm):
    return Path(out_dir) / "refs" / f"{sysm.key}.npz"


def build_reference(args):
    sysm, out_dir = args
    t0 = time.perf_counter()
    E, T, res = reference(sysm)
    p = _ref_path(out_dir, sysm)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, E=E, T=T, res=json.dumps(res), time=time.perf_counter() - t0)
    return sysm.key


def run_design(args):
    sysm, strategy, out_dir = args
    z = np.load(_ref_path(out_dir, sysm))
    E_ref, T_ref, res, t_ref_time = z["E"], z["T"], json.loads(str(z["res"])), float(z["time"])
    # AAA tolerance must not be below the data accuracy (known noise level of the oracle)
    _NOISE_RTOL["rtol"] = max(1e-13, 3 * sysm.noise)
    oracle = Oracle(sysm)
    dom = sysm.E_range
    rows = []
    acq_time = 0.0
    state = {}
    if strategy == "uniform":
        designs = {}
        for N in CHECKPOINTS:
            o = Oracle(sysm)
            E = np.linspace(*dom, N)
            designs[N] = (E, o(E), o.n)
    else:
        designs = {}
        state = {}
        E = np.linspace(*dom, CHECKPOINTS[0])
        t = oracle(E)
        while True:
            if E.size in CHECKPOINTS:
                o = np.argsort(E)
                designs[E.size] = (E[o], t[o], oracle.n)
            if E.size >= CHECKPOINTS[-1]:
                break
            t1 = time.perf_counter()
            if strategy == "aaa_greedy":
                E_new = acq_aaa_greedy(E, t, dom, state)
            else:
                E_new = acq_segment(E, t, 1, dom, strategy)
            acq_time += time.perf_counter() - t1
            t_new = oracle(E_new)
            if strategy == "aaa_greedy" and state.get("pred") is not None:
                la = abs(state["pred"] - t_new[0]) / max(abs(t_new[0]), 1e-300)
                state["streak"] = state.get("streak", 0) + 1 if la < max(LOOKAHEAD_TOL, 10 * sysm.noise) else 0
                if state["streak"] >= 3:
                    state["converged_at"] = E.size + 1
            E, t = np.append(E, E_new), np.append(t, t_new)
    for N, (E_tr, t_tr, n_evals) in designs.items():
        assert n_evals == N
        for rec in RECONS:
            try:
                T_hat = reconstruct(rec, E_tr, t_tr, E_ref)
                ok = np.all(np.isfinite(T_hat))
            except Exception:
                T_hat, ok = np.full_like(E_ref, np.nan), False
            rows.append(dict(system=sysm.key, group=sysm.group, V0=sysm.V0, a=sysm.a, d=sysm.d,
                             E_lo=dom[0], E_hi=dom[1], X=sysm.X, noise=sysm.noise, strategy=strategy, recon=rec,
                             N=N, n_evals=n_evals,
                             rmse_T=l2_rmse(E_ref, T_hat - T_ref) if ok else float("inf"),
                             max_err_T=float(np.max(np.abs(T_hat - T_ref))) if ok else float("inf"),
                             n_res=len(res), n_detected=detected(E_ref, T_hat, res) if ok else 0,
                             gamma_min=min((r["gamma"] for r in res), default=np.nan),
                             n_in_narrowest=int(np.sum(np.abs(E_tr - min(res, key=lambda r: r["gamma"])["E"])
                                                       < 2 * min(r["gamma"] for r in res))) if res else 0,
                             acq_time_s=acq_time, ref_time_s=t_ref_time,
                             self_certified_at=state.get("converged_at")))
    meta = dict(system=sysm.__dict__, resonances=res)
    return rows, meta


def run(out_dir="results/exp3", n_workers=4, systems=None, strategies=STRATEGIES):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    systems = systems or build_systems()
    (out / "systems.json").write_text(json.dumps([s.__dict__ for s in systems], indent=1))
    with Pool(n_workers) as pool:
        list(pool.imap_unordered(build_reference, [(s, out_dir) for s in systems
                                                    if not _ref_path(out_dir, s).exists()]))
    tasks = [(s, st, out_dir) for s in systems for st in strategies]
    tasks.sort(key=lambda t: t[1] != "aaa_greedy")
    metas = {}
    with Pool(n_workers) as pool, (out / "raw_results.jsonl").open("w") as fh:
        for i, (rows, meta) in enumerate(pool.imap_unordered(run_design, tasks)):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            metas[meta["system"]["key"]] = meta
            if i % 20 == 0:
                print(f"[{i + 1}/{len(tasks)}]", flush=True)
    (out / "resonances.json").write_text(json.dumps(metas, indent=1, default=float))
