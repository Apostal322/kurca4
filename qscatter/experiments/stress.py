"""Experiment 5 — independent stress test on genuinely unseen potentials.

All systems below use a random seed (20261013) and families (asymmetric barriers,
coupled wells, smooth potentials, wide intervals) that were never used to choose or
tune any method. Every method gets the same budget N of exact solver evaluations.

Scoring is peak-relative so that it is meaningful for asymmetric resonances
(T_peak < 1): a true resonance (exact pole E_r - i Gamma/2 whose real-axis peak is
a local maximum of T) counts as *reconstructed* if max |T_hat - T| over
|E - E_r| <= 2 Gamma is <= 10 % of its peak height. Spurious peaks: local maxima
of T_hat with prominence > 0.05 that lie farther than 3 Gamma from every true pole.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import find_peaks

from ..physics.poles import find_poles
from ..physics.potentials import (asymmetric_double_barrier, double_barrier, gaussian_double_barrier_V, single_barrier, staircase, triple_barrier)
from ..physics.smatrix import compute_scattering
from ..surrogates.rational import RationalAAA
from .adaptive import _candidates, _segment_acquire

OUT = Path("results/exp5")
BUDGETS = (20, 40, 80, 160)
ML_BUDGETS = (40, 160)
SEED = 20261013


# ------------------------------------------------------------------ systems
@dataclass
class StressSystem:
    key: str
    family: str
    kind: str  # asym | triple | smooth | sym
    p: dict
    E_range: tuple
    noise: float = 0.0
    ml: bool = False  # include the (expensive) neural methods

    def potential(self):
        if self.kind == "asym":
            return asymmetric_double_barrier(self.p["VL"], self.p["aL"], self.p["d"], self.p["VR"], self.p["aR"])
        if self.kind == "triple":
            return triple_barrier(self.p["V0"], self.p["a_out"], self.p["a_mid"], self.p["d"])
        if self.kind == "sym":
            return double_barrier(self.p["V0"], self.p["a"], self.p["d"])
        if self.kind == "smooth":
            L = self.p["c"] + 8 * self.p["s"]
            return staircase(gaussian_double_barrier_V(self.p["V0"], self.p["c"], self.p["s"]), -L, L, 4000,
                             name="gaussian_double_barrier", params=self.p)
        raise ValueError(self.kind)

    def blocks(self):
        """Barrier blocks and free gaps for the hybrid S-matrix model (None if no decomposition)."""
        if self.kind == "asym":
            return [("barrier", self.p["VL"], self.p["aL"]), ("gap", self.p["d"]),
                    ("barrier", self.p["VR"], self.p["aR"])]
        if self.kind == "triple":
            q = self.p
            return [("barrier", q["V0"], q["a_out"]), ("gap", q["d"]), ("barrier", q["V0"], q["a_mid"]),
                    ("gap", q["d"]), ("barrier", q["V0"], q["a_out"])]
        if self.kind == "sym":
            return [("barrier", self.p["V0"], self.p["a"]), ("gap", self.p["d"]), ("barrier", self.p["V0"], self.p["a"])]
        return None


def build_systems():
    rng = np.random.default_rng(SEED)
    S = []
    for i in range(15):
        VL, VR = rng.uniform(6, 14, 2)
        aL, aR = rng.uniform(0.4, 1.2, 2)
        d = rng.uniform(1.5, 3.5)
        S.append(StressSystem(f"asym_{i:02d}", "asymmetric double barrier", "asym",
                              dict(VL=round(VL, 3), aL=round(aL, 3), d=round(d, 3), VR=round(VR, 3), aR=round(aR, 3)),
                              (0.1, round(min(VL, VR), 3)), ml=i < 4))
    # coupled wells: the middle barrier controls the doublet splitting relative to the width
    for a_mid in (0.5, 0.9, 1.3, 1.7):
        S.append(StressSystem(f"triple_amid{a_mid}", "triple barrier (resonance doublets)", "triple",
                              dict(V0=10.0, a_out=0.7, a_mid=a_mid, d=2.0), (0.1, 10.0), ml=a_mid in (0.9, 1.7)))
    S.append(StressSystem("smooth_gauss_A", "smooth double-Gaussian barrier", "smooth",
                          dict(V0=8.0, c=1.6, s=0.35), (0.1, 8.0), ml=True))
    S.append(StressSystem("smooth_gauss_B", "smooth double-Gaussian barrier", "smooth",
                          dict(V0=12.0, c=1.8, s=0.4), (0.1, 12.0), ml=True))
    for hi in (30.0, 60.0):  # far above the barrier top: many broad, overlapping resonances
        S.append(StressSystem(f"wide_0.1_{hi:g}", "wide interval (above-barrier resonances)", "sym",
                              dict(V0=10.0, a=0.8, d=2.5), (0.1, hi), ml=hi == 30.0))
    for sig in (1e-8, 1e-4):  # numerical noise on unseen asymmetric systems
        for j in (0, 1):
            base = S[j]
            S.append(StressSystem(f"{base.key}_noise{sig:g}", f"asymmetric + noise {sig:g}", "asym", base.p,
                                  base.E_range, noise=sig))
    return S


# ------------------------------------------------------------------ reference
def reference(sysm: StressSystem):
    pot = sysm.potential()
    lo, hi = sysm.E_range
    poles = find_poles(pot, sysm.E_range, n_scan=8001)
    grids = [np.linspace(lo, hi, 20001)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(lo, z.real - 20 * G), min(hi, z.real + 20 * G), 2001))
    E = np.unique(np.concatenate(grids))
    t = compute_scattering(E, pot).t
    T = np.abs(t) ** 2
    res = []
    for z in poles:
        G = -2 * z.imag
        w = np.abs(E - z.real) <= 2 * G
        if not w.any():
            continue
        j = np.argmax(np.where(w, T, -1))
        # count as a resonance if T has a local maximum near the pole (peak visible on the real axis)
        if 0 < j < len(E) - 1 and T[j] >= T[j - 1] and T[j] >= T[j + 1] and G < 0.5 * (hi - lo):
            # FWHM measured on the real axis for the pole-vs-FWHM comparison
            half = T[j] / 2
            left = np.where((E < E[j]) & (T < half))[0]
            right = np.where((E > E[j]) & (T < half))[0]
            fwhm = (E[right[0]] - E[left[-1]]) if len(left) and len(right) else np.nan
            res.append(dict(E_pole=float(z.real), gamma_pole=float(G), E_peak=float(E[j]), T_peak=float(T[j]),
                            fwhm=float(fwhm)))
    return dict(E=E, t=t, T=T, poles=[complex(z) for z in poles], res=res)


def ref_path(sysm):
    return OUT / "refs" / f"{sysm.key}.npz"


def build_reference(sysm):
    p = ref_path(sysm)
    if p.exists():
        return sysm.key
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    r = reference(sysm)
    np.savez(p, E=r["E"], t=r["t"], T=r["T"], poles=np.array(r["poles"]), res=json.dumps(r["res"]),
             time=time.perf_counter() - t0)
    return sysm.key


def load_reference(sysm):
    z = np.load(ref_path(sysm))
    return dict(E=z["E"], t=z["t"], T=z["T"], poles=list(z["poles"]), res=json.loads(str(z["res"])))


# ------------------------------------------------------------------ oracle
class Oracle:
    def __init__(self, sysm, seed=0):
        self.s, self.pot, self.n = sysm, sysm.potential(), 0
        self.rng = np.random.default_rng(seed)

    def __call__(self, E):
        E = np.atleast_1d(E)
        self.n += E.size
        t = compute_scattering(E, self.pot).t
        if self.s.noise:
            t = t * (1 + self.s.noise * (self.rng.standard_normal(E.size) + 1j * self.rng.standard_normal(E.size)) / np.sqrt(2))
        return t


# ------------------------------------------------------------------ scoring
def score(ref, T_hat, E_range):
    E, T = ref["E"], ref["T"]
    out = dict(rmse_T=float(np.sqrt(np.trapezoid((T_hat - T) ** 2, E) / (E[-1] - E[0]))),
               max_err_T=float(np.max(np.abs(T_hat - T))),
               frac_unphysical=float(np.mean((T_hat < -1e-6) | (T_hat > 1 + 1e-6))))
    n_ok = 0
    for r in ref["res"]:
        w = np.abs(E - r["E_pole"]) <= 2 * r["gamma_pole"]
        n_ok += bool(np.max(np.abs(T_hat[w] - T[w])) <= 0.1 * r["T_peak"])
    out.update(n_res=len(ref["res"]), n_reconstructed=n_ok)
    pk, _ = find_peaks(T_hat, prominence=0.05)
    out["n_spurious_peaks"] = int(sum(all(abs(E[i] - z.real) > 3 * max(-2 * z.imag, 1e-9) for z in ref["poles"]) for i in pk))
    return out


def aaa_diagnostics(r: RationalAAA, ref, E_range, data_E, data_t):
    """Classify the approximant's poles against the exact poles."""
    lo, hi = E_range
    k_lo, k_hi = np.sqrt(lo), np.sqrt(hi)
    kp, res = r.r.poles(), r.r.residues()
    zeros = r.r.roots()
    scale = np.max(np.abs(data_t))
    exact = np.array(ref["poles"])
    true_res = [x for x in ref["res"]]
    d = dict(n_poles_total=len(kp), n_matched=0, n_froissart=0, n_spurious_near_axis=0, n_upper_half=0,
             n_missed_resonances=0, pole_E_rel_err=[], pole_gamma_rel_err=[])
    for z, a in zip(kp, res):
        in_window = k_lo - 0.1 <= z.real <= k_hi + 0.1
        if not in_window:
            continue
        Ez = z**2
        # pole-zero cancellation: a zero closer to the pole than 1e-3 of the pole's distance to the axis
        near_zero = len(zeros) > 0 and np.min(np.abs(zeros - z)) < 1e-3 * max(abs(z.imag), 1e-12)
        if abs(a) < 1e-10 * scale or near_zero:
            d["n_froissart"] += 1
            continue
        if z.imag > 0:
            d["n_upper_half"] += 1  # non-physical: t(k) has no resonance poles for Im k > 0
            continue
        if len(exact):
            j = np.argmin(np.abs(exact - Ez))
            G = -2 * exact[j].imag
            if abs(Ez - exact[j]) <= 0.1 * G + 1e-10:
                d["n_matched"] += 1
                continue
        # unmatched: is it close enough to the real axis to create a visible artefact?
        if -Ez.imag < 0.05 * (hi - lo) and lo <= Ez.real <= hi:
            d["n_spurious_near_axis"] += 1
    for rr in true_res:
        Ep = rr["E_pole"] - 0.5j * rr["gamma_pole"]
        Ez = kp**2
        if len(Ez) == 0 or np.min(np.abs(Ez - Ep)) > 0.1 * rr["gamma_pole"] + 1e-10:
            d["n_missed_resonances"] += 1
        else:
            j = np.argmin(np.abs(Ez - Ep))
            d["pole_E_rel_err"].append(float(abs(Ez[j].real - Ep.real) / rr["gamma_pole"]))
            d["pole_gamma_rel_err"].append(float(abs(-2 * Ez[j].imag - rr["gamma_pole"]) / rr["gamma_pole"]))
    # numerical stability: refit on data perturbed by 1e-10 (relative) and compare predictions
    rng = np.random.default_rng(1)
    tp = data_t * (1 + 1e-10 * rng.standard_normal(data_t.size))
    r2 = RationalAAA(rtol=r.rtol).fit(data_E, np.column_stack([tp.real, tp.imag]))
    Y1, Y2 = r.predict(ref["E"]), r2.predict(ref["E"])
    d["perturbation_sensitivity"] = float(np.max(np.abs((Y1[:, 0] ** 2 + Y1[:, 1] ** 2) - (Y2[:, 0] ** 2 + Y2[:, 1] ** 2))))
    return d


# ------------------------------------------------------------------ hybrid (block S-matrices, 1D)
def hybrid_predict(sysm, N, E_eval):
    """Learn each distinct barrier's local (t, r) on E with a cubic spline, compose exactly.

    Budget: N block evaluations in total, split evenly over the distinct barriers.
    """
    blocks = sysm.blocks()
    distinct = sorted({(b[1], b[2]) for b in blocks if b[0] == "barrier"})
    n_each = max(4, N // len(distinct))
    lo, hi = sysm.E_range
    Eg = np.linspace(lo, hi, n_each)
    k = np.sqrt(Eg)
    fits = {}
    for V, a in distinct:
        s = compute_scattering(Eg, single_barrier(V, a))
        tl, rl = s.t * np.exp(1j * k * a), s.r * np.exp(1j * k * a)
        fits[(V, a)] = tuple(CubicSpline(Eg, z) for z in (tl.real, tl.imag, rl.real, rl.imag))
    kE = np.sqrt(E_eval)
    S = None
    total = 0.0
    for b in blocks:
        if b[0] == "gap":
            P = np.exp(1j * kE * b[1])
            layer = (np.zeros_like(P), P, np.zeros_like(P), P)
            total += b[1]
        else:
            f = fits[(b[1], b[2])]
            t_ = f[0](E_eval) + 1j * f[1](E_eval)
            r_ = f[2](E_eval) + 1j * f[3](E_eval)
            layer = (r_, t_, r_, t_)
            total += b[2]
        if S is None:
            S = layer
        else:
            rA, tA, rpA, tpA = S
            rB, tB, rpB, tpB = layer
            inv = 1 / (1 - rpA * rB)
            S = (rA + tpA * rB * inv * tA, tB * inv * tA, rpB + tB * rpA * inv * tpB, tpA * inv * tpB)
    return S[1] * np.exp(-1j * kE * total), n_each * len(distinct)


# ------------------------------------------------------------------ methods
def fit_predict(method, sysm, N, ref, seed=0):
    """Returns (T_hat on ref grid, info dict). Timing is measured per stage."""
    lo, hi = sysm.E_range
    info = {}
    o = Oracle(sysm, seed=seed)
    t_solve = t_acq = 0.0
    if method in ("aaa", "cubic_T", "gp", "mlp", "mlp_fourier"):
        E = np.linspace(lo, hi, N)
        t0 = time.perf_counter()
        t = o(E)
        t_solve = time.perf_counter() - t0
    elif method == "segment_t_pchip":
        E = np.linspace(lo, hi, 10)
        t0 = time.perf_counter()
        t = o(E)
        t_solve += time.perf_counter() - t0
        while E.size < N:
            t1 = time.perf_counter()
            e = _segment_acquire(E, np.column_stack([t.real, t.imag]), 1, sysm.E_range)
            t_acq += time.perf_counter() - t1
            t1 = time.perf_counter()
            tn = o(e)
            t_solve += time.perf_counter() - t1
            E, t = np.append(E, e), np.append(t, tn)
        idx = np.argsort(E)
        E, t = E[idx], t[idx]
    elif method == "aaa_greedy":
        E = np.linspace(lo, hi, 10)
        t0 = time.perf_counter()
        t = o(E)
        t_solve += time.perf_counter() - t0
        streak, cert = 0, None
        tol = max(1e-8, 10 * sysm.noise)
        while E.size < N and cert is None:
            t1 = time.perf_counter()
            r = RationalAAA(rtol=max(1e-13, 3 * sysm.noise)).fit(E, np.column_stack([t.real, t.imag]))
            cand = _candidates(sysm.E_range, E, 8000)
            e = cand[int(np.argmin(r.denominator(cand)))]
            y = r.predict(np.array([e]))[0]
            t_acq += time.perf_counter() - t1
            t1 = time.perf_counter()
            tn = o(np.array([e]))
            t_solve += time.perf_counter() - t1
            streak = streak + 1 if abs(y[0] + 1j * y[1] - tn[0]) < tol * abs(tn[0]) else 0
            E, t = np.append(E, e), np.append(t, tn)
            if streak >= 3:
                cert = E.size
        info["self_certified_at"] = cert
        info["n_used"] = int(E.size)  # the greedy may stop below the budget
    elif method == "hybrid":
        if sysm.blocks() is None or sysm.noise:
            return None, {}
        t0 = time.perf_counter()
        t_hat, n_used = hybrid_predict(sysm, N, ref["E"])
        info.update(n_used=n_used, fit_s=time.perf_counter() - t0, predict_s=0.0, solve_s=0.0, acq_s=0.0,
                    note="block (single-barrier) evaluations; fit+predict timed together")
        return np.abs(t_hat) ** 2, info
    else:
        raise ValueError(method)

    t1 = time.perf_counter()
    if method in ("aaa", "aaa_greedy"):
        model = RationalAAA(rtol=max(1e-13, 3 * sysm.noise)).fit(E, np.column_stack([t.real, t.imag]))
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        Y = model.predict(ref["E"])
        T_hat = Y[:, 0] ** 2 + Y[:, 1] ** 2
        info["aaa_diag"] = aaa_diagnostics(model, ref, sysm.E_range, E, t)
    elif method == "cubic_T":
        f = CubicSpline(E, np.abs(t) ** 2)
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        T_hat = f(ref["E"])
    elif method == "segment_t_pchip":
        fr, fi = PchipInterpolator(E, t.real), PchipInterpolator(E, t.imag)
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        T_hat = fr(ref["E"]) ** 2 + fi(ref["E"]) ** 2
    elif method == "gp":
        from ..surrogates.baselines import GP

        m = GP(sysm.E_range).fit(E, np.column_stack([t.real, t.imag]))
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        Y = m.predict(ref["E"])
        T_hat = Y[:, 0] ** 2 + Y[:, 1] ** 2
    elif method in ("mlp", "mlp_fourier"):
        T_hat, fit_s, t2, extra = _mlp(method, E, t, sysm, ref, seed)
        info.update(extra)
    predict_s = time.perf_counter() - t2
    info.update(fit_s=fit_s, predict_s=predict_s, solve_s=t_solve, acq_s=t_acq)
    info.setdefault("n_used", int(N))
    return T_hat, info


def _mlp(method, E, t, sysm, ref, seed):
    """1D MLP on (Re t, Im t); fixed-length training (early stopping was shown not to help, Exp. 1 ablation).

    Fourier variant: input features [sin 2 pi B x, cos 2 pi B x], x in [-1, 1], B ~ N(0, sigma^2), 64 features;
    sigma chosen from {1, 3, 10, 30} on held-out interleaved training points (every 5th), then refit on all.
    """
    import torch
    from torch import nn

    torch.set_num_threads(1)
    lo, hi = sysm.E_range
    x_all = 2 * (E - lo) / (hi - lo) - 1
    Y = np.column_stack([t.real, t.imag])
    mu, sd = Y.mean(0), Y.std(0) + 1e-12

    def train(x, y, sigma, steps, xv=None, yv=None):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(1, 64, generator=g, dtype=torch.float64) * sigma if sigma else None
        feat = (lambda z: torch.cat([torch.sin(2 * np.pi * z @ B), torch.cos(2 * np.pi * z @ B)], 1)) if sigma else (lambda z: z)
        d_in = 128 if sigma else 1
        net = nn.Sequential(nn.Linear(d_in, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)).double()
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
        X = feat(torch.tensor(x[:, None]))
        Yt = torch.tensor((y - mu) / sd)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.mean((net(X) - Yt) ** 2)
            loss.backward()
            opt.step()
            sch.step()
        val = None
        if xv is not None:
            with torch.no_grad():
                val = torch.mean((net(feat(torch.tensor(xv[:, None]))) - torch.tensor((yv - mu) / sd)) ** 2).item()
        return net, feat, val

    t1 = time.perf_counter()
    extra = {}
    sigma = None
    if method == "mlp_fourier":
        vi = np.arange(1, len(E) - 1)[4::5]
        fi = np.setdiff1d(np.arange(len(E)), vi)
        best = None
        for sg in (1.0, 3.0, 10.0, 30.0):
            _, _, v = train(x_all[fi], Y[fi], sg, 3000, x_all[vi], Y[vi])
            if best is None or v < best[0]:
                best = (v, sg)
        sigma = best[1]
        extra["fourier_sigma"] = sigma
    net, feat, _ = train(x_all, Y, sigma, 10000)
    fit_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    xr = 2 * (ref["E"] - lo) / (hi - lo) - 1
    with torch.no_grad():
        P = net(feat(torch.tensor(xr[:, None]))).numpy() * sd + mu
    return P[:, 0] ** 2 + P[:, 1] ** 2, fit_s, t2, extra


METHODS = ("aaa", "aaa_greedy", "segment_t_pchip", "cubic_T", "gp", "hybrid", "mlp", "mlp_fourier")


def run_task(args):
    sysm, method, N = args
    warnings.filterwarnings("ignore")
    ref = load_reference(sysm)
    T_hat, info = fit_predict(method, sysm, N, ref)
    if T_hat is None:
        return None
    row = dict(system=sysm.key, family=sysm.family, noise=sysm.noise, method=method, N=N, **score(ref, T_hat, sysm.E_range))
    row.update({k: v for k, v in info.items()})
    return row


def run(n_workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    systems = build_systems()
    (OUT / "systems.json").write_text(json.dumps([asdict(s) for s in systems], indent=1))
    with Pool(n_workers) as p:
        list(p.imap_unordered(build_reference, systems))
    tasks = []
    for s in systems:
        for m in METHODS:
            for N in BUDGETS:
                if m in ("mlp", "mlp_fourier") and not (s.ml and N in ML_BUDGETS):
                    continue
                tasks.append((s, m, N))
    tasks.sort(key=lambda t: t[1] not in ("mlp_fourier", "mlp"))
    raw = OUT / "raw_results.jsonl"
    with Pool(n_workers) as p, raw.open("w") as fh:
        for i, row in enumerate(p.imap_unordered(run_task, tasks)):
            if row is not None:
                fh.write(json.dumps(row, default=float) + "\n")
                fh.flush()
            if i % 50 == 0:
                print(f"[{i + 1}/{len(tasks)}]", flush=True)
    # reference metadata (resonances, poles, FWHM) for the analysis
    meta = {}
    for s in systems:
        r = load_reference(s)
        meta[s.key] = dict(res=r["res"], poles=[[z.real, z.imag] for z in r["poles"]], family=s.family)
    (OUT / "references.json").write_text(json.dumps(meta, indent=1))
