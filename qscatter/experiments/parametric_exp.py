"""Experiment 4 — parametric surrogate (E, V0, a, d) -> t.

Budgets B (exact solver calls, validation data included):
  scan design      n_cfg = B / 32 Sobol configurations x 32 uniform energies
                   (7/8 of the configurations train, 1/8 validate)
  tensor design    m^3 configurations on a tensor grid x 32 energies (m = 4, 6, 10)
  hybrid design    single-barrier tensor grid (E, V0, a)
Methods: tensor-cubic, RBF, GP (<= 2048 points), MLP, MLP + Fourier features,
POD + MLP, MLP with dissipative pretraining, hybrid S-matrix composition.
"""
from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import parametric as pm

OUT = Path("results/exp4")
BUDGETS = (2048, 8192, 32768)
N_E = 32
SEEDS = (0, 1, 2)
TENSOR_M = {2048: 4, 8192: 6, 32768: 10}
HYBRID = {2048: (32, 8, 8), 8192: (64, 12, 12), 32768: (128, 16, 16)}
FOURIER_SIGMAS = (0.5, 1.0, 2.0, 4.0)
W_DISSIPATION = 0.05  # absorbing well potential -iW for pretraining (broadens resonances by ~2W)


def _refs(kind):
    """Reference solutions of the test / OOD configurations (git-ignored cache).

    Rebuilt deterministically and serially if missing (e.g. in a fresh clone); serial so
    that it is safe to call from scripts without a ``__main__`` guard on Windows.
    """
    p = OUT / "refs" / f"{kind}.npz"
    if not p.exists():
        C = pm.test_configs() if kind == "test" else pm.ood_configs()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, refs=np.array([pm.config_reference(c) for c in C], dtype=object))
    z = np.load(p, allow_pickle=True)
    return list(z["refs"])


def build_refs():
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    for kind, C in (("test", pm.test_configs()), ("ood", pm.ood_configs())):
        p = OUT / "refs" / f"{kind}.npz"
        if p.exists():
            continue
        with Pool(4) as pool:
            refs = pool.map(pm.config_reference, list(C))
        np.savez(p, refs=np.array(refs, dtype=object))


def evaluate(predict_cfg, refs):
    out = []
    for ref in refs:
        t0 = time.perf_counter()
        t_hat = predict_cfg(ref["cfg"], ref["E"])
        dt = time.perf_counter() - t0
        m = pm.config_metrics(ref, np.abs(t_hat) ** 2)
        m.update(cfg=ref["cfg"], predict_time_s=dt, n_points=len(ref["E"]))
        out.append(m)
    return out


def _xcfg(cfg, E):
    return np.column_stack([E, np.full(len(E), cfg[0]), np.full(len(E), cfg[1]), np.full(len(E), cfg[2])])


def split(X, t, C, n_E):
    n_cfg = len(C)
    val = np.zeros(n_cfg, bool)
    val[7::8] = True  # every 8th Sobol configuration validates
    rows = np.repeat(val, n_E)
    return X[~rows], t[~rows], X[rows], t[rows], C[~val], C[val]


def run_task(args):
    import torch

    torch.set_num_threads(1)
    method, B, seed = args
    test, ood = _refs("test"), _refs("ood")
    t0 = time.perf_counter()
    info = {}
    if method == "tensor_cubic":
        m = TENSOR_M[B]
        axes_c = [np.linspace(pm.LO[i], pm.HI[i], m) for i in (1, 2, 3)]
        E = np.linspace(pm.LO[0], pm.HI[0], N_E)
        tg = np.empty((m, m, m, N_E), complex)
        n = 0
        for i, V0 in enumerate(axes_c[0]):
            for j, a in enumerate(axes_c[1]):
                for k, d in enumerate(axes_c[2]):
                    s = pm.CountingSolver(pm.pot(V0, a, d))
                    tg[i, j, k] = s(E).t
                    n += s.n_evals
        gen = time.perf_counter() - t0
        model = pm.TensorCubic().fit_grid(axes_c + [E], tg)
        pred = lambda cfg, E_: model.predict(_xcfg(cfg, E_))
        n_evals = n
    elif method == "hybrid_smatrix":
        model = pm.HybridSMatrix().fit(*HYBRID[B])
        gen = time.perf_counter() - t0
        pred = lambda cfg, E_: model.predict(_xcfg(cfg, E_))
        n_evals = model.n_evals
    else:
        n_cfg = B // N_E
        if method == "mlp_dirtl":  # half the budget lossy, half lossless, same geometries (as in the paper)
            Xl, tl, nl, C = pm.scan_dataset(n_cfg // 2, N_E, seed, W=W_DISSIPATION)
            X, t, n0, _ = pm.scan_dataset(n_cfg // 2, N_E, seed, configs=C)
            n_evals = nl + n0
        else:
            X, t, n_evals, C = pm.scan_dataset(n_cfg, N_E, seed)
        gen = time.perf_counter() - t0
        Xtr, ttr, Xv, tv, Ctr, Cv = split(X, t, C, N_E)
        if method == "rbf":
            model = pm.RBF().fit(X, t)
        elif method == "gp":
            model = pm.GPParam(seed=seed).fit(Xtr, ttr)
            info["gp_points_used"] = model.n_used
        elif method == "mlp":
            model = pm.MLPParam(seed=seed).fit(Xtr, ttr, Xv, tv)
            info["best_step"] = model.best_step
        elif method == "mlp_fourier":
            best = None
            for sg in FOURIER_SIGMAS:  # Fourier scale chosen on validation configurations only
                mm = pm.MLPParam(fourier_sigma=sg, seed=seed, steps=10000).fit(Xtr, ttr, Xv, tv)
                if best is None or mm.best_val < best[0]:
                    best = (mm.best_val, sg)
            model = pm.MLPParam(fourier_sigma=best[1], seed=seed).fit(Xtr, ttr, Xv, tv)
            info["fourier_sigma"] = best[1]
        elif method == "mlp_dirtl":
            Xltr, tltr, Xlv, tlv, _, _ = split(Xl, tl, C, N_E)
            pre = pm.MLPParam(seed=seed, steps=10000).fit(Xltr, tltr, Xlv, tlv)
            model = pm.MLPParam(seed=seed, steps=10000, lr=5e-4, init_state=pre.state()).fit(Xtr, ttr, Xv, tv)
        elif method == "pod_mlp":
            Eg = np.linspace(pm.LO[0], pm.HI[0], N_E)
            Tm = t.reshape(len(C), N_E)
            val = np.zeros(len(C), bool)
            val[7::8] = True
            model = pm.PODMLP(seed=seed).fit_scans(C[~val], Eg, Tm[~val], C[val], Tm[val])
            info["pod_rank"] = model.r
            # compression error: best POD representation of the test scans on the training grid
            comp = []
            for ref in test:
                tt = pm.compute_scattering(Eg, pm.pot(*ref["cfg"])).t
                S = np.concatenate([tt.real, tt.imag]) - model.mean
                rec = model._scan_from_coef(S @ model.V.T)
                comp.append(float(np.sqrt(np.mean(np.abs(np.abs(rec) ** 2 - np.abs(tt) ** 2) ** 2))))
            info["pod_compression_rmse_T_on_grid_median"] = float(np.median(comp))
        else:
            raise ValueError(method)
        if method == "pod_mlp":
            pred = lambda cfg, E_: model.predict_config(cfg, E_)
        else:
            pred = lambda cfg, E_: model.predict(_xcfg(cfg, E_))
    fit_time = time.perf_counter() - t0 - gen
    rows = []
    for split_name, refs in (("test", test), ("ood", ood)):
        for m in evaluate(pred, refs):
            m.update(method=method, budget=B, seed=seed, n_evals=n_evals, split=split_name,
                     gen_time_s=gen, fit_time_s=fit_time, **info)
            rows.append(m)
    return rows


def run(n_workers=4, methods=None):
    OUT.mkdir(parents=True, exist_ok=True)
    build_refs()
    methods = methods or ("tensor_cubic", "hybrid_smatrix", "rbf", "gp", "mlp", "mlp_fourier", "pod_mlp", "mlp_dirtl")
    tasks = []
    for m in methods:
        for B in BUDGETS:
            if m == "gp" and B > 8192:
                continue  # exact GP on a 2048-point subset: larger budgets add no information to it
            seeds = (0,) if m in ("tensor_cubic", "hybrid_smatrix", "rbf") else SEEDS
            tasks += [(m, B, s) for s in seeds]
    raw = OUT / "raw_results.jsonl"
    done = set()
    if raw.exists():
        for l in raw.read_text().splitlines():
            r = json.loads(l)
            done.add((r["method"], r["budget"], r["seed"]))
    tasks = [t for t in tasks if t not in done]
    tasks.sort(key=lambda t: (t[0] not in ("mlp_fourier", "gp"), -t[1]))
    print(f"{len(tasks)} tasks", flush=True)
    with Pool(n_workers) as pool, raw.open("a") as fh:
        for rows in pool.imap_unordered(run_task, tasks):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            r = rows[0]
            med = np.median([x["rmse_T"] for x in rows if x["split"] == "test"])
            print(f"{r['method']} B={r['budget']} s={r['seed']} n_evals={r['n_evals']} median test RMSE_T={med:.2e} "
                  f"fit {r['fit_time_s']:.0f}s", flush=True)
