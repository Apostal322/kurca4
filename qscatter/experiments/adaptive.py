"""Experiment 2 — sequential data acquisition / adaptive sampling.

All strategies start from the same N0 uniformly spaced exact evaluations and then
request new energies one batch at a time from a ``CountingSolver``. Acquisition
decisions use only previously acquired solver data (never the reference grid).

Strategies
----------
uniform        non-adaptive: uniform design of size N (re-generated for each N)
random         non-adaptive: random design of size N (seeded)
segment_T      split the interval with the longest segment in the (E, T) plane
segment_t      split the interval with the longest segment in the (E, Re t, Im t)
               space — uses the complex amplitude, whose phase changes by ~pi
               across a resonance even when T at both ends is small
gp_std         maximise the GP posterior standard deviation of (Re t, Im t)
               (hyper-parameters refitted each batch; batch filled greedily with
               the "kriging believer" heuristic at fixed hyper-parameters)
mlp_ensemble   maximise the disagreement of an ensemble of PyTorch MLPs
               (deep-ensemble style; refitted each batch)

At checkpoints N the acquired design is scored with several reconstruction
methods (classical interpolants on the amplitude, GP, and optionally the MLP).
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import PROBLEMS
from ..data import TrainingSet, sample_energies
from ..metrics import curve_metrics, resonance_metrics, window_rmse
from ..physics.smatrix import CountingSolver
from ..surrogates.baselines import GP, T_from_prediction, make_baseline, targets_from_t
from .sample_efficiency import _ref


# --------------------------------------------------------------------------
# acquisition functions: (E, t, n_new, domain, rng) -> new energies
# --------------------------------------------------------------------------
def _segment_acquire(E, Y, n_new, domain, min_dx=1e-6):
    lo, hi = domain
    E = E.copy()
    Y = Y.copy()
    chosen = []
    for _ in range(n_new):
        order = np.argsort(E)
        Es, Ys = E[order], Y[order]
        dx = np.diff(Es) / (hi - lo)
        dy2 = np.sum(np.abs(np.diff(Ys, axis=0)) ** 2, axis=1)
        loss = np.sqrt(dx**2 + dy2)
        loss[dx < min_dx] = 0.0
        i = int(np.argmax(loss))
        e_new = 0.5 * (Es[i] + Es[i + 1])
        chosen.append(e_new)
        # no new data yet within a batch: predict the midpoint linearly so the same
        # interval is not chosen twice
        E = np.append(E, e_new)
        Y = np.vstack([Y, 0.5 * (Ys[i] + Ys[i + 1])])
    return np.array(chosen)


def acquire_segment_T(E, t, n_new, domain, rng, **kw):
    return _segment_acquire(E, (np.abs(t) ** 2)[:, None], n_new, domain)


def acquire_segment_t(E, t, n_new, domain, rng, **kw):
    return _segment_acquire(E, np.column_stack([t.real, t.imag]), n_new, domain)


def _candidates(domain, E, n_cand=4000):
    c = np.linspace(*domain, n_cand)
    spacing = (domain[1] - domain[0]) / (n_cand - 1)
    d = np.min(np.abs(c[:, None] - E[None, :]), axis=1)
    return c[d > 0.5 * spacing]


def acquire_gp_std(E, t, n_new, domain, rng, **kw):
    Y = np.column_stack([t.real, t.imag])
    gp = GP(domain, n_restarts=2).fit(E, Y)
    cand = _candidates(domain, E)
    chosen = []
    Ecur, Ycur = E.copy(), Y.copy()
    for _ in range(n_new):
        mu, sd = gp.predict(cand, return_std=True)
        score = np.sum(sd**2, axis=1)
        j = int(np.argmax(score))
        chosen.append(cand[j])
        # kriging believer: pretend the GP mean was observed, refit at fixed hyper-parameters
        Ecur = np.append(Ecur, cand[j])
        Ycur = np.vstack([Ycur, mu[j]])
        for m in gp.models:
            m.optimizer = None
        Xs = gp._x(Ecur)
        for c, m in enumerate(gp.models):
            m.kernel = m.kernel_
            m.fit(Xs, Ycur[:, c])
        cand = np.delete(cand, j)
    return np.array(chosen)


def acquire_mlp_ensemble(E, t, n_new, domain, rng, n_members=4, round_idx=0, **kw):
    from ..surrogates.mlp import MLPConfig, MLPSurrogate

    Y = np.column_stack([t.real, t.imag])
    cand = _candidates(domain, E)
    preds = []
    for m in range(n_members):
        cfg = MLPConfig(seed=1000 * round_idx + m, max_epochs=5000, patience=1000, refit=False)
        preds.append(MLPSurrogate(domain, cfg).fit(E, Y).predict(cand))
    P = np.stack(preds)  # (members, n_cand, 2)
    score = np.sum(P.var(axis=0), axis=1)
    chosen = []
    excl = 0.5 * (domain[1] - domain[0]) / max(len(E) + n_new, 1)  # spread the batch out
    for _ in range(n_new):
        j = int(np.argmax(score))
        chosen.append(cand[j])
        score[np.abs(cand - cand[j]) < excl] = -np.inf
    return np.array(chosen)


ACQUIRE = dict(
    segment_T=(acquire_segment_T, 1),
    segment_t=(acquire_segment_t, 1),
    gp_std=(acquire_gp_std, 5),
    mlp_ensemble=(acquire_mlp_ensemble, 10),
)


def run_strategy(problem, strategy, checkpoints, N0=10, seed=0):
    """Return {N: TrainingSet} for every N in ``checkpoints``."""
    P = problem
    pot = P.potential()
    solver = CountingSolver(pot)
    out = {}
    if strategy in ("uniform", "random"):
        for N in checkpoints:
            s = CountingSolver(pot)
            E = sample_energies(strategy, N, P.E_range, seed)
            out[N] = TrainingSet(E, s(E).t, P.name, pot.params, strategy, seed, s.n_evals)
        return out
    acquire, batch = ACQUIRE[strategy]
    rng = np.random.default_rng(seed)
    E = sample_energies("uniform", N0, P.E_range)
    t = solver(E).t
    round_idx = 0
    timings = []
    while True:
        N = E.size
        if N in checkpoints:
            order = np.argsort(E)
            out[N] = TrainingSet(E[order], t[order], P.name, pot.params, strategy, seed, solver.n_evals,
                                 meta=dict(acq_time_s=float(np.sum(timings))))
        if N >= max(checkpoints):
            break
        nxt = min(c for c in checkpoints if c > N)
        n_new = min(batch, nxt - N)
        t0 = time.perf_counter()
        E_new = acquire(E, t, n_new, P.E_range, rng, round_idx=round_idx)
        timings.append(time.perf_counter() - t0)
        E = np.append(E, E_new)
        t = np.append(t, solver(E_new).t)
        round_idx += 1
    assert solver.n_evals == E.size  # every point is one exact evaluation
    return out


# --------------------------------------------------------------------------
@dataclass
class Exp2Config:
    problems: tuple = ("single_barrier", "double_barrier")
    strategies: tuple = ("uniform", "random", "segment_T", "segment_t", "gp_std", "mlp_ensemble")
    checkpoints: tuple = (10, 20, 40, 80, 160, 320)
    mlp_ensemble_max: int = 160
    random_seeds: tuple = (0, 1, 2, 3, 4)
    recon_methods: tuple = ("linear", "pchip", "cubic", "gp")
    mlp_recon_N: tuple = (40, 80, 160)
    mlp_recon_seeds: tuple = (0, 1, 2)
    out_dir: str = "results/exp2"
    n_workers: int = 4


def _design_task(args):
    import torch

    torch.set_num_threads(1)
    name, strategy, seed, cfg = args
    P = PROBLEMS[name]
    cps = cfg.checkpoints
    if strategy == "mlp_ensemble":
        cps = tuple(c for c in cps if c <= cfg.mlp_ensemble_max)
    t0 = time.perf_counter()
    designs = run_strategy(P, strategy, cps, seed=seed)
    for N, ts in designs.items():
        ts.save(Path(cfg.out_dir) / "designs" / f"{name}_{strategy}_N{N}_s{seed}")
    return name, strategy, seed, time.perf_counter() - t0


def _score_task(args):
    import torch

    torch.set_num_threads(1)
    from ..surrogates.mlp import MLPConfig, MLPSurrogate

    name, strategy, seed, N, method, mseed, cfg = args
    P, ref, res = _ref(name)
    ts = TrainingSet.load(Path(cfg.out_dir) / "designs" / f"{name}_{strategy}_N{N}_s{seed}")
    Y = targets_from_t(ts.t, "amplitude")
    if method == "mlp":
        model = MLPSurrogate(P.E_range, MLPConfig(seed=mseed)).fit(ts.E, Y)
    else:
        model = make_baseline(method, P.E_range).fit(ts.E, Y)
    T_hat, t_hat = T_from_prediction(model.predict(ref.E), "amplitude")
    row = dict(problem=name, strategy=strategy, design_seed=seed, N=N, n_exact_evals=ts.n_evals,
               recon=method, model_seed=mseed, acq_time_s=ts.meta.get("acq_time_s", 0.0))
    row.update(curve_metrics(T_hat, ref.T, t_hat, ref.t))
    if res:
        row.update(resonance_metrics(ref.E, T_hat, res))
        row.update(window_rmse(ref.E, T_hat, ref.T, res))
    return row


def run(cfg: Exp2Config | None = None):
    cfg = cfg or Exp2Config()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    design_tasks = []
    for name, strat in itertools.product(cfg.problems, cfg.strategies):
        seeds = cfg.random_seeds if strat == "random" else (0,)
        design_tasks += [(name, strat, s, cfg) for s in seeds]
    design_tasks.sort(key=lambda t: t[1] != "mlp_ensemble")
    with Pool(cfg.n_workers) as pool:
        for name, strat, seed, dt in pool.imap_unordered(_design_task, design_tasks):
            print(f"design {name} {strat} s{seed}: {dt:.1f}s", flush=True)

    score_tasks = []
    for name, strat, seed, _ in design_tasks:
        cps = cfg.checkpoints if strat != "mlp_ensemble" else tuple(c for c in cfg.checkpoints if c <= cfg.mlp_ensemble_max)
        for N in cps:
            for m in cfg.recon_methods:
                score_tasks.append((name, strat, seed, N, m, 0, cfg))
            if N in cfg.mlp_recon_N and seed == 0:
                score_tasks += [(name, strat, seed, N, "mlp", ms, cfg) for ms in cfg.mlp_recon_seeds]
    score_tasks.sort(key=lambda t: t[4] != "mlp")
    raw = out / "raw_results.jsonl"
    with Pool(cfg.n_workers) as pool, raw.open("w") as fh:
        for i, row in enumerate(pool.imap_unordered(_score_task, score_tasks)):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"[score {i + 1}/{len(score_tasks)}]", flush=True)
    return raw
