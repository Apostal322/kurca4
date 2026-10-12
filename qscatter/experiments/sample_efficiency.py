"""Experiment 1 — sample efficiency.

For each problem, sampling design, budget N, method and target, fit the
surrogate on N exact solver evaluations and score it on the dense reference
grid. Every (configuration, seed) is an independent row in the raw results.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import PROBLEMS
from ..data import make_training_set, reference_solution
from ..metrics import curve_metrics, resonance_metrics, window_rmse
from ..physics.resonances import find_resonances
from ..physics.smatrix import compute_transmission
from ..surrogates.baselines import BASELINES, T_from_prediction, make_baseline, targets_from_t


@dataclass
class Exp1Config:
    problems: tuple = ("single_barrier", "double_barrier")
    budgets: tuple = (10, 20, 40, 80, 160, 320, 640)
    mlp_budgets: tuple = (10, 20, 40, 80, 160, 320)
    samplings: tuple = ("uniform", "random")
    seeds: tuple = (0, 1, 2, 3, 4)
    mlp_seeds: tuple = (0, 1, 2)  # MLP training is ~100x costlier than the baselines
    methods: tuple = BASELINES + ("mlp",)
    targets: tuple = ("amplitude", "T")
    out_dir: str = "results/exp1"
    save_models_seed: int = 0  # MLP checkpoints saved for this seed
    n_workers: int = 4
    mlp_overrides: dict = field(default_factory=dict)


_REF_CACHE: dict = {}


def true_resonances(problem):
    pot = problem.potential()
    E = np.linspace(*problem.E_range, 200001)
    return find_resonances(lambda e: compute_transmission(e, pot), E)


def _ref(name):
    if name not in _REF_CACHE:
        P = PROBLEMS[name]
        _REF_CACHE[name] = (P, reference_solution(P), true_resonances(P))
    return _REF_CACHE[name]


def run_one(task) -> dict:
    import torch

    torch.set_num_threads(1)
    from ..surrogates.mlp import MLPConfig, MLPSurrogate

    name, sampling, seed, N, method, target, cfg = task
    P, ref, res = _ref(name)
    ts = make_training_set(P, N, sampling, seed)
    ds_path = Path(cfg.out_dir) / "datasets" / f"{name}_{sampling}_N{N}_s{seed}"
    if not ds_path.with_suffix(".csv").exists():
        ts.save(ds_path)
    Y = targets_from_t(ts.t, target)
    t0 = time.perf_counter()
    extra = {}
    if method == "mlp":
        model = MLPSurrogate(P.E_range, MLPConfig(seed=seed, **cfg.mlp_overrides)).fit(ts.E, Y)
        extra = dict(best_epoch=model.log.best_epoch, best_val=model.log.best_val,
                     final_train_loss=model.log.final_train_loss)
        if seed == cfg.save_models_seed:
            model.save(Path(cfg.out_dir) / "models" / f"{name}_{sampling}_N{N}_{target}_s{seed}.pt")
    else:
        model = make_baseline(method, P.E_range, seed=seed).fit(ts.E, Y)
        if method == "gp":
            extra = dict(kernels=" | ".join(model.kernels))
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    Yhat = model.predict(ref.E)
    pred_time = time.perf_counter() - t0
    T_hat, t_hat = T_from_prediction(Yhat, target)
    row = dict(problem=name, sampling=sampling, seed=seed, N=N, n_exact_evals=ts.n_evals,
               method=method, target=target, fit_time_s=fit_time, predict_time_s=pred_time,
               n_ref=ref.E.size, **extra)
    row.update(curve_metrics(T_hat, ref.T, t_hat, ref.t if t_hat is not None else None))
    if res:
        row.update(resonance_metrics(ref.E, T_hat, res))
        row.update(window_rmse(ref.E, T_hat, ref.T, res))
    return row


def build_tasks(cfg: Exp1Config):
    tasks = []
    for name, sampling, N, method, target in itertools.product(
        cfg.problems, cfg.samplings, cfg.budgets, cfg.methods, cfg.targets
    ):
        if method == "mlp" and N not in cfg.mlp_budgets:
            continue
        # deterministic designs + deterministic classical interpolants need one seed only
        deterministic = sampling == "uniform" and method in ("linear", "pchip", "cubic")
        seeds = cfg.seeds[:1] if deterministic else (cfg.mlp_seeds if method == "mlp" else cfg.seeds)
        for seed in seeds:
            tasks.append((name, sampling, seed, N, method, target, cfg))
    # longest jobs first for better load balancing
    tasks.sort(key=lambda t: (t[4] != "mlp", t[4] != "gp", -t[3]))
    return tasks


def run(cfg: Exp1Config | None = None):
    cfg = cfg or Exp1Config()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    for name in cfg.problems:  # store true resonances for the report
        P, ref, res = _ref(name)
        (out / f"resonances_{name}.json").write_text(json.dumps([r.to_dict() for r in res], indent=2))
    tasks = build_tasks(cfg)
    raw = out / "raw_results.jsonl"
    done = set()
    if raw.exists():  # resume support
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            done.add((r["problem"], r["sampling"], r["seed"], r["N"], r["method"], r["target"]))
    tasks = [t for t in tasks if t[:6] not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done)", flush=True)
    with Pool(cfg.n_workers) as pool, raw.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks)):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{i + 1}/{len(tasks)}] {row['problem']} {row['sampling']} N={row['N']} "
                      f"{row['method']}/{row['target']} rmse_T={row['rmse_T']:.2e}", flush=True)
    return raw
