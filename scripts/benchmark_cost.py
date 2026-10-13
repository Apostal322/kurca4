"""Task 5 — measured end-to-end computational cost (run on an otherwise idle machine).

Scenario 1 (energy scan, Stage B double barrier): produce T(E) on a 20 001-point grid.
  direct      : exact solver on every grid point
  surrogates  : N exact calls + acquisition + fit + inference on the grid
Both solvers are benchmarked: the vectorised S-matrix solver (fast, analytic in
structure) and the independent DOP853 ODE solver (slow, one energy per call). The
ODE solver is first checked against the S-matrix reference on the surrogate's
training points. Every time below is measured wall-clock (time.perf_counter);
nothing is extrapolated except where a row is explicitly labelled "projected".

Scenario 2 (parametric, from Experiment 4): construction cost (data + training)
vs. per-query cost, and the number of queries needed to amortise construction.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from qscatter.config import STAGE_B  # noqa: E402
from qscatter.experiments.adaptive import _segment_acquire  # noqa: E402
from qscatter.physics.ode_solver import solve_ode  # noqa: E402
from qscatter.physics.smatrix import compute_scattering  # noqa: E402
from qscatter.surrogates.mlp import MLPConfig, MLPSurrogate  # noqa: E402
from qscatter.surrogates.rational import RationalAAA  # noqa: E402

torch.set_num_threads(1)
OUT = Path("results/cost")
OUT.mkdir(parents=True, exist_ok=True)
pot = STAGE_B.potential()
GRID = np.linspace(*STAGE_B.E_range, 20001)
T_REF = np.abs(compute_scattering(GRID, pot).t) ** 2
W = STAGE_B.E_range[1] - STAGE_B.E_range[0]


def fast(E):
    return compute_scattering(np.atleast_1d(E), pot).t


def slow(E):
    return np.array([solve_ode(float(e), pot).t for e in np.atleast_1d(E)])


SOLVERS = {"S-matrix (vectorised)": fast, "ODE DOP853": slow}


def timed(f, *a):
    t0 = time.perf_counter()
    out = f(*a)
    return out, time.perf_counter() - t0


def rmse(T_hat):
    return float(np.sqrt(np.mean((T_hat - T_REF) ** 2)))


rows = []
for sname, solve in SOLVERS.items():
    # ---- direct scan
    if sname.startswith("ODE"):
        # full 20 001-point ODE scan is measured, not extrapolated (~1 min)
        t_direct, dt = timed(solve, GRID)
        agree = float(np.max(np.abs(t_direct - compute_scattering(GRID, pot).t)))
    else:
        t_direct, dt = timed(solve, GRID)
        agree = 0.0
    rows.append(dict(solver=sname, method="direct scan (20 001 calls)", n_calls=GRID.size, solver_s=dt,
                     acquisition_s=0.0, fit_s=0.0, inference_s=0.0, total_s=dt,
                     rmse_T=rmse(np.abs(t_direct) ** 2), note=f"max|t - t_Smatrix| = {agree:.1e}"))

    # ---- uniform N=30 + AAA(t)
    E = np.linspace(*STAGE_B.E_range, 30)
    t, s_t = timed(solve, E)
    r, f_t = timed(lambda: RationalAAA().fit(E, np.column_stack([t.real, t.imag])))
    Y, i_t = timed(r.predict, GRID)
    rows.append(dict(solver=sname, method="uniform 30 + AAA(t)", n_calls=30, solver_s=s_t, acquisition_s=0.0,
                     fit_s=f_t, inference_s=i_t, total_s=s_t + f_t + i_t, rmse_T=rmse(Y[:, 0] ** 2 + Y[:, 1] ** 2)))

    # ---- AAA greedy until self-certified (look-ahead error < 1e-8, 3 times)
    E = np.linspace(*STAGE_B.E_range, 10)
    t, s_t = timed(solve, E)
    acq = 0.0
    streak = 0
    cand = np.linspace(*STAGE_B.E_range, 8000)
    while streak < 3 and E.size < 200:
        t0 = time.perf_counter()
        r = RationalAAA().fit(E, np.column_stack([t.real, t.imag]))
        c = cand[np.min(np.abs(cand[:, None] - E[None, :]), axis=1) > 0.5 * (cand[1] - cand[0])]
        e = c[int(np.argmin(r.denominator(c)))]
        y = r.predict(np.array([e]))[0]
        acq += time.perf_counter() - t0
        tn, dt = timed(solve, e)
        s_t += dt
        streak = streak + 1 if abs(y[0] + 1j * y[1] - tn[0]) < 1e-8 * abs(tn[0]) else 0
        E, t = np.append(E, e), np.append(t, tn)
    r, f_t = timed(lambda: RationalAAA().fit(E, np.column_stack([t.real, t.imag])))
    Y, i_t = timed(r.predict, GRID)
    rows.append(dict(solver=sname, method=f"AAA greedy (self-certified at N={E.size})", n_calls=E.size, solver_s=s_t,
                     acquisition_s=acq, fit_s=f_t, inference_s=i_t, total_s=s_t + acq + f_t + i_t,
                     rmse_T=rmse(Y[:, 0] ** 2 + Y[:, 1] ** 2)))

    # ---- segment(E,t) to N=160 + PCHIP
    E = np.linspace(*STAGE_B.E_range, 10)
    t, s_t = timed(solve, E)
    acq = 0.0
    while E.size < 160:
        e, dt_a = timed(_segment_acquire, E, np.column_stack([t.real, t.imag]), 1, STAGE_B.E_range)
        acq += dt_a
        tn, dt = timed(solve, e)
        s_t += dt
        E, t = np.append(E, e), np.append(t, tn)
    o = np.argsort(E)
    (fr, fi), f_t = timed(lambda: (PchipInterpolator(E[o], t[o].real), PchipInterpolator(E[o], t[o].imag)))
    Th, i_t = timed(lambda: fr(GRID) ** 2 + fi(GRID) ** 2)
    rows.append(dict(solver=sname, method="segment(E,t) 160 + PCHIP(t)", n_calls=160, solver_s=s_t, acquisition_s=acq,
                     fit_s=f_t, inference_s=i_t, total_s=s_t + acq + f_t + i_t, rmse_T=rmse(Th)))

    # ---- uniform 160 + MLP (Experiment 1 protocol)
    E = np.linspace(*STAGE_B.E_range, 160)
    t, s_t = timed(solve, E)
    m, f_t = timed(lambda: MLPSurrogate(STAGE_B.E_range, MLPConfig(seed=0)).fit(E, np.column_stack([t.real, t.imag])))
    Y, i_t = timed(m.predict, GRID)
    rows.append(dict(solver=sname, method="uniform 160 + MLP(t)", n_calls=160, solver_s=s_t, acquisition_s=0.0,
                     fit_s=f_t, inference_s=i_t, total_s=s_t + f_t + i_t, rmse_T=rmse(Y[:, 0] ** 2 + Y[:, 1] ** 2)))

# per-call costs and amortisation
per_call = {}
for sname, solve in SOLVERS.items():
    E = np.linspace(*STAGE_B.E_range, 400)
    _, dt = timed(lambda: [solve(e) for e in E])
    per_call[sname] = dt / E.size  # single-energy calls (no vectorisation)
for r in rows:
    direct = next(x for x in rows if x["solver"] == r["solver"] and x["method"].startswith("direct"))
    r["speedup_vs_direct_scan"] = direct["total_s"] / r["total_s"]
    # queries Q at which build + Q * inference < Q * solver_call  (single-energy queries)
    build = r["solver_s"] + r["acquisition_s"] + r["fit_s"]
    inf_per_q = r["inference_s"] / GRID.size
    c = per_call[r["solver"]]
    r["break_even_queries"] = float(build / (c - inf_per_q)) if c > inf_per_q and not r["method"].startswith("direct") else None

L = ["# Measured computational cost (Task 5)", "",
     f"Stage B double barrier, target: T(E) on a 20 001-point grid. Single-thread CPU. "
     f"Per-call cost of a single-energy solve: S-matrix {per_call['S-matrix (vectorised)'] * 1e6:.1f} µs, "
     f"ODE {per_call['ODE DOP853'] * 1e3:.2f} ms (the S-matrix solver is ~{per_call['ODE DOP853'] / per_call['S-matrix (vectorised)']:.0f}x "
     f"faster per call and additionally vectorises: {rows[0]['solver_s'] / GRID.size * 1e6:.2f} µs/energy for the full scan).", "",
     "| solver | method | exact calls | solver [s] | acquisition [s] | training/fit [s] | inference on 20 001 E [s] | "
     "total [s] | RMSE(T) | wall-clock speed-up vs direct scan | break-even #queries |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    be = "—" if r["break_even_queries"] is None else f"{r['break_even_queries']:.0f}"
    L.append(f"| {r['solver']} | {r['method']} | {r['n_calls']} | {r['solver_s']:.3g} | {r['acquisition_s']:.3g} | "
             f"{r['fit_s']:.3g} | {r['inference_s']:.3g} | {r['total_s']:.3g} | {r['rmse_T']:.1e} | "
             f"{r['speedup_vs_direct_scan']:.2g}x | {be} |")
L += ["", "Break-even #queries = number of single-energy queries after which building the surrogate "
      "(solver + acquisition + fit) plus answering the queries by inference is cheaper than calling the "
      "solver for each query. A speed-up below 1x means the surrogate pipeline is slower than the direct scan."]
(OUT / "cost_1d.md").write_text("\n".join(L) + "\n")
(OUT / "cost_1d.json").write_text(json.dumps(dict(rows=rows, per_call=per_call), indent=1))
print("\n".join(L))
