"""Figures and tables for Experiment 2 (adaptive sampling) from results/exp2/raw_results.jsonl."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.config import PROBLEMS
from qscatter.data import TrainingSet, reference_solution
from qscatter.plotting import INK2, METHOD_LABELS, STRATEGY_COLORS, STRATEGY_LABELS, setup

D = Path("results/exp2")
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]
STRATS = ["uniform", "random", "segment_T", "segment_t", "gp_std", "mlp_ensemble"]
TITLES = dict(single_barrier="Stage A: single barrier", double_barrier="Stage B: double barrier")
TICKS = ([10, 20, 40, 80, 160, 320], ["10", "20", "40", "80", "160", "320"])


def series(prob, strat, recon, metric):
    g = defaultdict(list)
    for r in rows:
        if r["problem"] == prob and r["strategy"] == strat and r["recon"] == recon:
            g[r["N"]].append(r.get(metric, np.nan))
    Ns = sorted(g)
    return np.array(Ns), [np.asarray(g[n], float) for n in Ns]


def strat_plot(recon, metric, ylabel, fname, logy=True, problems=tuple(PROBLEMS)):
    fig, axes = plt.subplots(1, len(problems), figsize=(5.5 * len(problems), 4.2), squeeze=False)
    for ax, prob in zip(axes[0], problems):
        for s in STRATS:
            Ns, vals = series(prob, s, recon, metric)
            if not len(Ns):
                continue
            med = np.array([np.nanmedian(v) for v in vals])
            ax.plot(Ns, med, "o-", ms=4.5, color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s],
                    ls="--" if s in ("uniform", "random") else "-")
            if s == "random" or recon == "mlp":
                lo = [np.nanmin(v) for v in vals]
                hi = [np.nanmax(v) for v in vals]
                ax.fill_between(Ns, lo, hi, color=STRATEGY_COLORS[s], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(*TICKS)
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel="N exact solver evaluations", title=f"{TITLES[prob]} — reconstruction: {METHOD_LABELS[recon]}")
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(FIG / fname)
    plt.close(fig)


for recon in ("pchip", "cubic", "gp", "linear"):
    strat_plot(recon, "rmse_T", "RMSE of T(E) on reference grid", f"exp2_rmse_{recon}.png")
strat_plot("mlp", "rmse_T", "RMSE of T(E) (MLP trained on the design)", "exp2_rmse_mlp.png")
strat_plot("pchip", "n_detected", "resonances reproduced (of 3)", "exp2_resonances_pchip.png", logy=False,
           problems=("double_barrier",))

# designs: where did each strategy put its points?
ref = reference_solution(PROBLEMS["double_barrier"])
N = 80
fig, axes = plt.subplots(len(STRATS), 1, figsize=(10, 9), sharex=True)
for ax, s in zip(axes, STRATS):
    ts = TrainingSet.load(D / "designs" / f"double_barrier_{s}_N{N}_s0")
    ax.plot(ref.E, ref.T, color=INK2, lw=0.9)
    ax.plot(ts.E, ts.T, "|", color=STRATEGY_COLORS[s], ms=14, mew=1.5)
    ax.plot(ts.E, np.full(ts.E.shape, -0.15), "|", color=STRATEGY_COLORS[s], ms=8, mew=1.2)
    ax.set_ylim(-0.25, 1.1)
    ax.set_ylabel("T", fontsize=8)
    ax.text(0.995, 0.85, STRATEGY_LABELS[s], transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
axes[-1].set_xlabel("E")
fig.suptitle(f"Stage B: where each strategy spends N = {N} exact evaluations (ticks at the bottom = sampled energies)",
             y=0.995)
fig.savefig(FIG / "exp2_designs_N80.png")
plt.close(fig)

# --- tables -------------------------------------------------------------------
fmt = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.1e}"
lines = ["# Experiment 2 — adaptive sampling: summary tables", "",
         "All strategies start from the same 10 uniform points. Errors on the dense reference grid. "
         "Random design: median over 5 seeds; MLP reconstruction: median over 3 model seeds.", ""]
for prob in PROBLEMS:
    for recon in ("pchip", "cubic", "gp", "mlp"):
        Nall = sorted({r["N"] for r in rows if r["problem"] == prob and r["recon"] == recon})
        lines += [f"## {TITLES[prob]} — RMSE of T, reconstruction = {METHOD_LABELS[recon]} (amplitude)", "",
                  "| strategy | " + " | ".join(f"N={n}" for n in Nall) + " |", "|---|" + "---|" * len(Nall)]
        for s in STRATS:
            Ns, vals = series(prob, s, recon, "rmse_T")
            d = {n: np.nanmedian(v) for n, v in zip(Ns, vals)}
            lines.append(f"| {STRATEGY_LABELS[s]} | " + " | ".join(fmt(d.get(n)) for n in Nall) + " |")
        lines.append("")

lines += ["## Stage B — resonances reproduced (of 3), PCHIP reconstruction", "",
          "| strategy | " + " | ".join(f"N={n}" for n in TICKS[0]) + " |", "|---|" + "---|" * len(TICKS[0])]
for s in STRATS:
    Ns, vals = series("double_barrier", s, "pchip", "n_detected")
    d = {n: np.nanmean(v) for n, v in zip(Ns, vals)}
    lines.append(f"| {STRATEGY_LABELS[s]} | " + " | ".join(
        "—" if n not in d else f"{d[n]:.1f}" for n in TICKS[0]) + " |")
lines.append("")

lines += ["## Evaluations needed (Stage B, PCHIP reconstruction)", "",
          "| strategy | all 3 resonances detected | RMSE ≤ 1e-2 | RMSE ≤ 1e-3 |", "|---|---|---|---|"]
for s in STRATS:
    Ns, det = series("double_barrier", s, "pchip", "n_detected")
    _, rm = series("double_barrier", s, "pchip", "rmse_T")
    def first(cond):
        hit = [n for n, ok in zip(Ns, cond) if ok]
        return str(hit[0]) if hit else f"> {Ns[-1]}"
    lines.append(f"| {STRATEGY_LABELS[s]} | {first([np.mean(v) == 3 for v in det])} | "
                 f"{first([np.median(v) <= 1e-2 for v in rm])} | {first([np.median(v) <= 1e-3 for v in rm])} |")
lines.append("")

lines += ["## Acquisition overhead (wall-clock spent choosing points, cumulative to N)", "",
          "| problem | strategy | N | acquisition time [s] |", "|---|---|---|---|"]
for prob in PROBLEMS:
    for s in ("segment_T", "segment_t", "gp_std", "mlp_ensemble"):
        sel = [r for r in rows if r["problem"] == prob and r["strategy"] == s and r["recon"] == "pchip"]
        if sel:
            r = max(sel, key=lambda r: r["N"])
            lines.append(f"| {prob} | {STRATEGY_LABELS[s]} | {r['N']} | {r['acq_time_s']:.3g} |")
lines.append("")
(D / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
