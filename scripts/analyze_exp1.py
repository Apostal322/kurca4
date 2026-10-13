"""Figures and tables for Experiment 1 (sample efficiency) from results/exp1/raw_results.jsonl."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.config import PROBLEMS
from qscatter.data import make_training_set, reference_solution
from qscatter.plotting import INK2, METHOD_COLORS, METHOD_LABELS, setup
from qscatter.surrogates.baselines import T_from_prediction, make_baseline, targets_from_t
from qscatter.surrogates.mlp import MLPSurrogate

D = Path("results/exp1")
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]
METHODS = ["linear", "pchip", "cubic", "gp", "mlp", "aaa"]
TITLES = dict(single_barrier="Stage A: single barrier (smooth)", double_barrier="Stage B: double barrier (resonant)")


def group(key_fields, metric, filt=lambda r: True):
    g = defaultdict(list)
    for r in rows:
        if filt(r):
            g[tuple(r[k] for k in key_fields)].append(r[metric])
    return g


def stats(v):
    v = np.asarray(v, float)
    return np.nanmedian(v), np.nanmin(v), np.nanmax(v), np.nanpercentile(v, 25), np.nanpercentile(v, 75)


def curve_plot(sampling, metric, ylabel, fname, logy=True, title_suffix=""):
    g = group(("problem", "method", "target", "N"), metric, lambda r: r["sampling"] == sampling)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, prob in zip(axes, PROBLEMS):
        for m in METHODS:
            for tgt, ls, mk in (("amplitude", "-", "o"), ("T", "--", "s")):
                Ns = sorted(N for (p, mm, t, N) in g if p == prob and mm == m and t == tgt)
                if not Ns:
                    continue
                st = np.array([stats(g[(prob, m, tgt, N)]) for N in Ns])
                ax.plot(Ns, st[:, 0], ls=ls, marker=mk, ms=4.5, color=METHOD_COLORS[m],
                        mfc=METHOD_COLORS[m] if tgt == "amplitude" else "#fcfcfb", lw=1.6)
                if st.shape[0] and np.any(st[:, 2] > st[:, 1]):
                    lo, hi = (st[:, 3], st[:, 4]) if sampling == "random" else (st[:, 1], st[:, 2])
                    ax.fill_between(Ns, lo, hi, color=METHOD_COLORS[m], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set(xlabel="N exact solver evaluations", title=TITLES[prob] + title_suffix)
        ax.set_xticks([10, 20, 40, 80, 160, 320, 640], ["10", "20", "40", "80", "160", "320", "640"])
    axes[0].set_ylabel(ylabel)
    handles = [plt.Line2D([], [], color=METHOD_COLORS[m], lw=2, label=METHOD_LABELS[m]) for m in METHODS]
    handles += [plt.Line2D([], [], color=INK2, ls="-", marker="o", label="fit amplitude t → T=|t|²"),
                plt.Line2D([], [], color=INK2, ls="--", marker="s", mfc="#fcfcfb", label="fit T directly")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / fname)
    plt.close(fig)


# --- primary plots -----------------------------------------------------------
curve_plot("uniform", "rmse_T", "RMSE of T(E) on reference grid", "exp1_rmse_uniform.png",
           title_suffix="\nuniform design; band = min–max over seeds")
curve_plot("random", "rmse_T", "RMSE of T(E) on reference grid", "exp1_rmse_random.png",
           title_suffix="\nrandom design; median, band = IQR over seeds")
curve_plot("uniform", "max_err_T", "max |T̂ − T| on reference grid", "exp1_maxerr_uniform.png",
           title_suffix="\nuniform design")

# --- resonance detection (Stage B) -------------------------------------------
g = group(("method", "target", "sampling", "N"), "n_detected", lambda r: r["problem"] == "double_barrier")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, samp in zip(axes, ("uniform", "random")):
    for i, m in enumerate(METHODS):
        Ns = sorted(N for (mm, t, s, N) in g if mm == m and t == "amplitude" and s == samp)
        med = [np.mean(g[(m, "amplitude", samp, N)]) for N in Ns]
        ax.plot(np.array(Ns) * (1 + 0.04 * (i - 2)), med, "o-", color=METHOD_COLORS[m], ms=5,
                label=METHOD_LABELS[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 80, 160, 320, 640], ["10", "20", "40", "80", "160", "320", "640"])
    ax.set(xlabel="N exact solver evaluations", title=f"{samp} design")
axes[0].set_ylabel("resonances detected")
axes[0].set_yticks([0, 1, 2, 3])
axes[0].legend(loc="upper left")
fig.suptitle("Stage B: resonances reproduced (of 3) — peak > 0.5 within ±Γ of the true peak; "
             "amplitude target; mean over seeds", y=1.02, fontsize=10)
fig.savefig(FIG / "exp1_resonances_detected.png")
plt.close(fig)

# --- example reconstructions -------------------------------------------------
def predict(prob, N, method, target, sampling="uniform"):
    P = PROBLEMS[prob]
    ref = reference_solution(P)
    ts = make_training_set(P, N, sampling, 0)
    if method == "mlp":
        model = MLPSurrogate.load(D / "models" / f"{prob}_{sampling}_N{N}_{target}_s0.pt")
    else:
        model = make_baseline(method, P.E_range).fit(ts.E, targets_from_t(ts.t, target))
    return ref, ts, T_from_prediction(model.predict(ref.E), target)[0]


fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), gridspec_kw=dict(height_ratios=[2, 1]), sharex="col")
for col, (prob, N) in enumerate((("single_barrier", 10), ("double_barrier", 80))):
    for m in METHODS:
        ref, ts, T_hat = predict(prob, N, m, "amplitude")
        axes[0, col].plot(ref.E, T_hat, color=METHOD_COLORS[m], lw=1.3, label=METHOD_LABELS[m])
        axes[1, col].semilogy(ref.E, np.abs(T_hat - ref.T) + 1e-16, color=METHOD_COLORS[m], lw=1)
    axes[0, col].plot(ref.E, ref.T, color="#0b0b0b", lw=1, ls=":", label="exact")
    axes[0, col].plot(ts.E, ts.T, "o", color="#0b0b0b", ms=4, label=f"{N} training points")
    axes[0, col].set(ylabel="T(E)", title=f"{TITLES[prob].split(':')[0]}, N={N} uniform, amplitude target")
    axes[1, col].set(xlabel="E", ylabel="|T̂ − T|", ylim=(1e-16, 2))
axes[0, 0].legend(fontsize=8)
fig.savefig(FIG / "exp1_examples.png")
plt.close(fig)

# zoom on Stage B resonances at N=160
res = json.loads((D / "resonances_double_barrier.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, r in zip(axes, res):
    for m in METHODS:
        ref, ts, T_hat = predict("double_barrier", 160, m, "amplitude")
        w = np.abs(ref.E - r["E_res"]) < 4 * max(r["fwhm"], 0.05)
        ax.plot(ref.E[w], T_hat[w], color=METHOD_COLORS[m], lw=1.3, label=METHOD_LABELS[m])
    ax.plot(ref.E[w], ref.T[w], color="#0b0b0b", ls=":", lw=1.2, label="exact")
    tw = np.abs(ts.E - r["E_res"]) < 4 * max(r["fwhm"], 0.05)
    ax.plot(ts.E[tw], ts.T[tw], "o", color="#0b0b0b", ms=4, label="training points")
    ax.set(xlabel="E", title=f"E={r['E_res']:.3f}, Γ={r['fwhm']:.2g}")
axes[0].set_ylabel("T(E)")
axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Stage B, N = 160 uniform points: reconstruction around each resonance", y=1.03)
fig.savefig(FIG / "exp1_stageB_zoom_N160.png")
plt.close(fig)

# --- tables -------------------------------------------------------------------
def fmt(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.1e}"


lines = ["# Experiment 1 — sample efficiency: summary tables", "",
         "All errors are evaluated on the dense reference grid (Stage A: 20 001 points, Stage B: 99 001 points).",
         "Median over seeds (uniform design: classical interpolants are deterministic; GP/MLP seeds 0–4/0–2).", ""]
for prob in PROBLEMS:
    for samp in ("uniform", "random"):
        g = group(("method", "target", "N"), "rmse_T", lambda r: r["problem"] == prob and r["sampling"] == samp)
        Ns = sorted({k[2] for k in g})
        lines += [f"## {TITLES[prob]} — RMSE of T, {samp} design", "",
                  "| method | target | " + " | ".join(f"N={n}" for n in Ns) + " |",
                  "|---|---|" + "---|" * len(Ns)]
        for m in METHODS:
            for tgt in ("amplitude", "T"):
                vals = [np.median(g[(m, tgt, n)]) if (m, tgt, n) in g else None for n in Ns]
                lines.append(f"| {METHOD_LABELS[m]} | {tgt} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")

# empirical convergence orders on Stage A (uniform), fitted for N >= 20
lines += ["## Empirical convergence order on Stage A (uniform design)", "",
          "Slope p of log RMSE_T vs log N (RMSE ∝ N^-p), least squares over N = 20…640 "
          "(N = 20…320 for MLP).", "", "| method | target | p |", "|---|---|---|"]
g = group(("method", "target", "N"), "rmse_T", lambda r: r["problem"] == "single_barrier" and r["sampling"] == "uniform")
orders = {}
for m in METHODS:
    for tgt in ("amplitude", "T"):
        Ns = np.array(sorted(n for (mm, t, n) in g if mm == m and t == tgt and n >= 20))
        y = np.array([np.median(g[(m, tgt, n)]) for n in Ns])
        p = -np.polyfit(np.log(Ns), np.log(y), 1)[0]
        orders[(m, tgt)] = p
        lines.append(f"| {METHOD_LABELS[m]} | {tgt} | {p:.2f} |")
lines.append("")

# evaluations needed to reach error targets
lines += ["## Smallest N (of the tested budgets) reaching an RMSE_T target, uniform design", "",
          "| problem | method | target | RMSE ≤ 1e-2 | RMSE ≤ 1e-3 | RMSE ≤ 1e-4 |", "|---|---|---|---|---|---|"]
for prob in PROBLEMS:
    g = group(("method", "target", "N"), "rmse_T", lambda r: r["problem"] == prob and r["sampling"] == "uniform")
    for m in METHODS:
        for tgt in ("amplitude", "T"):
            Ns = sorted(n for (mm, t, n) in g if mm == m and t == tgt)
            cells = []
            for thr in (1e-2, 1e-3, 1e-4):
                hit = [n for n in Ns if np.median(g[(m, tgt, n)]) <= thr]
                cells.append(str(hit[0]) if hit else f"> {Ns[-1]}")
            lines.append(f"| {prob} | {METHOD_LABELS[m]} | {tgt} | " + " | ".join(cells) + " |")
lines.append("")

# Stage B resonance details (uniform, amplitude)
lines += ["## Stage B resonance reproduction (uniform design, amplitude target, median over seeds)", "",
          "Detected = surrogate has a peak > 0.5 within ±Γ of the true resonance. Position error in units of Γ.", "",
          "| method | N | detected (of 3) | spurious peaks | res1 pos err/Γ | res2 pos err/Γ | res3 pos err/Γ | "
          "res2 width rel err | res3 width rel err | RMSE in resonance windows | RMSE background |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
for m in METHODS:
    for N in (40, 80, 160, 320, 640):
        sel = [r for r in rows if r["problem"] == "double_barrier" and r["sampling"] == "uniform"
               and r["method"] == m and r["target"] == "amplitude" and r["N"] == N]
        if not sel:
            continue
        med = lambda k: np.nanmedian([r.get(k, np.nan) for r in sel])
        lines.append(f"| {METHOD_LABELS[m]} | {N} | {med('n_detected'):.0f} | {med('n_spurious'):.0f} | "
                     f"{fmt(med('res0_pos_err_over_gamma'))} | {fmt(med('res1_pos_err_over_gamma'))} | "
                     f"{fmt(med('res2_pos_err_over_gamma'))} | {fmt(med('res1_width_rel_err'))} | "
                     f"{fmt(med('res2_width_rel_err'))} | {fmt(med('rmse_T_resonant'))} | {fmt(med('rmse_T_background'))} |")
lines.append("")

# physicality and cost
lines += ["## Unphysical predictions and cost (all runs)", "",
          "frac_unphysical = fraction of reference energies with T̂ < −1e-6 or T̂ > 1 + 1e-6.", "",
          "| method | target | runs with any unphysical T̂ | median frac_unphysical (when >0) | worst min T̂ | worst max T̂ | "
          "median fit time [s] | median predict time per 10^5 energies [s] |", "|---|---|---|---|---|---|---|---|"]
for m in METHODS:
    for tgt in ("amplitude", "T"):
        sel = [r for r in rows if r["method"] == m and r["target"] == tgt]
        bad = [r["frac_unphysical"] for r in sel if r["frac_unphysical"] > 0]
        lines.append(f"| {METHOD_LABELS[m]} | {tgt} | {len(bad)}/{len(sel)} | {fmt(np.median(bad)) if bad else '—'} | "
                     f"{min(r['min_T_hat'] for r in sel):.3g} | {max(r['max_T_hat'] for r in sel):.3g} | "
                     f"{np.median([r['fit_time_s'] for r in sel]):.3g} | "
                     f"{np.median([r['predict_time_s'] / r['n_ref'] * 1e5 for r in sel]):.3g} |")
lines.append("")
(D / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))

# --- confidence intervals over random designs (Task 2) ------------------------------
rng = np.random.default_rng(0)


def boot_ci(v, n_boot=10000):
    v = np.asarray(v, float)
    meds = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return np.median(v), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


ci = ["# Experiment 1 — random designs: median RMSE_T with 95 % bootstrap CI over seeds", "",
      "Percentile bootstrap of the median (10 000 resamples). With 5 seeds (3 for the MLP) the intervals are wide; "
      "they are reported so that differences smaller than the spread are not over-interpreted.", "",
      "| problem | method | target | N=20 | N=40 | N=160 |", "|---|---|---|---|---|---|"]
for prob in PROBLEMS:
    for m in METHODS:
        for tgt in ("amplitude", "T"):
            cells = []
            for N in (20, 40, 160):
                v = [r["rmse_T"] for r in rows if r["problem"] == prob and r["sampling"] == "random"
                     and r["method"] == m and r["target"] == tgt and r["N"] == N]
                if len(v) < 2:
                    cells.append("—")
                    continue
                med, lo, hi = boot_ci(v)
                cells.append(f"{med:.1e} [{lo:.1e}, {hi:.1e}]")
            ci.append(f"| {prob} | {METHOD_LABELS[m]} | {tgt} | " + " | ".join(cells) + " |")
(D / "confidence_intervals.md").write_text("\n".join(ci) + "\n")
