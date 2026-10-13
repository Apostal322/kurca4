"""Figures and tables for Experiment 4 (parametric surrogate)."""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.plotting import INK2, NEUTRAL, SLOTS, setup

warnings.filterwarnings("ignore")
D = Path("results/exp4")
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]
METHODS = [  # key, label, colour
    ("tensor_cubic", "tensor-grid cubic interpolation", SLOTS[0]),
    ("rbf", "RBF (thin-plate, local)", SLOTS[1]),
    ("gp", "GP (ARD Matérn, ≤ 2048 pts)", SLOTS[3]),
    ("pod_mlp", "POD + MLP", SLOTS[5]),
    ("mlp", "MLP", SLOTS[4]),
    ("mlp_fourier", "MLP + Fourier features", SLOTS[7]),
    ("mlp_dirtl", "MLP, dissipative pretraining", SLOTS[2]),
    ("hybrid_smatrix", "hybrid: learned single-barrier S + exact composition", SLOTS[6]),
]
rng = np.random.default_rng(0)


def per_run(method, B, split):
    """List over seeds of per-run summaries (median over configurations etc.)."""
    by_seed = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["budget"] == B and r["split"] == split:
            by_seed[r["seed"]].append(r)
    out = []
    for s, rs in sorted(by_seed.items()):
        out.append(dict(seed=s, n_evals=rs[0]["n_evals"],
                        med=np.median([r["rmse_T"] for r in rs]),
                        p90=np.percentile([r["rmse_T"] for r in rs], 90),
                        det=sum(r["n_det"] for r in rs), nres=sum(r["n_res"] for r in rs),
                        spur=sum(r["n_spurious"] for r in rs),
                        pos=np.median([x for r in rs for x in r["pos_err_over_gamma"]]) if any(
                            r["pos_err_over_gamma"] for r in rs) else np.nan,
                        unphys=np.mean([r["frac_unphysical"] for r in rs]),
                        fit=rs[0]["fit_time_s"], gen=rs[0]["gen_time_s"],
                        pred=np.sum([r["predict_time_s"] for r in rs]) / np.sum([r["n_points"] for r in rs]),
                        extra={k: rs[0][k] for k in ("fourier_sigma", "pod_rank", "gp_points_used",
                                                     "pod_compression_rmse_T_on_grid_median") if k in rs[0]}))
    return out


budgets = sorted({r["budget"] for r in rows})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, split, title in ((axes[0], "test", "held-out configurations (inside the domain)"),
                         (axes[1], "ood", "out-of-domain configurations (one parameter +2…10 %)")):
    for key, lab, c in METHODS:
        xs, ys, lo, hi = [], [], [], []
        for B in budgets:
            pr = per_run(key, B, split)
            if not pr:
                continue
            v = [p["med"] for p in pr]
            xs.append(np.mean([p["n_evals"] for p in pr]))
            ys.append(np.median(v))
            lo.append(min(v))
            hi.append(max(v))
        if xs:
            ax.plot(xs, ys, "o-", color=c, label=lab, ms=5)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set(xscale="log", yscale="log", xlabel="exact solver evaluations (training + validation data)",
           title=title)
axes[0].set_ylabel("median over configurations of RMSE(T)\n(band: min–max over seeds)")
axes[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.suptitle("Experiment 4: parametric surrogate (E, V₀, a, d) → t, symmetric double barrier", y=1.02)
fig.savefig(FIG / "exp4_rmse_vs_budget.png")
plt.close(fig)

# example predictions: the test configuration with the narrowest resonance
test_refs = list(np.load(D / "refs" / "test.npz", allow_pickle=True)["refs"]) if (D / "refs" / "test.npz").exists() else []

L = ["# Experiment 4 — parametric surrogate: summary", "",
     "Domain E ∈ [0.2, 5], V₀ ∈ [5, 10], a ∈ [0.3, 0.7], d ∈ [1.5, 3.0]. 40 held-out test configurations "
     "(independent random stream) and 20 out-of-domain configurations, each on a pole-resolved reference grid. "
     "Per run: median and 90th percentile over configurations of the L2 RMSE of T; resonances found / total; "
     "spurious peaks; median position error of found resonances in units of Γ. Several seeds: median over seeds "
     "[min–max].", ""]
for split in ("test", "ood"):
    L += [f"## {'Held-out test configurations' if split == 'test' else 'Out-of-domain configurations'}", "",
          "| method | budget (evals) | median RMSE(T) | 90th pct RMSE(T) | resonances found | spurious peaks | "
          "pos. err / Γ | frac. unphysical | data gen. [s] | fit [s] | predict [µs/point] | notes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, lab, _ in METHODS:
        for B in budgets:
            pr = per_run(key, B, split)
            if not pr:
                continue
            med = [p["med"] for p in pr]
            cell = f"{np.median(med):.1e}" + (f" [{min(med):.1e}–{max(med):.1e}]" if len(pr) > 1 else "")
            notes = "; ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in pr[0]["extra"].items())
            L.append(f"| {lab} | {pr[0]['n_evals']} | {cell} | {np.median([p['p90'] for p in pr]):.1e} | "
                     f"{np.median([p['det'] for p in pr]):.0f}/{pr[0]['nres']} | {np.median([p['spur'] for p in pr]):.0f} | "
                     f"{np.nanmedian([p['pos'] for p in pr]):.2g} | {np.median([p['unphys'] for p in pr]):.1e} | "
                     f"{np.median([p['gen'] for p in pr]):.3g} | {np.median([p['fit'] for p in pr]):.3g} | "
                     f"{np.median([p['pred'] for p in pr]) * 1e6:.3g} | {notes} |")
    L.append("")
(D / "summary.md").write_text("\n".join(L))
print("\n".join(L))
