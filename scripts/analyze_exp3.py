"""Figures and tables for Experiment 3 (robustness across systems, conventions, intervals, noise)."""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.experiments.robustness import CHECKPOINTS
from qscatter.plotting import INK2, NEUTRAL, SLOTS, setup

warnings.filterwarnings("ignore")
D = Path("results/exp3")
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]

COMBOS = [  # (strategy, reconstruction, label, colour)
    ("uniform", "pchip_t", "uniform + PCHIP(t)", SLOTS[0]),
    ("uniform", "cubicT", "uniform + cubic spline(T)", NEUTRAL),
    ("segment_T", "pchip_t", "segment(E,T) + PCHIP(t)", SLOTS[1]),
    ("segment_logT", "pchip_t", "segment(E,log T) + PCHIP(t)", SLOTS[7]),
    ("segment_t", "pchip_t", "segment(E,t) + PCHIP(t)", SLOTS[2]),
    ("uniform", "aaa_T", "uniform + AAA(T)", SLOTS[3]),
    ("uniform", "aaa_t", "uniform + AAA(t)", SLOTS[4]),
    ("aaa_greedy", "aaa_t", "AAA greedy + AAA(t)", SLOTS[6]),
]
CAP = 2 * CHECKPOINTS[-1]


def success(r, tol=1e-3):
    return r["n_detected"] == r["n_res"] and r["rmse_T"] <= tol


def n_needed(system, st, rec, tol=1e-3):
    rs = sorted((r for r in rows if r["system"] == system and r["strategy"] == st and r["recon"] == rec),
                key=lambda r: r["N"])
    for r in rs:
        if success(r, tol):
            return r["N"]
    return None


systems = sorted({r["system"] for r in rows})
meta = {r["system"]: r for r in rows}

# --- Fig 1: width scan -------------------------------------------------------
width = sorted([s for s in systems if s.startswith("width")], key=lambda s: -meta[s]["gamma_min"])
G = np.array([meta[s]["gamma_min"] for s in width])
fig, ax = plt.subplots(figsize=(9, 4.6))
never = []
for st, rec, lab, c in COMBOS:
    Ns = [n_needed(s, st, rec) for s in width]
    if not any(Ns):
        never.append(lab)
        continue
    ok = np.array([n is not None for n in Ns])
    y = np.array([n if n else np.nan for n in Ns], float)
    ax.plot(G, y, "o-", color=c, label=lab, ms=5, lw=1.8)
    if (~ok).any():  # first failure marked at the budget ceiling
        ax.plot(G[~ok], np.full((~ok).sum(), 400.0), "x", color=c, ms=8, mew=2)
ax.axhline(320, color=NEUTRAL, lw=0.8, ls=":")
ax.text(G[-1], 430, "× = not reached within the budget of 320", fontsize=8, color=INK2, ha="right")
ax.set(xscale="log", yscale="log", xlabel="width Γ_min of the narrowest resonance", ylim=(8, 600),
       ylabel="exact evaluations needed", title="Width scan (V₀ = 10, d = 2.5, a = 0.6 … 2.4): "
       "evaluations to find all resonances and reach RMSE(T) ≤ 1e-3")
ax.invert_xaxis()
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.6))
ax.text(1.02, 0.15, "never within 320:\n" + "\n".join(never), transform=ax.transAxes, fontsize=8, color=INK2,
        va="top")
fig.savefig(FIG / "exp3_width_scan.png")
plt.close(fig)

# --- Fig 2: success curves over 20 random configurations -----------------------
rand = [s for s in systems if s.startswith("random")]
fig, ax = plt.subplots(figsize=(8, 4.2))
for st, rec, lab, c in COMBOS:
    frac = [np.mean([any(success(r) for r in rows if r["system"] == s and r["strategy"] == st
                         and r["recon"] == rec and r["N"] == N) for s in rand]) for N in CHECKPOINTS]
    ax.plot(CHECKPOINTS, frac, "o-", color=c, label=lab, ms=4.5)
ax.set(xscale="log", xlabel="N exact solver evaluations", ylabel="fraction of configurations solved",
       title="20 random double barriers (V₀∈[6,14], a∈[0.4,1.2], d∈[1.5,3.5]):\n"
             "all resonances found and RMSE(T) ≤ 1e-3")
ax.set_xticks([10, 20, 40, 80, 160, 320], ["10", "20", "40", "80", "160", "320"])
ax.legend(fontsize=7.5, loc="center right")
fig.savefig(FIG / "exp3_random_success.png")
plt.close(fig)

# --- tables ----------------------------------------------------------------------
fmt = lambda n: str(n) if n else "> 320"
L = ["# Experiment 3 — robustness: evaluations needed", "",
     "Entry = smallest N (checkpoints 10…320) at which **all resonances are found** (T̂ > 0.5 within ±Γ) "
     "**and** the L2 RMSE of T on a pole-resolved reference grid is ≤ 1e-3. Exact poles from Newton refinement "
     "of 1/t(E) in the complex plane.", "",
     "| system | Γ_min | # res | " + " | ".join(c[2] for c in COMBOS) + " |", "|---|---|---|" + "---|" * len(COMBOS)]
for s in sorted(systems, key=lambda s: (s.split("_")[0], -meta[s]["gamma_min"])):
    L.append(f"| {s} | {meta[s]['gamma_min']:.1e} | {meta[s]['n_res']} | "
             + " | ".join(fmt(n_needed(s, st, rec)) for st, rec, _, _ in COMBOS) + " |")
L += ["", "## Summary over the 20 random configurations", "",
      "| method | solved within 320 | median N needed (solved only) | max N needed |", "|---|---|---|---|"]
for st, rec, lab, _ in COMBOS:
    ns = [n_needed(s, st, rec) for s in rand]
    ok = [n for n in ns if n]
    L.append(f"| {lab} | {len(ok)}/20 | {np.median(ok) if ok else '—'} | {max(ok) if ok else '—'} |")
# self-certification of the greedy rational method
L += ["", "## Self-certification of the AAA greedy (look-ahead error < 1e-8 three times in a row)", "",
      "| system | certified at N | true RMSE(T) at the next checkpoint ≥ N | all resonances found |", "|---|---|---|---|"]
for s in sorted(systems):
    rs = sorted((r for r in rows if r["system"] == s and r["strategy"] == "aaa_greedy" and r["recon"] == "aaa_t"),
                key=lambda r: r["N"])
    c = rs[-1]["self_certified_at"]
    if c is None:
        L.append(f"| {s} | not certified | — | — |")
        continue
    r = next(r for r in rs if r["N"] >= c)
    L.append(f"| {s} | {c} | {r['rmse_T']:.1e} | {r['n_detected']}/{r['n_res']} |")
(D / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
