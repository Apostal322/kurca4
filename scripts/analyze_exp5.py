"""Analysis of Experiment 5 (independent stress test) + pole-width vs FWHM study."""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.plotting import INK2, NEUTRAL, SLOTS, setup

warnings.filterwarnings("ignore")
D = Path("results/exp5")
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]
refs = json.loads((D / "references.json").read_text())
M = [("aaa", "uniform + AAA(t)", SLOTS[4]), ("aaa_greedy", "AAA greedy (look-ahead)", SLOTS[6]),
     ("segment_t_pchip", "segment(E,t) + PCHIP(t)", SLOTS[2]), ("cubic_T", "uniform + cubic spline(T)", NEUTRAL),
     ("gp", "uniform + GP(t)", SLOTS[3]), ("hybrid", "hybrid: block S-matrices + exact composition", SLOTS[0]),
     ("mlp", "uniform + MLP(t)", SLOTS[1]), ("mlp_fourier", "uniform + Fourier-feature MLP(t)", SLOTS[7])]
LAB = {k: l for k, l, _ in M}
families = list(dict.fromkeys(r["family"] for r in rows))
Ns = sorted({r["N"] for r in rows})


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


L = ["# Experiment 5 — independent stress test", "",
     "Systems never used for method development (seed 20261013): asymmetric double barriers, coupled-well "
     "triple barriers (resonance doublets), smooth double-Gaussian barriers (4000-layer staircase, "
     "1.6e-6 from the smooth-potential ODE solution), wide intervals with above-barrier resonances, and solver noise.",
     "Equal exact-evaluation budgets N. Reconstructed = max|T̂−T| within ±2Γ of the pole ≤ 10 % of the true peak. "
     "Spurious = peak of T̂ (prominence > 0.05) farther than 3Γ from every exact pole.", ""]

# --- main table: fraction of resonances reconstructed + median RMSE per family -----------------
for fam in families:
    systems = sorted({r["system"] for r in rows if r["family"] == fam})
    L += [f"## {fam} ({len(systems)} system{'s' if len(systems) > 1 else ''})", "",
          "| method | " + " | ".join(f"N={n}: resonances / median RMSE(T) / spurious" for n in Ns) + " |",
          "|---|" + "---|" * len(Ns)]
    for k, lab, _ in M:
        cells = []
        for n in Ns:
            rs = [r for r in rows if r["family"] == fam and r["method"] == k and r["N"] == n]
            if not rs:
                cells.append("—")
                continue
            rec, tot = sum(r["n_reconstructed"] for r in rs), sum(r["n_res"] for r in rs)
            cells.append(f"{rec}/{tot} · {np.median([r['rmse_T'] for r in rs]):.1e} · {sum(r['n_spurious_peaks'] for r in rs)}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("")

# --- figure: reconstructed fraction vs N, all families pooled (noise-free systems only) ---------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for k, lab, c in M:
    xs, frac, med = [], [], []
    for n in Ns:
        rs = [r for r in rows if r["method"] == k and r["N"] == n and r["noise"] == 0]
        if rs:
            xs.append(n)
            frac.append(sum(r["n_reconstructed"] for r in rs) / max(1, sum(r["n_res"] for r in rs)))
            med.append(np.median([r["rmse_T"] for r in rs]))
    if xs:
        axes[0].plot(xs, frac, "o-", color=c, label=lab, ms=5)
        axes[1].plot(xs, med, "o-", color=c, label=lab, ms=5)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlabel("exact evaluations N (equal budget)")
axes[0].set(ylabel="fraction of resonances reconstructed", ylim=(-0.03, 1.03),
            title="all unseen noise-free systems pooled")
axes[1].set(yscale="log", ylabel="median RMSE(T)", title="median over systems")
axes[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.savefig(FIG / "exp5_equal_budget.png")
plt.close(fig)

# --- AAA pole diagnostics ----------------------------------------------------------------------
L += ["## Rational approximant diagnostics (uniform AAA and greedy AAA, all systems)", "",
      "Poles with Re k inside the window. matched = within 0.1Γ of an exact pole; Froissart = residue < 1e-10·max|t| "
      "or a zero within 1e-3·|Im pole|; upper-half = Im k > 0 (non-physical here, no bound states); "
      "spurious near axis = unmatched, |Im E| < 0.05·W and Re E in the interval; missed = true resonance without a "
      "matching pole. Sensitivity = max change of T̂ when the data are perturbed by 1e-10 (relative).", "",
      "| method | N | fits | poles matched | Froissart | upper-half | spurious near axis | missed resonances | "
      "median pole E error / Γ | median pole Γ rel. error | median / max sensitivity |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in ("aaa", "aaa_greedy"):
    for n in Ns:
        rs = [r for r in rows if r["method"] == k and r["N"] == n and "aaa_diag" in r]
        if not rs:
            continue
        dg = [r["aaa_diag"] for r in rs]
        s = lambda key: sum(d[key] for d in dg)
        pe = [x for d in dg for x in d["pole_E_rel_err"]]
        pg = [x for d in dg for x in d["pole_gamma_rel_err"]]
        sens = [d["perturbation_sensitivity"] for d in dg]
        L.append(f"| {LAB[k]} | {n} | {len(rs)} | {s('n_matched')} | {s('n_froissart')} | {s('n_upper_half')} | "
                 f"{s('n_spurious_near_axis')} | {s('n_missed_resonances')} | {np.median(pe):.1e} | {np.median(pg):.1e} | "
                 f"{np.median(sens):.1e} / {max(sens):.1e} |")
L.append("")
# spurious near-axis poles vs visible artefacts
bad = [r for r in rows if r["method"] in ("aaa", "aaa_greedy") and r.get("aaa_diag", {}).get("n_spurious_near_axis", 0) > 0]
L += [f"Fits with at least one spurious near-axis pole: {len(bad)}; of these, with a spurious *peak* in T̂: "
      f"{sum(r['n_spurious_peaks'] > 0 for r in bad)}; median RMSE of these fits: "
      f"{np.median([r['rmse_T'] for r in bad]) if bad else float('nan'):.1e}.", ""]

# --- self-certification ---------------------------------------------------------------------
g = [r for r in rows if r["method"] == "aaa_greedy" and r["N"] == max(Ns)]
cert = [r for r in g if r.get("self_certified_at")]
L += ["## Greedy self-certification (budget 160)", "",
      f"Certified in {len(cert)}/{len(g)} systems. At certification: all resonances reconstructed in "
      f"{sum(r['n_reconstructed'] == r['n_res'] for r in cert)}/{len(cert)}; max true RMSE(T) "
      f"{max((r['rmse_T'] for r in cert), default=float('nan')):.1e}; median evaluations used "
      f"{np.median([r['self_certified_at'] for r in cert]) if cert else float('nan'):.0f}.", ""]
for r in sorted(g, key=lambda r: r["system"]):
    L.append(f"- {r['system']}: certified at {r.get('self_certified_at')}, RMSE {r['rmse_T']:.1e}, "
             f"{r['n_reconstructed']}/{r['n_res']} resonances")
L.append("")

# --- cost per method (fit + predict + acquisition, solver separately) -----------------------
L += ["## Measured cost per fit (median over systems, N = 160; MLPs N = 160 where run)", "",
      "| method | solver [s] | acquisition [s] | fit [s] | predict on ref grid [s] |", "|---|---|---|---|---|"]
for k, lab, _ in M:
    rs = [r for r in rows if r["method"] == k and r["N"] == 160]
    if rs:
        L.append(f"| {lab} | {np.median([r['solve_s'] for r in rs]):.3g} | {np.median([r['acq_s'] for r in rs]):.3g} | "
                 f"{np.median([r['fit_s'] for r in rs]):.3g} | {np.median([r['predict_s'] for r in rs]):.3g} |")
L.append("")

# --- pole width vs FWHM ------------------------------------------------------------------------
# The FWHM stored with the reference was read off grid points (window spacing 0.02 Γ, i.e. a ~2-4 %
# resolution error). Here it is recomputed with the exact solver: bounded maximisation of T near the
# pole and Brent root-finding of the half-maximum crossings (qscatter.physics.resonances).
from qscatter.experiments.stress import build_systems  # noqa: E402
from qscatter.physics.resonances import find_resonances  # noqa: E402
from qscatter.physics.smatrix import compute_transmission  # noqa: E402

SYS = {s.key: s for s in build_systems()}
P = []
for key, m in refs.items():
    poles = np.array([complex(a, b) for a, b in m["poles"]])
    pot = SYS[key].potential()
    lo_, hi_ = SYS[key].E_range
    for r in m["res"]:
        G0 = r["gamma_pole"]
        win = np.linspace(max(lo_, r["E_peak"] - 20 * G0), min(hi_, r["E_peak"] + 20 * G0), 4001)
        found = find_resonances(lambda e: compute_transmission(e, pot), win, min_height=0.0, min_prominence=1e-3)
        if not found:
            continue
        f = min(found, key=lambda x: abs(x.E_res - r["E_peak"]))
        r = dict(r, fwhm=f.fwhm, E_peak=f.E_res, T_peak=f.T_peak)
        if not np.isfinite(r["fwhm"]):
            continue
        Ep = complex(r["E_pole"], -r["gamma_pole"] / 2)
        others = poles[np.abs(poles - Ep) > 1e-9 * abs(Ep)]
        Dn = np.min(np.abs(others.real - Ep.real)) if len(others) else np.inf
        P.append(dict(system=key, family=m["family"], G=r["gamma_pole"], fwhm=r["fwhm"], E=r["E_pole"],
                      dpeak=(r["E_peak"] - r["E_pole"]) / r["gamma_pole"], Tp=r["T_peak"],
                      overlap=r["gamma_pole"] / Dn, rel_E=r["gamma_pole"] / r["E_pole"]))
dev = np.array([abs(p["fwhm"] / p["G"] - 1) for p in P])
ov = np.array([p["overlap"] for p in P])
re_ = np.array([p["rel_E"] for p in P])
x = np.maximum(ov, re_)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
cols = {f: SLOTS[i] for i, f in enumerate(dict.fromkeys(p["family"] for p in P))}
for p, dv in zip(P, dev):
    axes[0].loglog(max(p["overlap"], p["rel_E"]), max(dv, 1e-7), "o", color=cols[p["family"]], ms=5, alpha=0.8)
for f, c in cols.items():
    axes[0].plot([], [], "o", color=c, label=f)
ok = (dev > 1e-6) & np.isfinite(x)
slope, icpt = np.polyfit(np.log10(x[ok]), np.log10(dev[ok]), 1)
xx = np.geomspace(x[ok].min(), x[ok].max())
axes[0].plot(xx, 10 ** icpt * xx**slope, color=INK2, lw=1, ls="--")
axes[0].text(0.03, 0.9, f"fit: |FWHM/Γ_pole − 1| ∝ x^{slope:.2f}", transform=axes[0].transAxes, fontsize=9, color=INK2)
axes[0].set(xlabel="x = max(Γ/D, Γ/E_r)   (D: distance to the nearest other pole)",
            ylabel="|FWHM / Γ_pole − 1|", title="Pole width vs real-axis FWHM")
axes[0].legend(fontsize=7)
dp = np.array([p["dpeak"] for p in P])
for p in P:
    axes[1].semilogx(max(p["overlap"], p["rel_E"]), p["dpeak"], "o", color=cols[p["family"]], ms=5, alpha=0.8)
axes[1].axhline(0, color=NEUTRAL, lw=0.8)
axes[1].set(xlabel="x = max(Γ/D, Γ/E_r)", ylabel="(E_peak − Re E_pole) / Γ_pole", title="Peak position vs pole position")
fig.savefig(FIG / "exp5_pole_vs_fwhm.png")
plt.close(fig)
narrow = [d for d, xv in zip(dev, x) if xv < 1e-2]
L += ["## Pole width vs FWHM (all resonances of all stress-test systems)", "",
      f"{len(P)} resonances. |FWHM/Γ_pole − 1| grows with x = max(Γ/D, Γ/E_r) with fitted exponent {slope:.2f} "
      f"(log-log least squares). For x < 1e-2 the median deviation is "
      f"{np.median(narrow) if narrow else float('nan'):.1e}; overall median {np.median(dev):.1e}, max {dev.max():.1e}. "
      f"Peak offset (E_peak − Re E_pole)/Γ: median {np.median(np.abs(dp)):.1e}, max {np.max(np.abs(dp)):.2f}.", "",
      "| system | E_pole | Γ_pole | FWHM | FWHM/Γ_pole | (E_peak−E_pole)/Γ | T_peak | Γ/D | Γ/E |", "|---|---|---|---|---|---|---|---|---|"]
for p in sorted(P, key=lambda p: -abs(p["fwhm"] / p["G"] - 1))[:12]:
    L.append(f"| {p['system']} | {p['E']:.4f} | {p['G']:.3e} | {p['fwhm']:.3e} | {p['fwhm'] / p['G']:.4f} | "
             f"{p['dpeak']:+.3f} | {p['Tp']:.3f} | {p['overlap']:.2e} | {p['rel_E']:.2e} |")
L.append("")
(D / "summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
