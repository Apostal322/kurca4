"""Detectability map of the narrowest resonance for uniform sampling + AAA(t)."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.plotting import INK2, setup

D = Path("results/exp3b")
setup()
rows = [json.loads(l) for l in (D / "raw_results.jsonl").read_text().splitlines()]
g = defaultdict(list)
for r in rows:
    g[(r["gamma"], r["sigma"], r["N"])].append(r["found"])
gam = sorted({r["gamma"] for r in rows}, reverse=True)
sig = sorted({r["sigma"] for r in rows})
Ns = sorted({r["N"] for r in rows})
fig, axes = plt.subplots(1, len(Ns), figsize=(13, 3.6), sharey=True)
for ax, N in zip(axes, Ns):
    M = np.array([[np.mean(g[(G, s, N)]) for s in sig] for G in gam])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(gam)):
        for j in range(len(sig)):
            ax.text(j, i, f"{M[i, j]:.0%}" if M[i, j] not in (0, 1) else ("✓" if M[i, j] == 1 else "·"),
                    ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 else INK2)
    ax.set_xticks(range(len(sig)), ["0" if s == 0 else f"1e{int(np.log10(s))}" for s in sig], fontsize=8)
    ax.set(xlabel="relative noise σ of the solver output", title=f"N = {N} uniform (h = {9.9 / (N - 1):.3f})")
    ax.grid(False)
axes[0].set_yticks(range(len(gam)), [f"{G:.0e}" for G in gam], fontsize=8)
axes[0].set_ylabel("width Γ of the narrowest resonance")
fig.suptitle("Is the narrowest resonance recovered by AAA(t) from uniform samples? (✓ all 3 noise seeds, · none)",
             y=1.03, fontsize=10)
fig.savefig(D / "detectability_map.png")
plt.close(fig)
lines = ["# Experiment 3b — detectability of the narrowest resonance (uniform sampling + AAA(t))", "",
         "Fraction of 3 noise seeds in which T̂ > 0.5 within ±Γ of the narrowest resonance.", ""]
for N in Ns:
    lines += [f"## N = {N}", "", "| Γ \\ σ | " + " | ".join(str(s) for s in sig) + " |", "|---|" + "---|" * len(sig)]
    for G in gam:
        lines.append(f"| {G:.1e} | " + " | ".join(f"{np.mean(g[(G, s, N)]):.2f}" for s in sig) + " |")
    lines.append("")
(D / "summary.md").write_text("\n".join(lines))
