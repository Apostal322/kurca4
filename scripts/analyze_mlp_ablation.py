"""MLP with early stopping + refit (Experiment 1 default) vs. fixed 20k epochs on all points."""
import json
from pathlib import Path

import numpy as np

es = [json.loads(l) for l in Path("results/exp1/raw_results.jsonl").read_text().splitlines()]
no = [json.loads(l) for l in Path("results/exp1_mlp_no_es/raw_results.jsonl").read_text().splitlines()]
sel = lambda rows, p, tg, N: [r["rmse_T"] for r in rows if r["method"] == "mlp" and r["sampling"] == "uniform"
                              and r["problem"] == p and r["target"] == tg and r["N"] == N]
lines = ["# MLP ablation: early stopping + refit vs. fixed 20 000 epochs (uniform design)", "",
         "Median RMSE_T over seeds 0–2 (min–max in brackets).", "",
         "| problem | target | N | early stopping + refit | fixed 20k epochs, no validation split |", "|---|---|---|---|---|"]
for p in ("single_barrier", "double_barrier"):
    for tg in ("amplitude", "T"):
        for N in (10, 20, 40, 80, 160, 320):
            a, b = sel(es, p, tg, N), sel(no, p, tg, N)
            f = lambda v: f"{np.median(v):.1e} [{min(v):.1e}–{max(v):.1e}]"
            lines.append(f"| {p} | {tg} | {N} | {f(a)} | {f(b)} |")
out = "\n".join(lines) + "\n"
Path("results/exp1_mlp_no_es/summary.md").write_text(out)
print(out)
