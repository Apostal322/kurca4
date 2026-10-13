"""Recompute the (corrected) spurious-peak metric of Experiment 4 for the deterministic methods.

Trained models (MLP variants, GP, POD+MLP) were not stored, so their spurious-peak counts
from the flawed first metric are withdrawn rather than recomputed.
"""
import json
import warnings
from pathlib import Path

import numpy as np

from qscatter.experiments.parametric_exp import BUDGETS, run_task

warnings.filterwarnings("ignore")
out = []
for m in ("tensor_cubic", "rbf", "hybrid_smatrix"):
    for B in BUDGETS:
        rows = run_task((m, B, 0))
        for split in ("test", "ood"):
            rs = [r for r in rows if r["split"] == split]
            out.append(dict(method=m, budget=B, split=split, n_evals=rs[0]["n_evals"],
                            spurious=sum(r["n_spurious"] for r in rs),
                            median_rmse=float(np.median([r["rmse_T"] for r in rs]))))
            print(out[-1], flush=True)
Path("results/exp4/spurious_recomputed.json").write_text(json.dumps(out, indent=1))
