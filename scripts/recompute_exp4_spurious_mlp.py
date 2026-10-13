"""Corrected spurious-peak metric for the plain MLP (seed 0), by re-running the deterministic training."""
import json
import warnings
from pathlib import Path

import numpy as np

from qscatter.experiments.parametric_exp import BUDGETS, run_task

warnings.filterwarnings("ignore")
out = []
for B in BUDGETS:
    rows = run_task(("mlp", B, 0))
    for split in ("test", "ood"):
        rs = [r for r in rows if r["split"] == split]
        out.append(dict(method="mlp", budget=B, split=split, n_evals=rs[0]["n_evals"],
                        spurious=sum(r["n_spurious"] for r in rs),
                        median_rmse=float(np.median([r["rmse_T"] for r in rs]))))
        print(out[-1], flush=True)
Path("results/exp4/spurious_recomputed_mlp.json").write_text(json.dumps(out, indent=1))
