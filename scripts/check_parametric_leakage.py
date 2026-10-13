"""Point 6: no configuration leaks between training, validation, test and OOD sets (Experiment 4).

All sets are regenerated deterministically exactly as in the experiment. Distances are
measured in the scaled parameter cube [-1, 1]^3 of (V0, a, d).
"""
import json
from pathlib import Path

import numpy as np

from qscatter import parametric as pm
from qscatter.experiments.parametric_exp import BUDGETS, N_E, SEEDS

sc = lambda C: 2 * (np.asarray(C) - pm.LO[1:]) / (pm.HI[1:] - pm.LO[1:]) - 1
test, ood = pm.test_configs(), pm.ood_configs()
L = ["# Experiment 4 — configuration-leakage check", "",
     "Scaled parameter space [-1, 1]^3 of (V0, a, d). 'exact overlaps' = identical configurations; "
     "NN distance = distance from each test configuration to the nearest training configuration.", "",
     "| budget | seed | train cfgs | val cfgs | train∩val | train∩test | val∩test | train∩OOD | "
     "test NN distance median / min | typical train spacing |", "|---|---|---|---|---|---|---|---|---|---|"]
ok = True
out = {}
for B in BUDGETS:
    for s in SEEDS:
        C = pm.sobol_configs(B // N_E, s)
        val = np.zeros(len(C), bool)
        val[7::8] = True
        tr, va = C[~val], C[val]
        key = lambda A: {tuple(np.round(x, 12)) for x in A}
        ov = [len(key(tr) & key(va)), len(key(tr) & key(test)), len(key(va) & key(test)), len(key(tr) & key(ood))]
        ok &= sum(ov) == 0
        d = np.min(np.linalg.norm(sc(test)[:, None] - sc(tr)[None], axis=2), axis=1)
        spacing = 2 / len(tr) ** (1 / 3)
        out[f"{B}_{s}"] = d.tolist()
        L.append(f"| {B} | {s} | {len(tr)} | {len(va)} | {ov[0]} | {ov[1]} | {ov[2]} | {ov[3]} | "
                 f"{np.median(d):.3f} / {d.min():.3f} | {spacing:.3f} |")
L += ["", f"Result: {'no overlaps in any split' if ok else 'OVERLAP FOUND'}. Test configurations come from an "
      "independent random stream (seed 777) and OOD configurations (seed 888) each have one parameter beyond the "
      "domain edge; validation configurations are a disjoint 1/8 of the Sobol training configurations."]
Path("results/exp4/leakage_check.md").write_text("\n".join(L) + "\n")
Path("results/exp4/test_nn_distances.json").write_text(json.dumps(out))
print("\n".join(L))
