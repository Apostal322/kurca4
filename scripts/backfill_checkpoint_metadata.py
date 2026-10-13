"""Add provenance metadata to Experiment 1 MLP checkpoints saved before metadata existed.

The training data are regenerated deterministically (same design + exact solver) and
hashed, so the checkpoint can be matched to its dataset. Weights are not modified.
"""
import re
from pathlib import Path

import torch

from qscatter.config import PROBLEMS
from qscatter.data import make_training_set
from qscatter.surrogates.baselines import targets_from_t
from qscatter.surrogates.mlp import data_hash, provenance

for p in sorted(Path("results/exp1/models").glob("*.pt")):
    d = torch.load(p, weights_only=False)
    if "meta" in d:
        continue
    m = re.match(r"(single_barrier|double_barrier)_(uniform|random)_N(\d+)_(amplitude|T)_s(\d+)", p.stem)
    prob, samp, N, tgt, seed = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
    P = PROBLEMS[prob]
    ts = make_training_set(P, N, samp, seed)
    d["meta"] = {**provenance(), "dtype": "float64", "problem": prob, "sampling": samp, "N": N, "target": tgt,
                 "seed": seed, "potential": P.pot_kwargs, "E_range": P.E_range,
                 "data_hash": data_hash(ts.E, targets_from_t(ts.t, tgt)),
                 "dataset": f"results/exp1/datasets/{prob}_{samp}_N{N}_s{seed}",
                 "note": "metadata added retroactively; git_commit/created refer to the backfill, not to training"}
    torch.save(d, p)
    print("updated", p.name)
