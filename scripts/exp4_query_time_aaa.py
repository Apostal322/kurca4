"""Experiment 4 add-on: AAA used *per configuration at query time* (different cost model).

For every held-out test / OOD configuration, N exact evaluations of that configuration are
made on demand (uniform in E) and fitted with AAA(t). This needs no training set, but every
new configuration costs N evaluations, whereas a parametric surrogate costs its training
budget once. Break-even number of configurations: B_train / N.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np

from qscatter import parametric as pm
from qscatter.experiments.parametric_exp import _refs
from qscatter.physics.smatrix import CountingSolver
from qscatter.surrogates.rational import RationalAAA

warnings.filterwarnings("ignore")
out = []
for split in ("test", "ood"):
    refs = _refs(split)
    for N in (16, 24, 32, 48):
        rm, det, nres, tt = [], 0, 0, 0.0
        for ref in refs:
            s = CountingSolver(pm.pot(*ref["cfg"]))
            E = np.linspace(pm.LO[0], pm.HI[0], N)
            t0 = time.perf_counter()
            t = s(E).t
            r = RationalAAA().fit(E, np.column_stack([t.real, t.imag]))
            Y = r.predict(ref["E"])
            tt += time.perf_counter() - t0
            m = pm.config_metrics(ref, Y[:, 0] ** 2 + Y[:, 1] ** 2)
            assert s.n_evals == N
            rm.append(m["rmse_T"])
            det += m["n_det"]
            nres += m["n_res"]
        out.append(dict(split=split, N_per_config=N, median_rmse=float(np.median(rm)), p90=float(np.percentile(rm, 90)),
                        found=f"{det}/{nres}", time_per_config_s=tt / len(refs)))
        print(out[-1], flush=True)
Path("results/exp4/query_time_aaa.json").write_text(json.dumps(out, indent=1))
