"""Experiment 3b — where does rational reconstruction stop detecting a narrow resonance?

Hypothesis (Breit–Wigner tail argument): with uniform spacing h, the samples nearest an
unseen resonance (E_r, Gamma) carry a resonant contribution |delta t| ~ Gamma / h. AAA
with tolerance rtol ~ 3 sigma (sigma = relative noise of the solver output) can only
resolve that pole if Gamma / h exceeds roughly the noise floor, i.e.

    detectable  <=>  Gamma  >~  c * sigma * h ,   c = O(1).

Grid: width scan (Gamma_1 from 2.5e-2 down to ~1e-9) x noise sigma x uniform N.
Scored: is the narrowest resonance found (T_hat > 0.5 within +-Gamma)?
"""
import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

from qscatter.experiments.robustness import Oracle, System, reference, reconstruct, _NOISE_RTOL  # noqa: E402

A = (0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.2)
SIGMAS = (0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2)
NS = (20, 40, 80, 160)
OUT = Path("results/exp3b")


def task(a):
    base = System(f"a{a}", "detect", 10.0, a, 2.5, (0.1, 10.0))
    E_ref, T_ref, res = reference(base)
    narrow = min(res, key=lambda r: r["gamma"])
    rows = []
    for sigma in SIGMAS:
        for seed in range(3):
            s = System(base.key, "detect", 10.0, a, 2.5, (0.1, 10.0), noise=sigma)
            _NOISE_RTOL["rtol"] = max(1e-13, 3 * sigma)
            for N in NS:
                o = Oracle(s, seed=seed)
                E = np.linspace(0.1, 10.0, N)
                t = o(E)
                w = np.abs(E_ref - narrow["E"]) <= narrow["gamma"]
                T_hat = reconstruct("aaa_t", E, t, E_ref[w])
                found = bool(np.all(np.isfinite(T_hat)) and T_hat.max() > 0.5)
                rows.append(dict(a=a, gamma=narrow["gamma"], E_r=narrow["E"], sigma=sigma, seed=seed, N=N,
                                 h=9.9 / (N - 1), found=found))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as p, (OUT / "raw_results.jsonl").open("w") as fh:
        for rows in p.imap_unordered(task, A):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("done")
