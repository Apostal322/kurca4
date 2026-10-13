"""Quickstart: solve, reconstruct, train, plot - in about a minute on a laptop CPU.

    python -m qscatter.quickstart              # or: qscatter-quickstart
    python -m qscatter.quickstart --fast       # shorter MLP training (used in CI)

1. Solves the resonant symmetric double barrier (Stage B) on a dense reference grid and lists
   its resonances.
2. Reconstructs T(E) from N = 20 exact evaluations with PCHIP and with AAA rational
   approximation of the complex amplitude t(k) (Nakatsukasa, Sete & Trefethen 2018,
   scipy.interpolate.AAA).
3. Trains a small PyTorch MLP on 40 evaluations.
4. Writes a figure to quickstart_output/quickstart.png.
Output is plain ASCII so it also works in a Windows console.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="quickstart_output", help="output directory")
    ap.add_argument("--fast", action="store_true", help="short MLP training (smoke test)")
    args = ap.parse_args(argv)

    from scipy.interpolate import PchipInterpolator

    from .config import STAGE_B
    from .data import make_training_set, reference_solution
    from .physics.resonances import find_resonances
    from .physics.smatrix import compute_transmission
    from .surrogates.mlp import MLPConfig, MLPSurrogate
    from .surrogates.rational import RationalAAA

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

    # 1. exact solver
    t0 = time.perf_counter()
    ref = reference_solution(STAGE_B)
    print(f"[1] exact S-matrix solver: {ref.E.size} energies in {time.perf_counter() - t0:.3f} s")
    pot = STAGE_B.potential()
    res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(*STAGE_B.E_range, 100001))
    for r in res:
        print(f"    resonance E = {r.E_res:.6f}, FWHM = {r.fwhm:.3e}, T_peak = {r.T_peak:.6f}")

    # 2. reconstruction from 20 exact evaluations
    ts = make_training_set(STAGE_B, 20, "uniform")
    T_pchip = PchipInterpolator(ts.E, ts.t.real)(ref.E) ** 2 + PchipInterpolator(ts.E, ts.t.imag)(ref.E) ** 2
    Y = RationalAAA().fit(ts.E, np.column_stack([ts.t.real, ts.t.imag])).predict(ref.E)
    T_aaa = Y[:, 0] ** 2 + Y[:, 1] ** 2
    print(f"[2] N = {ts.n_evals} exact evaluations: RMSE(T) PCHIP = {rmse(T_pchip, ref.T):.2e}, "
          f"AAA(t(k)) = {rmse(T_aaa, ref.T):.2e}")

    # 3. minimal PyTorch surrogate
    ts40 = make_training_set(STAGE_B, 40, "uniform")
    cfg = MLPConfig(max_epochs=500 if args.fast else 5000, min_epochs=100, patience=10**6, early_stopping=False)
    t0 = time.perf_counter()
    m = MLPSurrogate(STAGE_B.E_range, cfg).fit(ts40.E, np.column_stack([ts40.t.real, ts40.t.imag]))
    Ym = m.predict(ref.E)
    print(f"[3] MLP (1-64-64-2, {cfg.max_epochs} epochs, {time.perf_counter() - t0:.1f} s) on N = 40: "
          f"RMSE(T) = {rmse(Ym[:, 0] ** 2 + Ym[:, 1] ** 2, ref.T):.2e}")

    # 4. figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(ref.E, ref.T, color="#0b0b0b", lw=1.0, label="exact (S-matrix solver)")
    ax.plot(ref.E, T_pchip, color="#eb6834", lw=1.2, ls="--", label="PCHIP, 20 evaluations")
    ax.plot(ref.E, T_aaa, color="#6250d6", lw=1.2, ls=":", label="AAA of t(k), 20 evaluations")
    ax.plot(ts.E, ts.T, "o", color="#0b0b0b", ms=4, label="exact evaluations")
    ax.set(xlabel="E", ylabel="T(E)", title="Symmetric double barrier: reconstruction from 20 exact evaluations")
    ax.legend(fontsize=8, frameon=False)
    path = out / "quickstart.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"[4] figure written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
