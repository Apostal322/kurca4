"""Recompute the headline numbers quoted in docs/REPORT_RU.md and compare with the stated values.

Fast (~1-2 min, CPU). Each line: quantity, value stated in the report, recomputed value, PASS/FAIL
(relative agreement within the stated rounding, or order of magnitude for values near round-off).
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from scipy.interpolate import CubicSpline, PchipInterpolator  # noqa: E402

from qscatter import parametric as pm  # noqa: E402
from qscatter.config import STAGE_A, STAGE_B  # noqa: E402
from qscatter.data import make_training_set, reference_solution  # noqa: E402
from qscatter.experiments.parametric_exp import _refs, _xcfg  # noqa: E402
from qscatter.physics.poles import find_poles  # noqa: E402
from qscatter.physics.resonances import find_resonances  # noqa: E402
from qscatter.physics.smatrix import compute_scattering, compute_transmission  # noqa: E402
from qscatter.surrogates.rational import RationalAAA  # noqa: E402

rows = []


def check(name, stated, value, rel=0.06, roundoff=False):
    ok = (abs(np.log10(value / stated)) < 1.0) if roundoff else abs(value / stated - 1) <= rel
    rows.append((name, stated, value, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: stated {stated:.3g}, recomputed {value:.3g}", flush=True)


rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
A, B = reference_solution(STAGE_A), reference_solution(STAGE_B)

ts = make_training_set(STAGE_A, 10, "uniform")
check("Stage A, cubic spline of T, N=10", 6.2e-5, rmse(CubicSpline(ts.E, ts.T)(A.E), A.T))
for N, stated in ((640, 1.8e-2), (160, 2.8e-2)):
    ts = make_training_set(STAGE_B, N, "uniform")
    T = PchipInterpolator(ts.E, ts.t.real)(B.E) ** 2 + PchipInterpolator(ts.E, ts.t.imag)(B.E) ** 2
    check(f"Stage B, PCHIP of t, uniform N={N}", stated, rmse(T, B.T))
for N, stated in ((20, 1.8e-7), (40, 3.2e-12)):
    ts = make_training_set(STAGE_B, N, "uniform")
    Y = RationalAAA().fit(ts.E, np.column_stack([ts.t.real, ts.t.imag])).predict(B.E)
    check(f"Stage B, AAA of t(k), uniform N={N}", stated, rmse(Y[:, 0] ** 2 + Y[:, 1] ** 2, B.T), roundoff=N == 40)

pot = STAGE_B.potential()
res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(*STAGE_B.E_range, 200001))
poles = find_poles(pot, STAGE_B.E_range)
for r, z, (E_st, G_st, Gp_st) in zip(res, poles, ((0.993817, 7.47e-3, 7.470e-3), (3.886801, 8.74e-2, 8.729e-2),
                                                  (8.384040, 0.532, 0.5185))):
    check(f"resonance E_n = {E_st}", E_st, r.E_res, rel=1e-6)
    check("  FWHM", G_st, r.fwhm, rel=3e-3)
    check("  pole width -2 Im E_pole", Gp_st, -2 * z.imag, rel=3e-3)
E = np.linspace(*STAGE_B.E_range, 99001)
check("Stage B unitarity max|T+R-1|", 6.8e-14, float(np.max(compute_scattering(E, pot).unitarity_defect)), roundoff=True)

test = _refs("test")
h = pm.HybridSMatrix().fit(32, 8, 8)
v = [pm.config_metrics(r, np.abs(h.predict(_xcfg(r["cfg"], r["E"]))) ** 2)["rmse_T"] for r in test]
check("Exp.4 hybrid, B=2048, median test RMSE", 4.5e-5, float(np.median(v)))
v = []
for r in test:
    E16 = np.linspace(pm.LO[0], pm.HI[0], 16)
    t16 = compute_scattering(E16, pm.pot(*r["cfg"])).t
    Y = RationalAAA().fit(E16, np.column_stack([t16.real, t16.imag])).predict(r["E"])
    v.append(pm.config_metrics(r, Y[:, 0] ** 2 + Y[:, 1] ** 2)["rmse_T"])
check("Exp.4 per-configuration AAA, 16 evals, median test RMSE", 6.3e-10, float(np.median(v)), roundoff=True)

n_fail = sum(not r[3] for r in rows)
print(f"\n{len(rows) - n_fail}/{len(rows)} reproduced")
with open("results/reproduction_check.md", "w") as fh:
    fh.write("# Reproduction of headline numbers (scripts/reproduce_key_results.py)\n\n"
             "| quantity | stated in report | recomputed | status |\n|---|---|---|---|\n")
    for n, s, val, ok in rows:
        fh.write(f"| {n} | {s:.4g} | {val:.4g} | {'PASS' if ok else 'FAIL'} |\n")
sys.exit(1 if n_fail else 0)
