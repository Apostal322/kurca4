"""Does the evaluation metric depend on the reference-grid density? (should not)"""
from qscatter.config import STAGE_A, STAGE_B
from qscatter.data import make_training_set, reference_solution
from qscatter.metrics import curve_metrics
from qscatter.surrogates.baselines import T_from_prediction, make_baseline, targets_from_t

for P in (STAGE_A, STAGE_B):
    for N in (40, 160, 640):
        ts = make_training_set(P, N, "uniform")
        m = make_baseline("pchip", P.E_range).fit(ts.E, targets_from_t(ts.t, "amplitude"))
        out = []
        for n in (P.n_ref, 4 * P.n_ref - 3):
            ref = reference_solution(P, n)
            out.append(curve_metrics(T_from_prediction(m.predict(ref.E), "amplitude")[0], ref.T)["rmse_T"])
        print(f"{P.name:15s} N={N:4d}  RMSE_T on n_ref={P.n_ref}: {out[0]:.6e}   on 4x finer grid: {out[1]:.6e}"
              f"   relative change {abs(out[0] / out[1] - 1):.1e}")
