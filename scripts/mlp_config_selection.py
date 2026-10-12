"""Select MLP optimiser settings by *validation* loss (training data only).

Reference-grid RMSE is printed for transparency but was NOT used for the choice.
Result (Stage A and B, N = 20, 80, uniform, amplitude target) is in
results/mlp_config_selection.txt; "silu_lr1e-2_20k" had the lowest validation
loss on Stage A and was tied on Stage B, and became the MLPConfig default.
"""
import sys
import time

import torch

from qscatter.config import STAGE_A, STAGE_B
from qscatter.data import make_training_set, reference_solution
from qscatter.metrics import curve_metrics
from qscatter.surrogates.baselines import T_from_prediction, targets_from_t
from qscatter.surrogates.mlp import MLPConfig, MLPSurrogate

torch.set_num_threads(1)
CANDIDATES = {
    "tanh_lr3e-3_6k": MLPConfig(activation="tanh", lr=3e-3, max_epochs=6000, patience=1000, lr_decay_final=0.05),
    "tanh_lr1e-2_20k": MLPConfig(activation="tanh", lr=1e-2, max_epochs=20000, patience=3000, lr_decay_final=1e-3),
    "silu_lr1e-2_20k": MLPConfig(activation="silu", lr=1e-2, max_epochs=20000, patience=3000, lr_decay_final=1e-3),
    "tanh128_lr1e-2_20k": MLPConfig(hidden=(128, 128), activation="tanh", lr=1e-2, max_epochs=20000,
                                    patience=3000, lr_decay_final=1e-3),
}

if __name__ == "__main__":
    probs = [STAGE_A, STAGE_B] if len(sys.argv) < 2 else [{"A": STAGE_A, "B": STAGE_B}[sys.argv[1]]]
    for P in probs:
        ref = reference_solution(P)
        for N in (20, 80):
            ts = make_training_set(P, N, "uniform")
            for name, cfg in CANDIDATES.items():
                t0 = time.time()
                m = MLPSurrogate(P.E_range, cfg).fit(ts.E, targets_from_t(ts.t, "amplitude"))
                T_hat, _ = T_from_prediction(m.predict(ref.E), "amplitude")
                print(f"{P.name:15s} N={N:3d} {name:20s} val_loss={m.log.best_val:.2e} "
                      f"best_epoch={m.log.best_epoch:5d} [ref rmse_T={curve_metrics(T_hat, ref.T)['rmse_T']:.1e}] "
                      f"{time.time() - t0:.0f}s", flush=True)
