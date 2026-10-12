"""Run Experiment 1 (sample efficiency). Resumable: re-running skips finished rows."""
import argparse
import os

# one BLAS thread per worker process (otherwise 4 workers x 4 threads oversubscribe the CPU)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from qscatter.experiments.sample_efficiency import Exp1Config, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()
    cfg = Exp1Config(n_workers=a.workers)
    if a.quick:
        cfg = Exp1Config(budgets=(10, 20), mlp_budgets=(10,), seeds=(0,), out_dir="results/exp1_quick",
                         n_workers=a.workers, mlp_overrides=dict(max_epochs=500, min_epochs=100))
    run(cfg)
