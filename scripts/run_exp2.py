"""Run Experiment 2 (sequential / adaptive sampling)."""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from qscatter.experiments.adaptive import Exp2Config, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(Exp2Config(n_workers=a.workers))
