"""Run Experiment 4 (parametric surrogate). Resumable."""
import argparse
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

from qscatter.experiments.parametric_exp import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", nargs="*")
    a = ap.parse_args()
    run(a.workers, a.methods)
