"""Run Experiment 3 (robustness of adaptive sampling and rational approximation)."""
import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

warnings.filterwarnings("ignore")

from qscatter.experiments.robustness import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    run(n_workers=ap.parse_args().workers)
