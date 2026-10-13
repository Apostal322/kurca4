"""Run Experiment 5 (independent stress test on unseen potentials)."""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
warnings.filterwarnings("ignore")

from qscatter.experiments.stress import run  # noqa: E402

if __name__ == "__main__":
    run(4)
