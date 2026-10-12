"""Run Experiment 2 (sequential / adaptive sampling)."""
import argparse

from qscatter.experiments.adaptive import Exp2Config, run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(Exp2Config(n_workers=a.workers))
