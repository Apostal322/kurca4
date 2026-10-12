import numpy as np
import pytest
import torch

from qscatter.config import STAGE_A, STAGE_B
from qscatter.data import TrainingSet, make_training_set, sample_energies
from qscatter.experiments.adaptive import run_strategy
from qscatter.metrics import curve_metrics
from qscatter.surrogates.baselines import BASELINES, T_from_prediction, make_baseline, targets_from_t
from qscatter.surrogates.mlp import MLPConfig, MLPSurrogate

torch.set_num_threads(1)
FAST = dict(max_epochs=300, min_epochs=50, patience=10**6)


@pytest.mark.parametrize("sampling", ["uniform", "random", "chebyshev"])
def test_designs_cover_domain_and_count_evaluations(sampling):
    ts = make_training_set(STAGE_A, 17, sampling, seed=3)
    assert len(ts) == 17 == ts.n_evals
    assert ts.E[0] == STAGE_A.E_range[0] and ts.E[-1] == STAGE_A.E_range[1]
    assert np.all(np.diff(ts.E) > 0)


def test_random_design_reproducible():
    a = sample_energies("random", 20, (0, 1), seed=5)
    assert np.array_equal(a, sample_energies("random", 20, (0, 1), seed=5))
    assert not np.array_equal(a, sample_energies("random", 20, (0, 1), seed=6))


def test_dataset_roundtrip(tmp_path):
    ts = make_training_set(STAGE_B, 12, "random", seed=1)
    ts.save(tmp_path / "d")
    back = TrainingSet.load(tmp_path / "d")
    assert np.array_equal(back.E, ts.E) and np.array_equal(back.t, ts.t)
    assert back.seed == 1 and back.sampling == "random" and back.n_evals == 12


@pytest.mark.parametrize("method", BASELINES)
@pytest.mark.parametrize("target", ["amplitude", "T"])
def test_baselines_reproduce_training_data(method, target):
    ts = make_training_set(STAGE_A, 15, "uniform")
    Y = targets_from_t(ts.t, target)
    model = make_baseline(method, STAGE_A.E_range).fit(ts.E, Y)
    np.testing.assert_allclose(model.predict(ts.E), Y, atol=1e-4)  # GP: noise-free up to jitter


def test_mlp_deterministic_and_save_load(tmp_path):
    ts = make_training_set(STAGE_A, 20, "uniform")
    Y = targets_from_t(ts.t, "amplitude")
    a = MLPSurrogate(STAGE_A.E_range, MLPConfig(seed=7, **FAST)).fit(ts.E, Y)
    b = MLPSurrogate(STAGE_A.E_range, MLPConfig(seed=7, **FAST)).fit(ts.E, Y)
    E = np.linspace(*STAGE_A.E_range, 101)
    assert np.array_equal(a.predict(E), b.predict(E))
    a.save(tmp_path / "m.pt")
    c = MLPSurrogate.load(tmp_path / "m.pt")
    assert np.array_equal(a.predict(E), c.predict(E))


def test_mlp_learns_smooth_curve():
    ts = make_training_set(STAGE_A, 40, "uniform")
    m = MLPSurrogate(STAGE_A.E_range, MLPConfig(seed=0, max_epochs=3000, min_epochs=200, patience=10**6)).fit(
        ts.E, targets_from_t(ts.t, "amplitude"))
    E = np.linspace(*STAGE_A.E_range, 2001)
    from qscatter.physics.smatrix import compute_transmission

    T_hat, _ = T_from_prediction(m.predict(E), "amplitude")
    assert curve_metrics(T_hat, compute_transmission(E, STAGE_A.potential()))["rmse_T"] < 1e-2


@pytest.mark.parametrize("strategy", ["segment_T", "segment_t", "uniform"])
def test_adaptive_designs_count_every_evaluation(strategy):
    d = run_strategy(STAGE_B, strategy, (10, 20, 40))
    for N, ts in d.items():
        assert len(ts) == N == ts.n_evals
        assert len(np.unique(ts.E)) == N
        assert ts.E.min() >= STAGE_B.E_range[0] and ts.E.max() <= STAGE_B.E_range[1]
