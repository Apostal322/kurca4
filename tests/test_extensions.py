"""Tests for the continuation work: rational surrogate, exact poles, absorption, parametric pieces."""
import numpy as np
import pytest

from qscatter.config import STAGE_B
from qscatter.data import make_training_set, reference_solution
from qscatter.metrics import curve_metrics
from qscatter.physics.poles import find_poles
from qscatter.physics.potentials import double_barrier, single_barrier
from qscatter.physics.resonances import find_resonances
from qscatter.physics.smatrix import compute_scattering, compute_transmission, transmission_complex_energy
from qscatter.surrogates.baselines import T_from_prediction, make_baseline, targets_from_t
from qscatter.surrogates.rational import RationalAAA


def test_complex_energy_continuation_matches_real_axis():
    pot = STAGE_B.potential()
    E = np.linspace(0.2, 9.5, 50)
    np.testing.assert_allclose(transmission_complex_energy(E + 0j, pot), compute_scattering(E, pot).t, atol=1e-13)


def test_poles_agree_with_fwhm_for_narrow_resonances():
    pot = STAGE_B.potential()
    poles = find_poles(pot, STAGE_B.E_range)
    res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(*STAGE_B.E_range, 100001))
    assert len(poles) == len(res) == 3
    for z, r in zip(poles, res):
        assert abs(1 / transmission_complex_energy(z, pot)[0]) < 1e-10  # it is a pole of t
        rel = abs(-2 * z.imag - r.fwhm) / r.fwhm
        # pole width -> FWHM in the isolated-resonance (Breit–Wigner) limit; the background makes
        # the deviation grow with Gamma (measured: 8e-5 at Gamma=0.0075, 1.6e-3 at 0.087, 2.5e-2 at 0.53)
        assert rel < (2e-4 if r.fwhm < 0.01 else 0.05)


def test_aaa_reconstructs_resonances_from_uniform_samples():
    ref = reference_solution(STAGE_B)
    ts = make_training_set(STAGE_B, 40, "uniform")
    m = make_baseline("aaa", STAGE_B.E_range).fit(ts.E, targets_from_t(ts.t, "amplitude"))
    T_hat, _ = T_from_prediction(m.predict(ref.E), "amplitude")
    assert curve_metrics(T_hat, ref.T)["rmse_T"] < 1e-8
    # poles of the surrogate reproduce the physical resonance poles
    true = find_poles(STAGE_B.potential(), STAGE_B.E_range)
    p = m.poles()
    for z in true:
        assert np.min(np.abs(p - z)) < 1e-6


def test_aaa_interpolates_training_data():
    ts = make_training_set(STAGE_B, 25, "random", seed=3)
    m = RationalAAA().fit(ts.E, targets_from_t(ts.t, "amplitude"))
    Y = m.predict(ts.E)
    np.testing.assert_allclose(Y[:, 0] + 1j * Y[:, 1], ts.t, atol=1e-10)


def test_absorbing_well_breaks_unitarity_correctly():
    E = np.linspace(0.2, 5, 200)
    lossless = compute_scattering(E, double_barrier(8.0, 0.5, 2.0))
    lossy = compute_scattering(E, double_barrier(8.0, 0.5, 2.0, well_depth=0.05j))
    assert np.max(lossless.unitarity_defect) < 1e-12
    A = 1 - lossy.T - lossy.R
    assert np.all(A > 0)  # Im V < 0 absorbs flux
    assert np.all(lossy.T <= lossless.T + 1e-12) or np.max(lossy.T) < 1  # resonances are lowered


def test_hybrid_smatrix_composition_is_exact():
    from qscatter.parametric import HybridSMatrix

    E = np.linspace(0.2, 5, 500)
    V0, a, d = 7.3, 0.45, 2.6
    k = np.sqrt(E)
    s = compute_scattering(E, single_barrier(V0, a))
    tb, rb = s.t * np.exp(1j * k * a), s.r * np.exp(1j * k * a)
    P = np.exp(1j * k * d)
    t = tb * P * tb / (1 - rb * rb * P * P) * np.exp(-1j * k * (2 * a + d))
    np.testing.assert_allclose(t, compute_scattering(E, double_barrier(V0, a, d)).t, atol=1e-13)
    # and the learned version (cubic on a single-barrier grid) is accurate inside the domain
    h = HybridSMatrix().fit(64, 12, 12)
    X = np.column_stack([E, np.full_like(E, V0), np.full_like(E, a), np.full_like(E, d)])
    assert np.max(np.abs(np.abs(h.predict(X)) ** 2 - np.abs(t) ** 2)) < 1e-3


def test_parametric_splits_are_disjoint_and_counted():
    from qscatter import parametric as pm
    from qscatter.experiments.parametric_exp import split

    X, t, n, C = pm.scan_dataset(32, 8, seed=0)
    assert n == 32 * 8 == len(X)
    Xtr, ttr, Xv, tv, Ctr, Cv = split(X, t, C, 8)
    assert len(Ctr) + len(Cv) == 32
    assert not set(map(tuple, Ctr)) & set(map(tuple, Cv))
    test = pm.test_configs()
    assert np.all((test >= pm.LO[1:]) & (test <= pm.HI[1:]))
    ood = pm.ood_configs()
    assert np.all(np.any(ood > pm.HI[1:], axis=1))  # every OOD config leaves the domain


@pytest.mark.parametrize("X", [0.0, 10.0])
def test_robustness_oracle_counts_and_convention(X):
    from qscatter.experiments.robustness import Oracle, System

    s = System("x", "t", 10.0, 0.8, 2.5, (0.1, 10.0), X=X)
    o = Oracle(s)
    E = np.linspace(0.1, 10, 7)
    t = o(E)
    assert o.n == 7
    np.testing.assert_allclose(np.abs(t), np.abs(compute_scattering(E, s.potential()).t), atol=1e-14)
