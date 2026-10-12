"""Mandatory numerical tests for the reference solver.

Tolerances (documented in docs/REPORT.md):
  TOL_ROUNDOFF = 1e-12  agreement with closed-form formulas / unitarity (pure round-off)
  TOL_ODE      = 1e-8   agreement with the independent ODE integrator (rtol 1e-11)
"""
import numpy as np
import pytest

from qscatter.physics.analytic import single_barrier_T, single_barrier_t
from qscatter.physics.ode_solver import solve_ode
from qscatter.physics.potentials import double_barrier, free_space, from_layers, single_barrier
from qscatter.physics.resonances import find_resonances
from qscatter.physics.smatrix import CountingSolver, compute_scattering, compute_transmission
from qscatter.config import STAGE_A, STAGE_B

TOL_ROUNDOFF = 1e-12
TOL_ODE = 1e-8

E_WIDE = np.concatenate([np.geomspace(1e-4, 1, 200), np.linspace(1, 60, 2000)])


# 1. free propagation -------------------------------------------------------
@pytest.mark.parametrize("length", [0.1, 1.0, 37.0])
def test_free_propagation(length):
    res = compute_scattering(E_WIDE, free_space(length))
    np.testing.assert_allclose(res.t, 1.0, atol=TOL_ROUNDOFF)
    np.testing.assert_allclose(res.r, 0.0, atol=TOL_ROUNDOFF)


def test_zero_height_layers_are_transparent():
    pot = from_layers([0.3, 1.2, 0.7], [0.0, 0.0, 0.0], x0=-5.0)
    res = compute_scattering(E_WIDE, pot)
    np.testing.assert_allclose(res.t, 1.0, atol=TOL_ROUNDOFF)


# 2. single barrier vs analytic ---------------------------------------------
@pytest.mark.parametrize("V0,a", [(5.0, 1.0), (10.0, 0.5), (1.0, 3.0), (-4.0, 1.0)])
def test_single_barrier_analytic_T(V0, a):
    E = E_WIDE[E_WIDE != V0]
    T = compute_transmission(E, single_barrier(V0, a))
    np.testing.assert_allclose(T, single_barrier_T(E, V0, a), rtol=1e-10, atol=TOL_ROUNDOFF)


def test_single_barrier_analytic_amplitude():
    V0, a = 5.0, 1.0
    E = np.linspace(0.05, 30, 3001)
    E = E[np.abs(E - V0) > 1e-6]
    for center in (0.0, 2.5, -7.0):  # global amplitude t must not depend on position
        t = compute_scattering(E, single_barrier(V0, a, center)).t
        np.testing.assert_allclose(t, single_barrier_t(E, V0, a), atol=TOL_ROUNDOFF)


def test_barrier_top_limit():
    """E = V0 exactly: T = 1/(1 + V0 a^2/4); the solver is regular there."""
    V0, a = 5.0, 1.0
    T = compute_transmission(np.array([V0, V0 * (1 + 1e-12), V0 * (1 - 1e-12)]), single_barrier(V0, a))
    np.testing.assert_allclose(T, 1 / (1 + V0 * a**2 / 4), rtol=1e-9)


def test_layer_splitting_invariance():
    """Splitting a barrier into sub-layers must not change the result."""
    E = np.linspace(0.05, 20, 500)
    one = compute_scattering(E, single_barrier(5.0, 1.0))
    split = compute_scattering(E, from_layers([0.25] * 4, [5.0] * 4, x0=-0.5))
    np.testing.assert_allclose(split.t, one.t, atol=1e-11)
    np.testing.assert_allclose(split.r, one.r, atol=1e-11)


# 3. probability conservation -----------------------------------------------
@pytest.mark.parametrize("pot", [single_barrier(5.0, 1.0), double_barrier(10.0, 0.8, 2.5),
                                 double_barrier(10.0, 0.6, 2.0, 7.0, 1.1, well_depth=3.0),
                                 from_layers([0.5, 1, 0.2, 2], [3, -2, 30, 1])])
def test_unitarity(pot):
    res = compute_scattering(E_WIDE, pot)
    assert np.max(res.unitarity_defect) < TOL_ROUNDOFF


# 4. finite, physical outputs incl. thick evanescent barriers ----------------
def test_finite_outputs_opaque_barrier():
    pot = single_barrier(100.0, 40.0)  # kappa*d up to 400: T ~ e^-800 underflows
    res = compute_scattering(np.linspace(0.01, 99, 200), pot)
    assert np.all(np.isfinite(res.t)) and np.all(np.isfinite(res.r))
    assert np.all((res.T >= 0) & (res.T <= 1 + TOL_ROUNDOFF))
    assert np.max(res.unitarity_defect) < TOL_ROUNDOFF


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        compute_scattering([0.0, 1.0], single_barrier(1, 1))
    with pytest.raises(ValueError):
        compute_scattering([np.nan], single_barrier(1, 1))
    with pytest.raises(ValueError):
        compute_scattering([1.0], single_barrier(1e6, 10.0))  # kappa*d > limit


# 5. high-energy behaviour ---------------------------------------------------
def test_high_energy_limit():
    pot = double_barrier(10.0, 0.8, 2.5)
    E = np.array([1e4, 1e5, 1e6])
    res = compute_scattering(E, pot)
    # Born approximation: |r| ~ |V-tilde(2k)| / (2k) <= sum|V| d / (2k) -> 0
    bound = sum(abs(h) * w for h, w in zip(pot.heights, pot.widths)) / (2 * np.sqrt(E))
    assert np.all(res.R <= bound**2 + 1e-12)
    assert res.T[-1] > 1 - 1e-4


def test_low_energy_limit():
    T = compute_transmission(np.array([1e-8, 1e-6]), single_barrier(5.0, 1.0))
    assert np.all(T < 1e-4) and T[0] < T[1]


# 6. double-barrier resonances ----------------------------------------------
def test_symmetric_double_barrier_resonances_reach_unity():
    pot = STAGE_B.potential()
    res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(*STAGE_B.E_range, 200001))
    assert len(res) == 3
    for r in res:
        assert abs(r.T_peak - 1.0) < 1e-9  # symmetric barriers: perfect resonant transmission
    widths = [r.fwhm for r in res]
    assert widths == sorted(widths)  # higher resonances are broader


def test_resonance_independent_of_grid_and_confirmed_by_ode():
    pot = STAGE_B.potential()
    f = lambda e: compute_transmission(e, pot)
    a = find_resonances(f, np.linspace(*STAGE_B.E_range, 20001))
    b = find_resonances(f, np.linspace(*STAGE_B.E_range, 200001))
    for ra, rb in zip(a, b):
        # locating a maximum is conditioned like sqrt(eps): T ~ 1 - c (E-E0)^2,
        # so the peak position is only determined to ~1e-7 of the width
        assert abs(ra.E_res - rb.E_res) < 1e-6 * rb.fwhm
        assert abs(ra.fwhm - rb.fwhm) < 1e-9 * rb.fwhm  # half-max crossings are well-conditioned
    for r in b:  # the independent ODE integrator sees the same peak
        assert abs(abs(solve_ode(r.E_res, pot).t) ** 2 - 1.0) < TOL_ODE


def test_asymmetric_double_barrier_peak_below_one():
    pot = double_barrier(10.0, 0.8, 2.5, height_right=10.0, width_right=1.0)
    res = find_resonances(lambda e: compute_transmission(e, pot), np.linspace(0.1, 10, 200001),
                          min_height=0.01)
    assert res and all(r.T_peak < 0.999 for r in res)


@pytest.mark.parametrize("pot", [STAGE_A.potential(), STAGE_B.potential(),
                                 from_layers([0.5, 1, 0.2, 2], [3, -2, 30, 1])])
def test_against_independent_ode(pot):
    for E in [0.137, 0.9938, 2.0, 5.0, 8.384, 11.3]:
        s = compute_scattering(E, pot)
        o = solve_ode(E, pot)
        assert abs(s.t[0] - o.t) < TOL_ODE
        assert abs(s.r[0] - o.r) < TOL_ODE


# 7. reproducibility ---------------------------------------------------------
def test_reproducible_and_counted():
    pot = STAGE_B.potential()
    E = np.linspace(*STAGE_B.E_range, 1001)
    a = compute_scattering(E, pot)
    b = compute_scattering(E, STAGE_B.potential())
    assert np.array_equal(a.t, b.t) and np.array_equal(a.r, b.r)
    solver = CountingSolver(pot)
    solver(E[:10])
    solver(1.0)
    assert solver.n_evals == 11
