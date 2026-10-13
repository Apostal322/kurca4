"""Classical reconstruction methods (no PyTorch).

Every method is a ``fit(E, Y) -> predict(E_new) -> Y_new`` map with Y of shape
(n, m). Two targets are compared:
  * "amplitude": Y = (Re t, Im t), predicted T = |t_hat|^2
  * "T":         Y = T directly
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern


class Linear:
    name = "linear"

    def fit(self, x, Y):
        self.x, self.Y = np.asarray(x), np.atleast_2d(np.asarray(Y).T).T
        return self

    def predict(self, x_new):
        return np.column_stack([np.interp(x_new, self.x, y) for y in self.Y.T])


class Pchip:
    name = "pchip"

    def fit(self, x, Y):
        self.f = PchipInterpolator(x, Y, axis=0)
        return self

    def predict(self, x_new):
        return np.atleast_2d(self.f(x_new).T).T


class Cubic:
    """Not-a-knot cubic spline (scipy default)."""

    name = "cubic"

    def fit(self, x, Y):
        self.f = CubicSpline(x, Y, axis=0, bc_type="not-a-knot")
        return self

    def predict(self, x_new):
        return np.atleast_2d(self.f(x_new).T).T


class GP:
    """Gaussian-process regression, one independent GP per output column.

    Inputs are scaled to [0, 1] with the *known* domain bounds and outputs are
    standardised (normalize_y). The kernel (RBF vs Matern-5/2) is selected per
    output by the log marginal likelihood of the training data only — no
    reference data are used. A tiny jitter (alpha) is used because the data are
    noise-free.

    The length-scale lower bound is half the mean sample spacing (scaled units).
    Without it, the noise-free marginal likelihood on resonant data can collapse to
    a length scale far below the sample spacing: the GP then interpolates the data
    with isolated spikes and returns the prior mean in between (observed with
    N = 80, 160 on the double barrier; RMSE 0.2). A length scale shorter than the
    data spacing is not identifiable from the data, so this bound only removes a
    degenerate optimum.
    """

    name = "gp"

    def __init__(self, domain, n_restarts: int = 4, alpha: float = 1e-10, seed: int = 0):
        self.lo, self.hi = domain
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.seed = seed

    def _x(self, x):
        return ((np.asarray(x) - self.lo) / (self.hi - self.lo))[:, None]

    def fit(self, x, Y):
        Y = np.atleast_2d(np.asarray(Y).T).T
        X = self._x(x)
        l_min = 0.5 / (len(X) - 1)
        self.models = []
        for y in Y.T:
            best = None
            for base in (RBF(0.1, (l_min, 10.0)), Matern(0.1, (l_min, 10.0), nu=2.5)):
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base
                gp = GaussianProcessRegressor(
                    kernel, alpha=self.alpha, normalize_y=True,
                    n_restarts_optimizer=self.n_restarts, random_state=self.seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(X, y)
                lml = gp.log_marginal_likelihood_value_
                if best is None or lml > best[0]:
                    best = (lml, gp)
            self.models.append(best[1])
        return self

    def predict(self, x_new, return_std: bool = False):
        X = self._x(x_new)
        out = [m.predict(X, return_std=return_std) for m in self.models]
        if return_std:
            return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])
        return np.column_stack(out)

    @property
    def kernels(self):
        return [str(m.kernel_) for m in self.models]


def make_baseline(name: str, domain, seed: int = 0):
    if name == "gp":
        return GP(domain, seed=seed)
    if name == "aaa":
        from .rational import RationalAAA

        return RationalAAA()
    return {"linear": Linear, "pchip": Pchip, "cubic": Cubic}[name]()


BASELINES = ("linear", "pchip", "cubic", "gp", "aaa")


def targets_from_t(t: np.ndarray, target: str) -> np.ndarray:
    if target == "amplitude":
        return np.column_stack([t.real, t.imag])
    if target == "T":
        return (np.abs(t) ** 2)[:, None]
    raise ValueError(target)


def T_from_prediction(Y: np.ndarray, target: str):
    """Return (T_hat, t_hat or None)."""
    if target == "amplitude":
        t_hat = Y[:, 0] + 1j * Y[:, 1]
        return np.abs(t_hat) ** 2, t_hat
    return Y[:, 0], None
