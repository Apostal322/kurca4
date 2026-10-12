"""Reference grids and training datasets.

The dense reference grid is used only for evaluation and plots. Training sets are
produced exclusively through a ``CountingSolver`` so the number of exact solver
evaluations is measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Problem
from .physics.smatrix import CountingSolver, compute_scattering


@dataclass
class Reference:
    problem: str
    E: np.ndarray
    t: np.ndarray
    T: np.ndarray


def reference_solution(problem: Problem, n: int | None = None) -> Reference:
    E = np.linspace(*problem.E_range, n or problem.n_ref)
    res = compute_scattering(E, problem.potential())
    return Reference(problem.name, E, res.t, res.T)


def sample_energies(method: str, n: int, E_range, seed: int = 0) -> np.ndarray:
    """Training energies. All designs include both endpoints (no extrapolation)."""
    lo, hi = E_range
    if n < 2:
        raise ValueError("n >= 2")
    if method == "uniform":
        return np.linspace(lo, hi, n)
    if method == "random":
        rng = np.random.default_rng(seed)
        inner = rng.uniform(lo, hi, n - 2)
        return np.sort(np.concatenate([[lo, hi], inner]))
    if method == "chebyshev":  # Chebyshev–Lobatto nodes, clustered at the ends
        j = np.arange(n)
        return np.sort(lo + (hi - lo) * 0.5 * (1 - np.cos(np.pi * j / (n - 1))))
    raise ValueError(method)


@dataclass
class TrainingSet:
    E: np.ndarray
    t: np.ndarray
    problem: str
    potential_params: dict
    sampling: str
    seed: int
    n_evals: int
    meta: dict = field(default_factory=dict)

    @property
    def T(self):
        return np.abs(self.t) ** 2

    def __len__(self):
        return self.E.size

    def to_rows(self):
        return [
            dict(E=float(e), re_t=float(z.real), im_t=float(z.imag), T=float(abs(z) ** 2))
            for e, z in zip(self.E, self.t)
        ]

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "E,re_t,im_t,T"
        arr = np.column_stack([self.E, self.t.real, self.t.imag, self.T])
        np.savetxt(path.with_suffix(".csv"), arr, delimiter=",", header=header, comments="", fmt="%.17g")
        meta = dict(
            problem=self.problem,
            potential_params=self.potential_params,
            sampling=self.sampling,
            seed=self.seed,
            n_points=len(self),
            n_exact_solver_evals=self.n_evals,
            **self.meta,
        )
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrainingSet":
        path = Path(path)
        arr = np.loadtxt(path.with_suffix(".csv"), delimiter=",", skiprows=1, ndmin=2)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            E=arr[:, 0],
            t=arr[:, 1] + 1j * arr[:, 2],
            problem=meta.pop("problem"),
            potential_params=meta.pop("potential_params"),
            sampling=meta.pop("sampling"),
            seed=meta.pop("seed"),
            n_evals=meta.pop("n_exact_solver_evals"),
            meta={k: v for k, v in meta.items() if k != "n_points"},
        )


def make_training_set(problem: Problem, n: int, sampling: str, seed: int = 0) -> TrainingSet:
    pot = problem.potential()
    solver = CountingSolver(pot)
    E = sample_energies(sampling, n, problem.E_range, seed)
    res = solver(E)
    return TrainingSet(E, res.t, problem.name, pot.params, sampling, seed, solver.n_evals)
