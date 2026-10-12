"""Piecewise-constant 1D potentials.

Units: hbar^2 / (2m) = 1, so the Schrodinger equation reads
    -psi''(x) + V(x) psi(x) = E psi(x),
and the asymptotic wave number is k = sqrt(E).

A potential is a sequence of layers with constant height V_j on
[x_j, x_{j+1}). Outside [x_0, x_N] the potential is zero on both sides
(equal asymptotic potentials), so T + R = 1 for real V.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PiecewiseConstantPotential:
    """Layers of constant potential between ``edges[j]`` and ``edges[j+1]``."""

    edges: tuple[float, ...]
    heights: tuple[float, ...]
    name: str = "custom"
    params: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if len(self.edges) != len(self.heights) + 1:
            raise ValueError("need len(edges) == len(heights) + 1")
        if len(self.heights) == 0:
            raise ValueError("need at least one layer")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        if not np.all(np.isfinite(self.heights)):
            raise ValueError("heights must be finite")

    @property
    def widths(self) -> np.ndarray:
        return np.diff(np.asarray(self.edges, dtype=float))

    @property
    def x_left(self) -> float:
        return float(self.edges[0])

    @property
    def x_right(self) -> float:
        return float(self.edges[-1])

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        v = np.zeros_like(x)
        for lo, hi, h in zip(self.edges[:-1], self.edges[1:], self.heights):
            v = np.where((x >= lo) & (x < hi), h, v)
        return v

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edges"] = list(self.edges)
        d["heights"] = list(self.heights)
        return d


def single_barrier(height: float, width: float, center: float = 0.0) -> PiecewiseConstantPotential:
    """Rectangular barrier of given height on [center - width/2, center + width/2].

    Centering the barrier at x=0 removes a trivial e^{-ik a} phase winding from t(E)
    (see docs/REPORT.md, phase convention).
    """
    if width <= 0:
        raise ValueError("width must be positive")
    lo, hi = center - width / 2, center + width / 2
    return PiecewiseConstantPotential(
        edges=(lo, hi),
        heights=(float(height),),
        name="single_barrier",
        params=dict(height=height, width=width, center=center),
    )


def double_barrier(
    height_left: float,
    width_left: float,
    well_width: float,
    height_right: float | None = None,
    width_right: float | None = None,
    well_depth: float = 0.0,
    center: float = 0.0,
) -> PiecewiseConstantPotential:
    """Two rectangular barriers separated by a well, centred on ``center``.

    Symmetric by default (right barrier = left barrier).
    """
    if height_right is None:
        height_right = height_left
    if width_right is None:
        width_right = width_left
    if min(width_left, width_right, well_width) <= 0:
        raise ValueError("widths must be positive")
    total = width_left + well_width + width_right
    x0 = center - total / 2
    edges = (
        x0,
        x0 + width_left,
        x0 + width_left + well_width,
        x0 + total,
    )
    return PiecewiseConstantPotential(
        edges=edges,
        heights=(float(height_left), float(-well_depth), float(height_right)),
        name="double_barrier",
        params=dict(
            height_left=height_left,
            width_left=width_left,
            well_width=well_width,
            height_right=height_right,
            width_right=width_right,
            well_depth=well_depth,
            center=center,
        ),
    )


def free_space(length: float = 1.0) -> PiecewiseConstantPotential:
    """Zero potential written as one layer (used to test the solver)."""
    return PiecewiseConstantPotential(edges=(-length / 2, length / 2), heights=(0.0,), name="free")


def from_layers(widths: Sequence[float], heights: Sequence[float], x0: float = 0.0) -> PiecewiseConstantPotential:
    edges = tuple(float(x0 + s) for s in np.concatenate([[0.0], np.cumsum(widths)]))
    return PiecewiseConstantPotential(edges=edges, heights=tuple(float(h) for h in heights))
