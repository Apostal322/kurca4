"""Physical problem definitions used throughout the project (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .physics.potentials import double_barrier, single_barrier


@dataclass(frozen=True)
class Problem:
    name: str
    kind: str  # "single" | "double"
    pot_kwargs: dict = field(default_factory=dict)
    E_range: tuple[float, float] = (0.1, 10.0)
    n_ref: int = 20001  # dense reference grid (evaluation only)

    def potential(self):
        if self.kind == "single":
            return single_barrier(**self.pot_kwargs)
        if self.kind == "double":
            return double_barrier(**self.pot_kwargs)
        raise ValueError(self.kind)


# Stage A: smooth, non-resonant transmission through one rectangular barrier.
# Below the first above-barrier transmission resonance at E = V0 + (pi/a)^2 ~ 14.87,
# T(E) rises monotonically across the barrier top E = V0.
STAGE_A = Problem(
    name="single_barrier",
    kind="single",
    pot_kwargs=dict(height=5.0, width=1.0),
    E_range=(0.1, 12.0),
    n_ref=20001,
)

# Stage B: symmetric double barrier, three resonances in range with widths
# ~7.5e-3, 8.7e-2, 5.3e-1 (parameter search in scripts/stage_b_parameter_search.py).
# Reference grid spacing 1e-4 resolves the narrowest resonance with ~75 points per FWHM.
STAGE_B = Problem(
    name="double_barrier",
    kind="double",
    pot_kwargs=dict(height_left=10.0, width_left=0.8, well_width=2.5),
    E_range=(0.1, 10.0),
    n_ref=99001,
)

PROBLEMS = {p.name: p for p in (STAGE_A, STAGE_B)}
