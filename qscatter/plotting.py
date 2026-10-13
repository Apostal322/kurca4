"""Shared matplotlib style (static report figures, light surface)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots in fixed order (validated reference palette)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"

METHOD_COLORS = dict(linear=SLOTS[0], pchip=SLOTS[1], cubic=SLOTS[2], gp=SLOTS[3], mlp=SLOTS[4], aaa=SLOTS[6])
METHOD_LABELS = dict(linear="Linear", pchip="PCHIP", cubic="Cubic spline", gp="Gaussian process", mlp="MLP (PyTorch)", aaa="Rational (AAA, in k)")
STRATEGY_COLORS = dict(uniform=SLOTS[0], random=NEUTRAL, segment_T=SLOTS[1], segment_t=SLOTS[2],
                       gp_std=SLOTS[3], mlp_ensemble=SLOTS[4])
STRATEGY_LABELS = dict(uniform="Uniform grid", random="Random", segment_T="Adaptive: segment (E, T)",
                       segment_t="Adaptive: segment (E, t)", gp_std="Adaptive: GP std",
                       mlp_ensemble="Adaptive: MLP ensemble")


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
