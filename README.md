# Surrogate models for accelerating parametric computations in quantum scattering

*Применение суррогатных моделей для ускорения параметрических расчётов в задаче квантового рассеяния* (provisional title)

Undergraduate computational-physics research project. A validated 1D quantum-scattering solver
produces the complex transmission amplitude t(E). The project then asks how few exact solver
evaluations suffice to reconstruct t(E) and T(E) = |t|². It compares classical interpolation,
Gaussian processes and a PyTorch MLP, with and without adaptive (sequential) sampling.

**Read the report: [`docs/REPORT.md`](docs/REPORT.md)**: physics, methods, results, conclusions and limitations.

## Layout

```
qscatter/
  physics/            NumPy/SciPy only — no PyTorch
    potentials.py     piecewise-constant potentials (single / double barrier)
    smatrix.py        reference solver: per-layer S-matrices + Redheffer star product
    analytic.py       textbook single-barrier T(E) and t(E)
    ode_solver.py     independent check: direct ODE integration (DOP853), wavefunctions
    resonances.py     peak finding, refinement, FWHM, Breit–Wigner
  config.py           the two physical problems (Stage A, Stage B)
  data.py             reference grids, sampling designs, training sets (+ CSV/JSON storage)
  metrics.py          curve, resonance and window metrics
  surrogates/
    baselines.py      linear, PCHIP, cubic spline, Gaussian process
    mlp.py            PyTorch MLP, low-data training protocol, checkpoints
  experiments/
    sample_efficiency.py   Experiment 1
    adaptive.py            Experiment 2 (sequential / adaptive sampling)
scripts/              entry points (run experiments, make figures/tables)
tests/                pytest suite (physics + surrogates)
results/              raw results (JSONL), datasets, models, figures, summary tables
docs/REPORT.md        the write-up
```

## Reproduce

```bash
pip install -r requirements.txt && pip install -e .
pytest                                   # 44 tests, ~15 s
python scripts/make_physics_figures.py   # Stage A/B validation -> results/physics
python scripts/run_exp1.py --workers 4   # Experiment 1 (resumable; ~1 h on 4 CPU cores)
python scripts/analyze_exp1.py           # -> results/exp1/figures, results/exp1/summary.md
python scripts/run_exp2.py --workers 4   # Experiment 2 (~30 min)
python scripts/analyze_exp2.py           # -> results/exp2/figures, results/exp2/summary.md
```

Everything is deterministic: fixed seeds, `torch.use_deterministic_algorithms(True)`, CPU, float64.
Units: ħ²/2m = 1, so −ψ'' + Vψ = Eψ and k = √E.
