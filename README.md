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
pip install -r requirements.txt && pip install -e .      # exact versions used: requirements-lock.txt
pytest                                                  # 56 tests, ~25 s
python scripts/audit.py                                 # independent audit, 10 checks -> results/audit/
python scripts/make_physics_figures.py                  # Stage A/B validation -> results/physics/
python scripts/check_reference_grid.py                  # reference-grid convergence
python scripts/run_exp1.py && python scripts/analyze_exp1.py          # Exp. 1 sample efficiency (~1 h, resumable)
python scripts/run_exp1.py --mlp-no-early-stopping && python scripts/analyze_mlp_ablation.py
python scripts/run_exp2.py && python scripts/analyze_exp2.py          # Exp. 2 adaptive sampling (~30 min)
python scripts/run_exp3.py && python scripts/analyze_exp3.py          # Exp. 3 robustness, 38 systems (~20 min)
python scripts/run_exp3b_detectability.py && python scripts/analyze_exp3b.py   # Exp. 3b detectability map
python scripts/run_exp4.py && python scripts/analyze_exp4.py          # Exp. 4 parametric surrogate (~1.5 h)
python scripts/check_parametric_leakage.py                            # split disjointness, NN distances
python scripts/recompute_exp4_spurious.py && python scripts/recompute_exp4_spurious_mlp.py
python scripts/exp4_query_time_aaa.py                                 # per-configuration AAA comparison
python scripts/run_exp5.py && python scripts/analyze_exp5.py          # Exp. 5 independent stress test (~30 min)
python scripts/benchmark_cost.py                        # measured wall-clock cost (run on an idle machine)
python scripts/reproduce_key_results.py                 # recomputes 17 headline numbers of the report (~2 min)
```

Every experiment writes its raw rows (JSONL), configuration and figures under `results/expN/`. Seeds are fixed
and documented in each module. Computation is CPU-only. Physics runs in float64, the 1D MLP in float64, and the
parametric MLP in float32. MLP checkpoints carry provenance metadata: versions, git commit, data hash, seed.
Large regenerable reference caches (`results/*/refs/`) are git-ignored.

## Key results (details and caveats in the reports)

| question | answer (measured) |
|---|---|
| Smooth T(E): evaluations needed | ~10. Cubic spline / GP / AAA reach 10⁻⁵…10⁻⁹. The MLP plateaus at ~10⁻⁴. |
| Resonant T(E), uniform grid, local methods (spline, GP, MLP) | stall at RMSE ≈ 2·10⁻² even with 640 evaluations; miss the narrowest resonance |
| Resonant T(E), **AAA rational approximation of t(k)** | all resonances from **15–30 evaluations** in 65 systems: Γ down to 5·10⁻⁷, asymmetric, doublets, smooth potential, noise |
| Adaptive sampling for local interpolants | helps, but less, and depends on the phase convention |
| Parametric (E, V₀, a, d) → t | MLP is best among generic methods (3.5·10⁻³). The **hybrid S-matrix composition** reaches 4.7·10⁻⁶. Per-configuration AAA: 16 evaluations each. |
| Wall-clock | with a 4.4 ms/call ODE solver: 30 points + AAA is 670–700× faster (two runs) than a direct 20,001-point scan. With the fast solver: no gain. |

## Documents

* [`docs/REPORT_RU.md`](docs/REPORT_RU.md): the coursework report (Russian, 14 sections).
* [`docs/SUPERVISOR_BRIEF_RU.md`](docs/SUPERVISOR_BRIEF_RU.md): 2-page brief for the scientific supervisor.
* [`docs/DEFENSE_PREPARATION_RU.md`](docs/DEFENSE_PREPARATION_RU.md): 28 likely defence questions with answers.
* [`docs/KEY_FIGURES_RU.md`](docs/KEY_FIGURES_RU.md): the 7 key figures and how to explain them.
* [`docs/REPORT.md`](docs/REPORT.md): English research report, with errata.
* [`docs/PROVENANCE.md`](docs/PROVENANCE.md): published methods vs our implementation choices vs new findings,
  plus the reference-verification log.
* `results/audit/audit_report.md`: the independent audit.
