# Experiment 4 — parametric surrogate: summary

Domain E ∈ [0.2, 5], V₀ ∈ [5, 10], a ∈ [0.3, 0.7], d ∈ [1.5, 3.0]. 40 held-out test configurations (independent random stream) and 20 out-of-domain configurations, each on a pole-resolved reference grid. Per run: median and 90th percentile over configurations of the L2 RMSE of T; resonances found / total; median position error of found resonances in units of Γ. Several seeds: median over seeds [min–max].

## Held-out test configurations

| method | budget (evals) | median RMSE(T) | 90th pct RMSE(T) | resonances found | pos. err / Γ | frac. unphysical | data gen. [s] | fit [s] | predict [µs/point] | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| tensor-grid cubic interpolation | 2048 | 2.0e-01 | 2.4e-01 | 44/68 | 0.39 | 9.8e-03 | 0.0128 | 0.0623 | 3.79 |  |
| tensor-grid cubic interpolation | 6912 | 9.8e-02 | 1.6e-01 | 53/68 | 0.18 | 5.6e-03 | 0.0437 | 0.141 | 3.67 |  |
| tensor-grid cubic interpolation | 32000 | 4.6e-02 | 1.1e-01 | 56/68 | 0.12 | 2.5e-03 | 0.204 | 1.01 | 4.78 |  |
| RBF (thin-plate, local) | 2048 | 1.7e-01 | 2.0e-01 | 21/68 | 0.19 | 0.0e+00 | 0.0157 | 0.0011 | 128 |  |
| RBF (thin-plate, local) | 8192 | 1.2e-01 | 1.4e-01 | 31/68 | 0.16 | 3.7e-04 | 0.0552 | 0.00324 | 185 |  |
| RBF (thin-plate, local) | 32768 | 8.0e-02 | 1.0e-01 | 43/68 | 0.12 | 1.1e-03 | 0.395 | 0.0179 | 273 |  |
| GP (ARD Matérn, ≤ 2048 pts) | 2048 | 1.6e-01 [1.1e-01–3.0e-01] | 2.2e-01 | 43/68 | 0.26 | 7.6e-03 | 0.0149 | 222 | 194 | gp_points_used=1792 |
| GP (ARD Matérn, ≤ 2048 pts) | 8192 | 2.9e-01 [2.6e-01–2.9e-01] | 4.6e-01 | 50/68 | 0.2 | 2.1e-04 | 0.0605 | 781 | 219 | gp_points_used=2048 |
| POD + MLP | 2048 | 1.5e-01 [1.4e-01–1.6e-01] | 2.3e-01 | 49/68 | 0.27 | 2.4e-02 | 0.0188 | 10.9 | 0.169 | pod_rank=54; pod_compression_rmse_T_on_grid_median=0.000173 |
| POD + MLP | 8192 | 1.1e-01 [7.8e-02–1.7e-01] | 1.6e-01 | 39/68 | 0.21 | 1.4e-02 | 0.0596 | 15.1 | 0.154 | pod_rank=64; pod_compression_rmse_T_on_grid_median=3e-16 |
| POD + MLP | 32768 | 5.0e-02 [4.9e-02–5.8e-02] | 9.0e-02 | 57/68 | 0.11 | 1.4e-02 | 0.263 | 45.7 | 0.16 | pod_rank=64; pod_compression_rmse_T_on_grid_median=3.18e-16 |
| MLP | 2048 | 9.4e-03 [7.8e-03–1.2e-02] | 2.9e-02 | 66/68 | 0.073 | 8.6e-03 | 0.0153 | 108 | 1.42 |  |
| MLP | 8192 | 4.3e-03 [3.0e-03–5.8e-03] | 2.1e-02 | 68/68 | 0.065 | 7.8e-03 | 0.0597 | 108 | 1.61 |  |
| MLP | 32768 | 3.5e-03 [2.8e-03–4.4e-03] | 1.5e-02 | 67/68 | 0.065 | 7.6e-03 | 0.256 | 116 | 1.64 |  |
| MLP + Fourier features | 2048 | 1.7e-01 [1.1e-01–1.9e-01] | 3.4e-01 | 33/68 | 0.2 | 7.3e-03 | 0.0148 | 456 | 2.86 | fourier_sigma=0.5 |
| MLP + Fourier features | 8192 | 4.6e-02 [4.3e-02–5.4e-02] | 1.4e-01 | 55/68 | 0.091 | 5.1e-03 | 0.0561 | 453 | 2.98 | fourier_sigma=0.5 |
| MLP + Fourier features | 32768 | 9.6e-03 [5.4e-03–1.4e-02] | 6.4e-02 | 61/68 | 0.075 | 6.5e-03 | 0.418 | 512 | 2.9 | fourier_sigma=0.5 |
| MLP, dissipative pretraining | 2048 | 3.0e-02 [2.4e-02–4.2e-02] | 5.3e-02 | 57/68 | 0.093 | 1.7e-02 | 0.0156 | 94.8 | 1.43 |  |
| MLP, dissipative pretraining | 8192 | 1.1e-02 [6.8e-03–1.2e-02] | 3.0e-02 | 64/68 | 0.061 | 9.3e-03 | 0.0602 | 104 | 1.74 |  |
| MLP, dissipative pretraining | 32768 | 9.0e-03 [6.4e-03–9.9e-03] | 3.4e-02 | 62/68 | 0.062 | 7.8e-03 | 0.229 | 107 | 1.62 |  |
| hybrid: learned single-barrier S + exact composition | 2048 | 4.5e-05 | 1.8e-04 | 68/68 | 0.054 | 4.5e-04 | 0.0248 | 2.39e-06 | 2.51 |  |
| hybrid: learned single-barrier S + exact composition | 9216 | 1.3e-05 | 2.8e-05 | 68/68 | 0.054 | 3.6e-04 | 0.0704 | 2.8e-06 | 2.46 |  |
| hybrid: learned single-barrier S + exact composition | 32768 | 4.7e-06 | 9.1e-06 | 68/68 | 0.054 | 2.1e-04 | 0.848 | 4.1e-06 | 5.05 |  |

## Out-of-domain configurations

| method | budget (evals) | median RMSE(T) | 90th pct RMSE(T) | resonances found | pos. err / Γ | frac. unphysical | data gen. [s] | fit [s] | predict [µs/point] | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| tensor-grid cubic interpolation | 2048 | 2.2e-01 | 3.2e-01 | 18/35 | 0.36 | 2.8e-02 | 0.0128 | 0.0623 | 4.16 |  |
| tensor-grid cubic interpolation | 6912 | 2.0e-01 | 4.3e-01 | 24/35 | 0.49 | 5.0e-02 | 0.0437 | 0.141 | 3.61 |  |
| tensor-grid cubic interpolation | 32000 | 1.2e-01 | 8.8e-01 | 26/35 | 0.31 | 5.6e-02 | 0.204 | 1.01 | 4.42 |  |
| RBF (thin-plate, local) | 2048 | 1.8e-01 | 2.3e-01 | 1/35 | 0.94 | 0.0e+00 | 0.0157 | 0.0011 | 115 |  |
| RBF (thin-plate, local) | 8192 | 1.7e-01 | 1.9e-01 | 1/35 | 0.45 | 0.0e+00 | 0.0552 | 0.00324 | 154 |  |
| RBF (thin-plate, local) | 32768 | 1.5e-01 | 1.8e-01 | 3/35 | 0.36 | 0.0e+00 | 0.395 | 0.0179 | 211 |  |
| GP (ARD Matérn, ≤ 2048 pts) | 2048 | 1.8e-01 [1.4e-01–2.3e-01] | 2.3e-01 | 10/35 | 0.42 | 1.9e-05 | 0.0149 | 222 | 175 | gp_points_used=1792 |
| GP (ARD Matérn, ≤ 2048 pts) | 8192 | 2.2e-01 [1.9e-01–2.3e-01] | 2.9e-01 | 13/35 | 0.33 | 6.6e-05 | 0.0605 | 781 | 213 | gp_points_used=2048 |
| POD + MLP | 2048 | 2.2e-01 [2.1e-01–3.1e-01] | 3.0e-01 | 16/35 | 0.68 | 2.9e-02 | 0.0188 | 10.9 | 0.153 | pod_rank=54; pod_compression_rmse_T_on_grid_median=0.000173 |
| POD + MLP | 8192 | 1.8e-01 [1.4e-01–2.0e-01] | 2.4e-01 | 14/35 | 0.53 | 2.3e-02 | 0.0596 | 15.1 | 0.149 | pod_rank=64; pod_compression_rmse_T_on_grid_median=3e-16 |
| POD + MLP | 32768 | 8.9e-02 [8.9e-02–9.4e-02] | 1.8e-01 | 24/35 | 0.12 | 1.6e-02 | 0.263 | 45.7 | 0.148 | pod_rank=64; pod_compression_rmse_T_on_grid_median=3.18e-16 |
| MLP | 2048 | 2.4e-02 [2.3e-02–3.2e-02] | 4.9e-02 | 31/35 | 0.075 | 3.6e-03 | 0.0153 | 108 | 1.54 |  |
| MLP | 8192 | 1.1e-02 [9.0e-03–1.5e-02] | 4.2e-02 | 32/35 | 0.043 | 3.4e-03 | 0.0597 | 108 | 1.59 |  |
| MLP | 32768 | 7.9e-03 [7.0e-03–8.7e-03] | 3.9e-02 | 32/35 | 0.038 | 2.3e-03 | 0.256 | 116 | 1.75 |  |
| MLP + Fourier features | 2048 | 1.9e-01 [1.9e-01–2.0e-01] | 2.6e-01 | 8/35 | 0.34 | 0.0e+00 | 0.0148 | 456 | 2.8 | fourier_sigma=0.5 |
| MLP + Fourier features | 8192 | 1.5e-01 [1.3e-01–1.6e-01] | 2.2e-01 | 17/35 | 0.29 | 7.5e-04 | 0.0561 | 453 | 2.78 | fourier_sigma=0.5 |
| MLP + Fourier features | 32768 | 1.1e-01 [6.6e-02–1.2e-01] | 1.9e-01 | 22/35 | 0.18 | 8.6e-04 | 0.418 | 512 | 2.88 | fourier_sigma=0.5 |
| MLP, dissipative pretraining | 2048 | 4.7e-02 [4.7e-02–5.8e-02] | 7.3e-02 | 24/35 | 0.064 | 7.3e-03 | 0.0156 | 94.8 | 1.44 |  |
| MLP, dissipative pretraining | 8192 | 2.3e-02 [2.0e-02–2.5e-02] | 4.4e-02 | 32/35 | 0.058 | 3.5e-03 | 0.0602 | 104 | 1.67 |  |
| MLP, dissipative pretraining | 32768 | 1.9e-02 [1.9e-02–2.6e-02] | 4.9e-02 | 31/35 | 0.047 | 3.1e-03 | 0.229 | 107 | 1.66 |  |
| hybrid: learned single-barrier S + exact composition | 2048 | 1.9e-04 | 4.6e-04 | 35/35 | 0.024 | 9.0e-04 | 0.0248 | 2.39e-06 | 2.92 |  |
| hybrid: learned single-barrier S + exact composition | 9216 | 4.7e-05 | 2.6e-04 | 35/35 | 0.028 | 4.4e-04 | 0.0704 | 2.8e-06 | 2.29 |  |
| hybrid: learned single-barrier S + exact composition | 32768 | 2.3e-05 | 2.2e-04 | 35/35 | 0.028 | 4.9e-04 | 0.848 | 4.1e-06 | 5.43 |  |

## Spurious peaks (corrected metric)

The spurious-peak count logged during the run was wrong (flat pairs of nearly coincident points of the composite reference grid were counted as maxima) and is withdrawn. Corrected count = peaks of T̂ with prominence > 0.05 and height > 0.5 farther than 3Γ from every true resonance, summed over the 40 test (20 OOD) configurations. Recomputed for the deterministic methods and for the plain MLP (seed 0, retrained deterministically); RMSE values reproduce the logged ones exactly.

| method | budget | test: spurious / median RMSE | OOD: spurious / median RMSE |
|---|---|---|---|
| tensor_cubic | 2048 | 2 / 2.0e-01 | 2 / 2.2e-01 |
| tensor_cubic | 6912 | 0 / 9.8e-02 | 10 / 2.0e-01 |
| tensor_cubic | 32000 | 0 / 4.6e-02 | 6 / 1.2e-01 |
| rbf | 2048 | 1 / 1.7e-01 | 0 / 1.8e-01 |
| rbf | 8192 | 0 / 1.2e-01 | 0 / 1.7e-01 |
| rbf | 32768 | 0 / 8.0e-02 | 0 / 1.5e-01 |
| hybrid_smatrix | 2048 | 0 / 4.5e-05 | 0 / 1.9e-04 |
| hybrid_smatrix | 9216 | 0 / 1.3e-05 | 0 / 4.7e-05 |
| hybrid_smatrix | 32768 | 0 / 4.7e-06 | 0 / 2.3e-05 |
| mlp | 2048 | 0 / 7.8e-03 | 0 / 2.3e-02 |
| mlp | 8192 | 0 / 4.3e-03 | 0 / 1.5e-02 |
| mlp | 32768 | 0 / 3.5e-03 | 0 / 8.7e-03 |

## Error vs distance to the nearest training configuration (held-out test set)

Spearman rank correlation between per-configuration RMSE(T) and the scaled distance to the nearest training configuration (positive = error grows with distance). Seed 0.

| method | budget | Spearman ρ | p-value |
|---|---|---|---|
| RBF (thin-plate, local) | 2048 | +0.70 | 4.7e-07 |
| RBF (thin-plate, local) | 8192 | +0.31 | 0.048 |
| RBF (thin-plate, local) | 32768 | +0.20 | 0.22 |
| GP (ARD Matérn, ≤ 2048 pts) | 2048 | +0.49 | 0.0012 |
| GP (ARD Matérn, ≤ 2048 pts) | 8192 | -0.13 | 0.44 |
| POD + MLP | 2048 | +0.19 | 0.23 |
| POD + MLP | 8192 | -0.46 | 0.0027 |
| POD + MLP | 32768 | -0.01 | 0.96 |
| MLP | 2048 | +0.18 | 0.27 |
| MLP | 8192 | +0.23 | 0.15 |
| MLP | 32768 | +0.07 | 0.66 |
| MLP + Fourier features | 2048 | +0.71 | 2.6e-07 |
| MLP + Fourier features | 8192 | +0.46 | 0.0029 |
| MLP + Fourier features | 32768 | +0.33 | 0.04 |
| MLP, dissipative pretraining | 2048 | -0.04 | 0.81 |
| MLP, dissipative pretraining | 8192 | +0.20 | 0.22 |
| MLP, dissipative pretraining | 32768 | +0.01 | 0.97 |
