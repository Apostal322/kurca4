# Experiment 1 — sample efficiency: summary tables

All errors are evaluated on the dense reference grid (Stage A: 20 001 points, Stage B: 99 001 points).
Median over seeds (uniform design: classical interpolants are deterministic; GP/MLP seeds 0–4/0–2).

## Stage A: single barrier (smooth) — RMSE of T, uniform design

| method | target | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 | N=640 |
|---|---|---|---|---|---|---|---|---|
| Linear | amplitude | 4.5e-03 | 1.0e-03 | 2.5e-04 | 6.3e-05 | 1.6e-05 | 3.9e-06 | 9.7e-07 |
| Linear | T | 3.0e-03 | 6.7e-04 | 1.6e-04 | 3.9e-05 | 9.6e-06 | 2.4e-06 | 5.9e-07 |
| PCHIP | amplitude | 1.7e-03 | 3.3e-04 | 6.8e-05 | 1.1e-05 | 1.6e-06 | 2.4e-07 | 3.5e-08 |
| PCHIP | T | 2.9e-04 | 3.1e-05 | 3.5e-06 | 4.2e-07 | 5.1e-08 | 6.4e-09 | 7.9e-10 |
| Cubic spline | amplitude | 8.1e-04 | 1.7e-04 | 3.2e-05 | 5.1e-06 | 6.3e-07 | 5.7e-08 | 4.0e-09 |
| Cubic spline | T | 6.2e-05 | 1.4e-06 | 6.5e-08 | 3.6e-09 | 2.1e-10 | 1.3e-11 | 7.9e-13 |
| Gaussian process | amplitude | 1.1e-03 | 2.3e-04 | 1.4e-05 | 2.1e-06 | 5.9e-07 | 3.6e-07 | 2.5e-07 |
| Gaussian process | T | 1.6e-05 | 8.9e-07 | 7.7e-07 | 6.2e-07 | 4.3e-07 | 2.6e-07 | 1.7e-07 |
| MLP (PyTorch) | amplitude | 1.4e-03 | 3.0e-04 | 9.8e-04 | 1.0e-03 | 1.1e-04 | 1.1e-04 | — |
| MLP (PyTorch) | T | 2.4e-04 | 1.0e-04 | 1.4e-04 | 1.3e-04 | 1.3e-04 | 5.8e-05 | — |

## Stage A: single barrier (smooth) — RMSE of T, random design

| method | target | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 | N=640 |
|---|---|---|---|---|---|---|---|---|
| Linear | amplitude | 1.8e-02 | 7.4e-03 | 1.2e-03 | 4.1e-04 | 1.0e-04 | 3.3e-05 | 8.8e-06 |
| Linear | T | 8.1e-03 | 3.3e-03 | 9.9e-04 | 2.5e-04 | 7.5e-05 | 2.1e-05 | 5.4e-06 |
| PCHIP | amplitude | 7.3e-03 | 3.8e-03 | 3.5e-04 | 1.4e-04 | 2.6e-05 | 8.1e-06 | 2.3e-06 |
| PCHIP | T | 1.6e-03 | 7.2e-04 | 2.0e-04 | 5.1e-05 | 2.0e-05 | 5.4e-06 | 1.4e-06 |
| Cubic spline | amplitude | 1.1e-03 | 3.1e-04 | 4.6e-05 | 1.7e-05 | 1.9e-06 | 2.6e-07 | 1.4e-08 |
| Cubic spline | T | 3.6e-04 | 1.9e-04 | 6.6e-06 | 2.8e-07 | 2.6e-08 | 4.1e-09 | 2.0e-10 |
| Gaussian process | amplitude | 1.7e-03 | 3.3e-04 | 6.3e-05 | 7.4e-06 | 1.6e-06 | 5.2e-07 | 2.9e-07 |
| Gaussian process | T | 1.0e-04 | 2.8e-06 | 9.2e-07 | 6.8e-07 | 4.5e-07 | 2.5e-07 | 1.6e-07 |
| MLP (PyTorch) | amplitude | 3.4e-03 | 1.4e-03 | 6.5e-04 | 2.3e-04 | 1.1e-04 | 1.1e-04 | — |
| MLP (PyTorch) | T | 6.7e-04 | 2.5e-04 | 8.1e-05 | 9.7e-05 | 1.3e-04 | 9.7e-05 | — |

## Stage B: double barrier (resonant) — RMSE of T, uniform design

| method | target | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 | N=640 |
|---|---|---|---|---|---|---|---|---|
| Linear | amplitude | 1.9e-01 | 9.2e-02 | 5.4e-02 | 4.8e-02 | 3.2e-02 | 2.1e-02 | 1.9e-02 |
| Linear | T | 1.7e-01 | 8.4e-02 | 6.4e-02 | 3.9e-02 | 2.6e-02 | 1.8e-02 | 1.5e-02 |
| PCHIP | amplitude | 1.9e-01 | 8.6e-02 | 5.2e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 | 1.8e-02 |
| PCHIP | T | 1.7e-01 | 8.3e-02 | 6.6e-02 | 3.7e-02 | 2.5e-02 | 1.7e-02 | 1.4e-02 |
| Cubic spline | amplitude | 2.0e-01 | 8.9e-02 | 5.9e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 | 1.8e-02 |
| Cubic spline | T | 1.7e-01 | 9.0e-02 | 7.5e-02 | 3.8e-02 | 2.5e-02 | 1.7e-02 | 1.4e-02 |
| Gaussian process | amplitude | 1.9e-01 | 8.8e-02 | 5.9e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 | 1.8e-02 |
| Gaussian process | T | 1.6e-01 | 8.4e-02 | 7.3e-02 | 3.8e-02 | 2.5e-02 | 1.7e-02 | 1.4e-02 |
| MLP (PyTorch) | amplitude | 1.9e-01 | 8.3e-02 | 4.5e-02 | 4.7e-02 | 2.8e-02 | 2.5e-02 | — |
| MLP (PyTorch) | T | 1.7e-01 | 8.5e-02 | 5.9e-02 | 4.1e-02 | 6.5e-02 | 2.7e-02 | — |

## Stage B: double barrier (resonant) — RMSE of T, random design

| method | target | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 | N=640 |
|---|---|---|---|---|---|---|---|---|
| Linear | amplitude | 1.7e-01 | 1.3e-01 | 1.2e-01 | 7.6e-02 | 3.7e-02 | 2.4e-02 | 1.8e-02 |
| Linear | T | 1.6e-01 | 1.2e-01 | 1.2e-01 | 6.2e-02 | 3.0e-02 | 2.4e-02 | 1.5e-02 |
| PCHIP | amplitude | 1.6e-01 | 1.2e-01 | 1.2e-01 | 7.3e-02 | 3.2e-02 | 2.4e-02 | 1.7e-02 |
| PCHIP | T | 1.6e-01 | 1.1e-01 | 1.1e-01 | 5.6e-02 | 2.8e-02 | 2.3e-02 | 1.5e-02 |
| Cubic spline | amplitude | 2.9e-01 | 1.2e-01 | 1.3e-01 | 6.7e-02 | 2.7e-02 | 2.4e-02 | 1.8e-02 |
| Cubic spline | T | 1.9e-01 | 1.3e-01 | 1.1e-01 | 7.2e-02 | 2.6e-02 | 2.2e-02 | 1.6e-02 |
| Gaussian process | amplitude | 1.8e-01 | 1.3e-01 | 1.9e-01 | 1.2e-01 | 6.2e-02 | 2.9e-02 | 2.4e-02 |
| Gaussian process | T | 1.6e-01 | 1.1e-01 | 1.7e-01 | 8.7e-02 | 3.8e-02 | 2.3e-02 | 2.0e-02 |
| MLP (PyTorch) | amplitude | 1.4e-01 | 9.6e-02 | 8.4e-02 | 8.1e-02 | 3.2e-02 | 3.1e-02 | — |
| MLP (PyTorch) | T | 1.8e-01 | 1.2e-01 | 8.6e-02 | 8.1e-02 | 3.3e-02 | 2.8e-02 | — |

## Empirical convergence order on Stage A (uniform design)

Slope p of log RMSE_T vs log N (RMSE ∝ N^-p), least squares over N = 20…640 (N = 20…320 for MLP).

| method | target | p |
|---|---|---|
| Linear | amplitude | 2.01 |
| Linear | T | 2.03 |
| PCHIP | amplitude | 2.67 |
| PCHIP | T | 3.04 |
| Cubic spline | amplitude | 3.07 |
| Cubic spline | T | 4.14 |
| Gaussian process | amplitude | 1.91 |
| Gaussian process | T | 0.50 |
| MLP (PyTorch) | amplitude | 0.59 |
| MLP (PyTorch) | T | 0.17 |

## Smallest N (of the tested budgets) reaching an RMSE_T target, uniform design

| problem | method | target | RMSE ≤ 1e-2 | RMSE ≤ 1e-3 | RMSE ≤ 1e-4 |
|---|---|---|---|---|---|
| single_barrier | Linear | amplitude | 10 | 40 | 80 |
| single_barrier | Linear | T | 10 | 20 | 80 |
| single_barrier | PCHIP | amplitude | 10 | 20 | 40 |
| single_barrier | PCHIP | T | 10 | 10 | 20 |
| single_barrier | Cubic spline | amplitude | 10 | 10 | 40 |
| single_barrier | Cubic spline | T | 10 | 10 | 10 |
| single_barrier | Gaussian process | amplitude | 10 | 20 | 40 |
| single_barrier | Gaussian process | T | 10 | 10 | 10 |
| single_barrier | MLP (PyTorch) | amplitude | 10 | 20 | > 320 |
| single_barrier | MLP (PyTorch) | T | 10 | 10 | 320 |
| double_barrier | Linear | amplitude | > 640 | > 640 | > 640 |
| double_barrier | Linear | T | > 640 | > 640 | > 640 |
| double_barrier | PCHIP | amplitude | > 640 | > 640 | > 640 |
| double_barrier | PCHIP | T | > 640 | > 640 | > 640 |
| double_barrier | Cubic spline | amplitude | > 640 | > 640 | > 640 |
| double_barrier | Cubic spline | T | > 640 | > 640 | > 640 |
| double_barrier | Gaussian process | amplitude | > 640 | > 640 | > 640 |
| double_barrier | Gaussian process | T | > 640 | > 640 | > 640 |
| double_barrier | MLP (PyTorch) | amplitude | > 320 | > 320 | > 320 |
| double_barrier | MLP (PyTorch) | T | > 320 | > 320 | > 320 |

## Stage B resonance reproduction (uniform design, amplitude target, median over seeds)

Detected = surrogate has a peak > 0.5 within ±Γ of the true resonance. Position error in units of Γ.

| method | N | detected (of 3) | spurious peaks | res1 pos err/Γ | res2 pos err/Γ | res3 pos err/Γ | res2 width rel err | res3 width rel err | RMSE in resonance windows | RMSE background |
|---|---|---|---|---|---|---|---|---|---|---|
| Linear | 40 | 2 | 0 | — | 2.4e-01 | 1.7e-01 | 8.5e-01 | 1.3e-01 | 8.8e-02 | 1.5e-03 |
| Linear | 80 | 2 | 0 | — | 3.1e-01 | 2.5e-02 | 1.2e-03 | 5.4e-03 | 7.8e-02 | 2.1e-03 |
| Linear | 160 | 2 | 0 | — | 1.3e-01 | 5.5e-03 | 2.4e-01 | 1.1e-03 | 5.1e-02 | 9.6e-04 |
| Linear | 320 | 2 | 0 | — | 6.9e-03 | 4.1e-03 | 7.9e-03 | 3.0e-04 | 3.3e-02 | 1.6e-03 |
| Linear | 640 | 2 | 0 | — | 7.4e-02 | 8.8e-03 | 2.5e-02 | 4.1e-04 | 3.1e-02 | 1.6e-04 |
| PCHIP | 40 | 2 | 0 | — | 2.4e-01 | 1.8e-01 | 1.2e+00 | 1.3e-01 | 8.4e-02 | 1.6e-03 |
| PCHIP | 80 | 2 | 0 | — | 3.1e-01 | 5.3e-02 | 1.8e-01 | 1.8e-02 | 7.2e-02 | 2.5e-03 |
| PCHIP | 160 | 2 | 0 | — | 1.1e-01 | 2.4e-02 | 9.4e-02 | 5.5e-03 | 4.5e-02 | 1.0e-03 |
| PCHIP | 320 | 2 | 0 | — | 4.0e-02 | 5.9e-03 | 3.4e-02 | 1.1e-03 | 3.0e-02 | 1.4e-03 |
| PCHIP | 640 | 2 | 0 | — | 2.3e-03 | 6.8e-04 | 1.8e-02 | 1.5e-04 | 2.9e-02 | 6.3e-05 |
| Cubic spline | 40 | 2 | 0 | — | 3.2e-01 | 1.5e-01 | 1.6e+00 | 1.8e-01 | 9.6e-02 | 3.7e-03 |
| Cubic spline | 80 | 2 | 0 | — | 3.9e-01 | 4.6e-03 | 3.8e-01 | 8.8e-04 | 7.3e-02 | 3.2e-03 |
| Cubic spline | 160 | 2 | 0 | — | 1.5e-01 | 7.4e-05 | 4.4e-02 | 4.2e-05 | 4.5e-02 | 1.4e-03 |
| Cubic spline | 320 | 2 | 0 | — | 2.3e-03 | 1.1e-04 | 3.3e-02 | 2.8e-06 | 3.0e-02 | 2.8e-03 |
| Cubic spline | 640 | 2 | 0 | — | 3.4e-03 | 7.4e-05 | 3.7e-03 | 1.2e-06 | 3.0e-02 | 5.6e-04 |
| Gaussian process | 40 | 2 | 0 | — | 3.2e-01 | 1.6e-01 | 1.5e+00 | 1.9e-01 | 9.5e-02 | 3.4e-03 |
| Gaussian process | 80 | 2 | 0 | — | 3.9e-01 | 1.1e-04 | 3.9e-01 | 2.4e-03 | 7.2e-02 | 3.2e-03 |
| Gaussian process | 160 | 2 | 0 | — | 1.5e-01 | 4.5e-04 | 7.1e-02 | 1.6e-04 | 4.5e-02 | 1.5e-03 |
| Gaussian process | 320 | 2 | 0 | — | 9.3e-06 | 3.0e-04 | 2.0e-02 | 1.0e-05 | 3.0e-02 | 3.1e-03 |
| Gaussian process | 640 | 2 | 0 | — | 1.2e-03 | 4.9e-04 | 1.5e-03 | 1.3e-05 | 3.0e-02 | 7.4e-04 |
| MLP (PyTorch) | 40 | 2 | 0 | — | 2.2e-01 | 6.2e-02 | 1.1e+00 | 1.0e-01 | 7.3e-02 | 1.9e-03 |
| MLP (PyTorch) | 80 | 2 | 0 | — | 3.6e-01 | 8.9e-03 | 1.6e-01 | 4.1e-03 | 7.5e-02 | 1.3e-03 |
| MLP (PyTorch) | 160 | 2 | 0 | — | 4.8e-02 | 1.3e-02 | 3.0e-01 | 4.2e-03 | 4.6e-02 | 1.5e-03 |
| MLP (PyTorch) | 320 | 2 | 0 | — | 1.1e-01 | 1.4e-02 | 1.1e-01 | 5.2e-03 | 4.0e-02 | 1.2e-03 |

## Unphysical predictions and cost (all runs)

frac_unphysical = fraction of reference energies with T̂ < −1e-6 or T̂ > 1 + 1e-6.

| method | target | runs with any unphysical T̂ | median frac_unphysical (when >0) | worst min T̂ | worst max T̂ | median fit time [s] | median predict time per 10^5 energies [s] |
|---|---|---|---|---|---|---|---|
| Linear | amplitude | 0/84 | — | 2.56e-07 | 1 | 9.46e-06 | 0.00147 |
| Linear | T | 0/84 | — | 3.88e-07 | 1 | 9.15e-06 | 0.000638 |
| PCHIP | amplitude | 11/84 | 5.9e-04 | 2.44e-07 | 1.01 | 0.000431 | 0.00194 |
| PCHIP | T | 0/84 | — | 3.88e-07 | 1 | 0.000382 | 0.00132 |
| Cubic spline | amplitude | 20/84 | 5.3e-03 | 3.44e-08 | 4.08 | 0.00041 | 0.00199 |
| Cubic spline | T | 37/84 | 6.4e-02 | -1.11 | 1.98 | 0.000371 | 0.00133 |
| Gaussian process | amplitude | 18/84 | 2.5e-03 | 2.83e-09 | 7.69 | 0.966 | 0.4 |
| Gaussian process | T | 42/84 | 5.8e-02 | -0.688 | 2.85 | 0.438 | 0.176 |
| MLP (PyTorch) | amplitude | 12/72 | 4.7e-03 | 4.27e-10 | 2.14 | 17.1 | 0.155 |
| MLP (PyTorch) | T | 34/72 | 1.4e-01 | -0.0538 | 1.24 | 13.1 | 0.15 |
