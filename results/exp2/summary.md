# Experiment 2 — adaptive sampling: summary tables

All strategies start from the same 10 uniform points. Errors on the dense reference grid. Random design: median over 5 seeds; MLP reconstruction: median over 3 model seeds.

## Stage A: single barrier — RMSE of T, reconstruction = PCHIP (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 1.7e-03 | 3.3e-04 | 6.8e-05 | 1.1e-05 | 1.6e-06 | 2.4e-07 |
| Random | 7.3e-03 | 3.8e-03 | 3.5e-04 | 1.4e-04 | 2.6e-05 | 8.1e-06 |
| Adaptive: segment (E, T) | 1.7e-03 | 3.6e-04 | 7.5e-05 | 1.3e-05 | 2.6e-06 | 3.0e-07 |
| Adaptive: segment (E, t) | 1.7e-03 | 2.5e-04 | 4.7e-05 | 6.9e-06 | 1.8e-06 | 1.5e-07 |
| Adaptive: GP std | 1.7e-03 | 3.2e-04 | 4.5e-05 | 7.2e-06 | 1.2e-05 | 4.4e-06 |
| Adaptive: MLP ensemble | 1.7e-03 | 1.4e-03 | 1.4e-03 | 1.0e-03 | 6.0e-04 | — |

## Stage A: single barrier — RMSE of T, reconstruction = Cubic spline (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 8.1e-04 | 1.7e-04 | 3.2e-05 | 5.1e-06 | 6.3e-07 | 5.7e-08 |
| Random | 1.1e-03 | 3.1e-04 | 4.6e-05 | 1.7e-05 | 1.9e-06 | 2.6e-07 |
| Adaptive: segment (E, T) | 8.1e-04 | 1.9e-04 | 3.9e-05 | 6.7e-06 | 8.6e-07 | 8.3e-08 |
| Adaptive: segment (E, t) | 8.1e-04 | 4.2e-05 | 6.7e-06 | 8.6e-07 | 8.4e-09 | 6.9e-10 |
| Adaptive: GP std | 8.1e-04 | 1.8e-04 | 2.6e-05 | 2.4e-07 | 3.9e-08 | 9.5e-09 |
| Adaptive: MLP ensemble | 8.1e-04 | 5.8e-05 | 3.0e-05 | 1.1e-05 | 1.5e-06 | — |

## Stage A: single barrier — RMSE of T, reconstruction = Gaussian process (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 1.1e-03 | 2.3e-04 | 1.4e-05 | 2.1e-06 | 5.9e-07 | 3.6e-07 |
| Random | 1.7e-03 | 3.3e-04 | 6.3e-05 | 7.4e-06 | 1.6e-06 | 5.2e-07 |
| Adaptive: segment (E, T) | 1.1e-03 | 2.5e-04 | 1.8e-05 | 2.7e-06 | 6.4e-07 | 3.7e-07 |
| Adaptive: segment (E, t) | 1.1e-03 | 1.2e-04 | 1.0e-05 | 1.5e-06 | 4.6e-07 | 3.5e-07 |
| Adaptive: GP std | 1.1e-03 | 2.4e-04 | 1.0e-05 | 4.7e-07 | 5.6e-07 | 3.9e-07 |
| Adaptive: MLP ensemble | 1.1e-03 | 8.1e-04 | 4.4e-04 | 2.4e-06 | 6.4e-07 | — |

## Stage A: single barrier — RMSE of T, reconstruction = MLP (PyTorch) (amplitude)

| strategy | N=80 | N=160 |
|---|---|---|
| Uniform grid | 1.0e-03 | 1.1e-04 |
| Random | 4.7e-04 | 1.2e-04 |
| Adaptive: segment (E, T) | 1.0e-03 | 9.9e-05 |
| Adaptive: segment (E, t) | 1.2e-04 | 1.2e-04 |
| Adaptive: GP std | 1.6e-04 | 1.6e-04 |
| Adaptive: MLP ensemble | 1.7e-04 | 1.9e-04 |

## Stage B: double barrier — RMSE of T, reconstruction = PCHIP (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 1.9e-01 | 8.6e-02 | 5.2e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 |
| Random | 1.6e-01 | 1.2e-01 | 1.2e-01 | 7.3e-02 | 3.2e-02 | 2.4e-02 |
| Adaptive: segment (E, T) | 1.9e-01 | 8.7e-02 | 6.8e-02 | 2.4e-02 | 2.4e-02 | 2.4e-02 |
| Adaptive: segment (E, t) | 1.9e-01 | 8.7e-02 | 2.4e-02 | 2.4e-02 | 1.7e-04 | 3.7e-05 |
| Adaptive: GP std | 1.9e-01 | 6.4e-02 | 6.6e-02 | 6.8e-02 | 3.2e-02 | 1.8e-02 |
| Adaptive: MLP ensemble | 1.9e-01 | 9.2e-02 | 7.7e-02 | 2.5e-02 | 2.4e-02 | — |

## Stage B: double barrier — RMSE of T, reconstruction = Cubic spline (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 2.0e-01 | 8.9e-02 | 5.9e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 |
| Random | 2.9e-01 | 1.2e-01 | 1.3e-01 | 6.7e-02 | 2.7e-02 | 2.4e-02 |
| Adaptive: segment (E, T) | 2.0e-01 | 8.7e-02 | 8.8e-02 | 2.4e-02 | 2.4e-02 | 2.4e-02 |
| Adaptive: segment (E, t) | 2.0e-01 | 8.8e-02 | 2.4e-02 | 2.4e-02 | 2.2e-03 | 3.8e-05 |
| Adaptive: GP std | 2.0e-01 | 6.9e-02 | 6.5e-02 | 6.8e-02 | 3.5e-02 | 1.9e-02 |
| Adaptive: MLP ensemble | 2.0e-01 | 9.6e-02 | 7.3e-02 | 2.4e-02 | 2.4e-02 | — |

## Stage B: double barrier — RMSE of T, reconstruction = Gaussian process (amplitude)

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 1.9e-01 | 8.8e-02 | 5.9e-02 | 4.5e-02 | 2.8e-02 | 1.9e-02 |
| Random | 1.8e-01 | 1.3e-01 | 1.9e-01 | 1.2e-01 | 6.2e-02 | 2.9e-02 |
| Adaptive: segment (E, T) | 1.9e-01 | 8.4e-02 | 8.2e-02 | 8.9e-02 | 2.5e-02 | 9.8e-02 |
| Adaptive: segment (E, t) | 1.9e-01 | 8.4e-02 | 1.3e-01 | 9.8e-02 | 1.5e-01 | 1.9e-01 |
| Adaptive: GP std | 1.9e-01 | 7.0e-02 | 6.6e-02 | 6.8e-02 | 3.6e-02 | 2.0e-02 |
| Adaptive: MLP ensemble | 1.9e-01 | 9.1e-02 | 7.1e-02 | 3.7e-02 | 3.4e-02 | — |

## Stage B: double barrier — RMSE of T, reconstruction = MLP (PyTorch) (amplitude)

| strategy | N=80 | N=160 |
|---|---|---|
| Uniform grid | 4.7e-02 | 2.8e-02 |
| Random | 3.8e-02 | 2.5e-02 |
| Adaptive: segment (E, T) | 2.5e-02 | 2.4e-02 |
| Adaptive: segment (E, t) | 2.4e-02 | 2.9e-02 |
| Adaptive: GP std | 8.7e-02 | 3.0e-02 |
| Adaptive: MLP ensemble | 2.5e-02 | 2.4e-02 |

## Stage B — resonances reproduced (of 3), PCHIP reconstruction

| strategy | N=10 | N=20 | N=40 | N=80 | N=160 | N=320 |
|---|---|---|---|---|---|---|
| Uniform grid | 0.0 | 1.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| Random | 0.6 | 0.6 | 0.8 | 1.6 | 2.0 | 2.0 |
| Adaptive: segment (E, T) | 0.0 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 |
| Adaptive: segment (E, t) | 0.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 |
| Adaptive: GP std | 0.0 | 1.0 | 1.0 | 1.0 | 3.0 | 3.0 |
| Adaptive: MLP ensemble | 0.0 | 1.0 | 1.0 | 2.0 | 2.0 | — |

## Evaluations needed (Stage B, PCHIP reconstruction)

| strategy | all 3 resonances detected | RMSE ≤ 1e-2 | RMSE ≤ 1e-3 |
|---|---|---|---|
| Uniform grid | > 320 | > 320 | > 320 |
| Random | > 320 | > 320 | > 320 |
| Adaptive: segment (E, T) | > 320 | > 320 | > 320 |
| Adaptive: segment (E, t) | 160 | 160 | 160 |
| Adaptive: GP std | 160 | > 320 | > 320 |
| Adaptive: MLP ensemble | > 160 | > 160 | > 160 |

## Acquisition overhead (wall-clock spent choosing points, cumulative to N)

| problem | strategy | N | acquisition time [s] |
|---|---|---|---|
| single_barrier | Adaptive: segment (E, T) | 320 | 0.0256 |
| single_barrier | Adaptive: segment (E, t) | 320 | 0.0199 |
| single_barrier | Adaptive: GP std | 320 | 244 |
| single_barrier | Adaptive: MLP ensemble | 160 | 437 |
| double_barrier | Adaptive: segment (E, T) | 320 | 0.0173 |
| double_barrier | Adaptive: segment (E, t) | 320 | 0.0199 |
| double_barrier | Adaptive: GP std | 320 | 198 |
| double_barrier | Adaptive: MLP ensemble | 160 | 386 |
