# Experiment 1 — random designs: median RMSE_T with 95 % bootstrap CI over seeds

Percentile bootstrap of the median (10 000 resamples). With 5 seeds (3 for the MLP) the intervals are wide; they are reported so that differences smaller than the spread are not over-interpreted.

| problem | method | target | N=20 | N=40 | N=160 |
|---|---|---|---|---|---|
| single_barrier | Linear | amplitude | 7.4e-03 [3.6e-03, 1.1e-02] | 1.2e-03 [6.6e-04, 2.1e-03] | 1.0e-04 [7.8e-05, 1.8e-04] |
| single_barrier | Linear | T | 3.3e-03 [2.3e-03, 6.6e-03] | 9.9e-04 [4.3e-04, 1.3e-03] | 7.5e-05 [6.2e-05, 1.1e-04] |
| single_barrier | PCHIP | amplitude | 3.8e-03 [8.5e-04, 5.4e-03] | 3.5e-04 [2.5e-04, 8.1e-04] | 2.6e-05 [1.8e-05, 3.8e-05] |
| single_barrier | PCHIP | T | 7.2e-04 [5.4e-04, 2.0e-03] | 2.0e-04 [7.9e-05, 3.4e-04] | 2.0e-05 [1.8e-05, 2.9e-05] |
| single_barrier | Cubic spline | amplitude | 3.1e-04 [1.5e-04, 5.5e-04] | 4.6e-05 [4.0e-05, 3.2e-04] | 1.9e-06 [3.6e-07, 3.7e-05] |
| single_barrier | Cubic spline | T | 1.9e-04 [1.5e-05, 2.4e-04] | 6.6e-06 [1.1e-06, 8.6e-06] | 2.6e-08 [1.5e-08, 4.7e-08] |
| single_barrier | Gaussian process | amplitude | 3.3e-04 [2.1e-04, 2.1e-03] | 6.3e-05 [1.6e-05, 1.6e-04] | 1.6e-06 [8.2e-07, 7.4e-06] |
| single_barrier | Gaussian process | T | 2.8e-06 [1.3e-06, 1.7e-05] | 9.2e-07 [8.4e-07, 1.4e-06] | 4.5e-07 [4.2e-07, 4.6e-07] |
| single_barrier | MLP (PyTorch) | amplitude | 1.4e-03 [1.8e-04, 4.2e-03] | 6.5e-04 [1.2e-04, 7.9e-04] | 1.1e-04 [1.1e-04, 1.4e-04] |
| single_barrier | MLP (PyTorch) | T | 2.5e-04 [2.3e-04, 2.8e-04] | 8.1e-05 [5.4e-05, 1.2e-04] | 1.3e-04 [5.9e-05, 1.3e-04] |
| single_barrier | Rational (AAA, in k) | amplitude | 1.5e-13 [4.4e-14, 5.3e-12] | 1.9e-14 [3.6e-16, 4.7e-13] | 1.6e-15 [4.3e-16, 2.6e-15] |
| single_barrier | Rational (AAA, in k) | T | 2.8e-11 [3.5e-12, 1.2e-08] | 3.0e-12 [1.1e-15, 1.3e-10] | 1.3e-15 [9.5e-16, 4.1e-15] |
| double_barrier | Linear | amplitude | 1.3e-01 [9.0e-02, 1.6e-01] | 1.2e-01 [9.2e-02, 1.7e-01] | 3.7e-02 [2.6e-02, 4.0e-02] |
| double_barrier | Linear | T | 1.2e-01 [9.9e-02, 1.5e-01] | 1.2e-01 [7.8e-02, 1.4e-01] | 3.0e-02 [2.5e-02, 3.4e-02] |
| double_barrier | PCHIP | amplitude | 1.2e-01 [8.7e-02, 1.6e-01] | 1.2e-01 [8.3e-02, 1.6e-01] | 3.2e-02 [2.6e-02, 3.5e-02] |
| double_barrier | PCHIP | T | 1.1e-01 [1.0e-01, 1.4e-01] | 1.1e-01 [7.4e-02, 1.4e-01] | 2.8e-02 [2.2e-02, 3.5e-02] |
| double_barrier | Cubic spline | amplitude | 1.2e-01 [9.3e-02, 7.3e-01] | 1.3e-01 [8.1e-02, 6.2e-01] | 2.7e-02 [2.5e-02, 4.4e-02] |
| double_barrier | Cubic spline | T | 1.3e-01 [9.0e-02, 3.6e-01] | 1.1e-01 [7.0e-02, 3.6e-01] | 2.6e-02 [2.0e-02, 5.1e-02] |
| double_barrier | Gaussian process | amplitude | 1.3e-01 [7.9e-02, 1.9e-01] | 1.9e-01 [7.3e-02, 1.2e+00] | 6.2e-02 [2.5e-02, 6.8e-02] |
| double_barrier | Gaussian process | T | 1.1e-01 [8.8e-02, 1.6e-01] | 1.7e-01 [6.8e-02, 5.4e-01] | 3.8e-02 [2.8e-02, 6.8e-02] |
| double_barrier | MLP (PyTorch) | amplitude | 9.6e-02 [8.2e-02, 1.2e-01] | 8.4e-02 [6.8e-02, 1.9e-01] | 3.2e-02 [2.4e-02, 3.3e-02] |
| double_barrier | MLP (PyTorch) | T | 1.2e-01 [9.1e-02, 1.3e-01] | 8.6e-02 [8.5e-02, 1.4e-01] | 3.3e-02 [2.6e-02, 7.2e-02] |
| double_barrier | Rational (AAA, in k) | amplitude | 2.0e-06 [1.2e-09, 2.4e-05] | 2.7e-11 [1.8e-14, 1.8e-08] | 8.7e-15 [4.8e-15, 3.5e-13] |
| double_barrier | Rational (AAA, in k) | T | 2.3e-02 [2.6e-04, 2.7e-02] | 1.3e-06 [3.9e-10, 5.6e-04] | 4.3e-12 [2.4e-14, 1.0e-09] |
