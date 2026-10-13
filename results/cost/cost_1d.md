# Measured computational cost (Task 5)

Stage B double barrier, target: T(E) on a 20 001-point grid. Single-thread CPU. Per-call cost of a single-energy solve: S-matrix 164.3 µs, ODE 4.36 ms (the S-matrix solver is ~27x faster per call and additionally vectorises: 0.61 µs/energy for the full scan).

| solver | method | exact calls | solver [s] | acquisition [s] | training/fit [s] | inference on 20 001 E [s] | total [s] | RMSE(T) | wall-clock speed-up vs direct scan | break-even #queries |
|---|---|---|---|---|---|---|---|---|---|---|
| S-matrix (vectorised) | direct scan (20 001 calls) | 20001 | 0.0123 | 0 | 0 | 0 | 0.0123 | 0.0e+00 | 1x | — |
| S-matrix (vectorised) | uniform 30 + AAA(t) | 30 | 0.000357 | 0 | 0.00424 | 0.00637 | 0.011 | 2.1e-11 | 1.1x | 28 |
| S-matrix (vectorised) | AAA greedy (self-certified at N=24) | 24 | 0.00362 | 0.058 | 0.00265 | 0.00529 | 0.0695 | 1.2e-12 | 0.18x | 392 |
| S-matrix (vectorised) | segment(E,t) 160 + PCHIP(t) | 160 | 0.0255 | 0.00839 | 0.000807 | 0.00067 | 0.0354 | 1.7e-04 | 0.35x | 211 |
| S-matrix (vectorised) | uniform 160 + MLP(t) | 160 | 0.000354 | 0 | 24.5 | 0.0366 | 24.5 | 2.5e-02 | 0.0005x | 150854 |
| ODE DOP853 | direct scan (20 001 calls) | 20001 | 89.5 | 0 | 0 | 0 | 89.5 | 8.6e-12 | 1x | — |
| ODE DOP853 | uniform 30 + AAA(t) | 30 | 0.117 | 0 | 0.00698 | 0.00425 | 0.128 | 5.4e-10 | 7e+02x | 28 |
| ODE DOP853 | AAA greedy (self-certified at N=24) | 24 | 0.0983 | 0.0691 | 0.00295 | 0.00344 | 0.174 | 8.7e-12 | 5.1e+02x | 39 |
| ODE DOP853 | segment(E,t) 160 + PCHIP(t) | 160 | 0.708 | 0.0174 | 0.000748 | 0.000689 | 0.726 | 1.7e-04 | 1.2e+02x | 166 |
| ODE DOP853 | uniform 160 + MLP(t) | 160 | 0.66 | 0 | 22.4 | 0.0322 | 23 | 2.6e-02 | 3.9x | 5275 |

Break-even #queries = number of single-energy queries after which building the surrogate (solver + acquisition + fit) plus answering the queries by inference is cheaper than calling the solver for each query. A speed-up below 1x means the surrogate pipeline is slower than the direct scan.
