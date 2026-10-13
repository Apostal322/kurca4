# Measured computational cost (Task 5)

Stage B double barrier, target: T(E) on a 20 001-point grid. Single-thread CPU. Per-call cost of a single-energy solve: S-matrix 103.7 µs, ODE 3.05 ms (the S-matrix solver is ~29x faster per call and additionally vectorises: 0.53 µs/energy for the full scan).

| solver | method | exact calls | solver [s] | acquisition [s] | training/fit [s] | inference on 20 001 E [s] | total [s] | RMSE(T) | wall-clock speed-up vs direct scan | break-even #queries |
|---|---|---|---|---|---|---|---|---|---|---|
| S-matrix (vectorised) | direct scan (20 001 calls) | 20001 | 0.0107 | 0 | 0 | 0 | 0.0107 | 0.0e+00 | 1x | — |
| S-matrix (vectorised) | uniform 30 + AAA(t) | 30 | 0.000278 | 0 | 0.00361 | 0.00642 | 0.0103 | 2.1e-11 | 1x | 38 |
| S-matrix (vectorised) | AAA greedy (self-certified at N=24) | 24 | 0.00317 | 0.0503 | 0.00199 | 0.00557 | 0.0611 | 1.2e-12 | 0.17x | 537 |
| S-matrix (vectorised) | segment(E,t) 160 + PCHIP(t) | 160 | 0.019 | 0.00697 | 0.000736 | 0.000793 | 0.0275 | 1.7e-04 | 0.39x | 257 |
| S-matrix (vectorised) | uniform 160 + MLP(t) | 160 | 0.000264 | 0 | 25.1 | 0.0712 | 25.2 | 2.5e-02 | 0.00042x | 250756 |
| ODE DOP853 | direct scan (20 001 calls) | 20001 | 59.4 | 0 | 0 | 0 | 59.4 | 8.6e-12 | 1x | — |
| ODE DOP853 | uniform 30 + AAA(t) | 30 | 0.0801 | 0 | 0.00407 | 0.00465 | 0.0889 | 5.4e-10 | 6.7e+02x | 28 |
| ODE DOP853 | AAA greedy (self-certified at N=24) | 24 | 0.0648 | 0.0546 | 0.00215 | 0.00312 | 0.125 | 8.7e-12 | 4.8e+02x | 40 |
| ODE DOP853 | segment(E,t) 160 + PCHIP(t) | 160 | 0.668 | 0.0195 | 0.000843 | 0.000842 | 0.689 | 1.7e-04 | 86x | 226 |
| ODE DOP853 | uniform 160 + MLP(t) | 160 | 0.694 | 0 | 21.8 | 0.0964 | 22.6 | 2.6e-02 | 2.6x | 7389 |

Break-even #queries = number of single-energy queries after which building the surrogate (solver + acquisition + fit) plus answering the queries by inference is cheaper than calling the solver for each query. A speed-up below 1x means the surrogate pipeline is slower than the direct scan.
