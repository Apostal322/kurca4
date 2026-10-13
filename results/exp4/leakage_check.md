# Experiment 4 — configuration-leakage check

Scaled parameter space [-1, 1]^3 of (V0, a, d). 'exact overlaps' = identical configurations; NN distance = distance from each test configuration to the nearest training configuration.

| budget | seed | train cfgs | val cfgs | train∩val | train∩test | val∩test | train∩OOD | test NN distance median / min | typical train spacing |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | 0 | 56 | 8 | 0 | 0 | 0 | 0 | 0.290 / 0.076 | 0.523 |
| 2048 | 1 | 56 | 8 | 0 | 0 | 0 | 0 | 0.318 / 0.044 | 0.523 |
| 2048 | 2 | 56 | 8 | 0 | 0 | 0 | 0 | 0.265 / 0.101 | 0.523 |
| 8192 | 0 | 224 | 32 | 0 | 0 | 0 | 0 | 0.163 / 0.076 | 0.329 |
| 8192 | 1 | 224 | 32 | 0 | 0 | 0 | 0 | 0.171 / 0.024 | 0.329 |
| 8192 | 2 | 224 | 32 | 0 | 0 | 0 | 0 | 0.177 / 0.047 | 0.329 |
| 32768 | 0 | 896 | 128 | 0 | 0 | 0 | 0 | 0.100 / 0.025 | 0.207 |
| 32768 | 1 | 896 | 128 | 0 | 0 | 0 | 0 | 0.118 / 0.024 | 0.207 |
| 32768 | 2 | 896 | 128 | 0 | 0 | 0 | 0 | 0.114 / 0.023 | 0.207 |

Result: no overlaps in any split. Test configurations come from an independent random stream (seed 777) and OOD configurations (seed 888) each have one parameter beyond the domain edge; validation configurations are a disjoint 1/8 of the Sobol training configurations.
