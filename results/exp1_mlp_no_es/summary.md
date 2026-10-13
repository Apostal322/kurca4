# MLP ablation: early stopping + refit vs. fixed 20 000 epochs (uniform design)

Median RMSE_T over seeds 0–2 (min–max in brackets).

| problem | target | N | early stopping + refit | fixed 20k epochs, no validation split |
|---|---|---|---|---|
| single_barrier | amplitude | 10 | 1.4e-03 [6.3e-04–4.1e-03] | 5.6e-04 [5.3e-04–6.0e-04] |
| single_barrier | amplitude | 20 | 3.0e-04 [2.2e-04–6.0e-04] | 1.4e-04 [1.4e-04–1.7e-04] |
| single_barrier | amplitude | 40 | 9.8e-04 [6.6e-04–1.6e-03] | 1.1e-04 [9.0e-05–1.4e-04] |
| single_barrier | amplitude | 80 | 1.0e-03 [1.5e-04–1.9e-03] | 1.2e-04 [1.1e-04–1.5e-04] |
| single_barrier | amplitude | 160 | 1.1e-04 [1.0e-04–1.4e-04] | 1.1e-04 [1.0e-04–1.4e-04] |
| single_barrier | amplitude | 320 | 1.1e-04 [9.9e-05–1.4e-04] | 1.1e-04 [9.9e-05–1.4e-04] |
| single_barrier | T | 10 | 2.4e-04 [2.2e-04–3.7e-04] | 1.0e-04 [2.6e-05–1.1e-04] |
| single_barrier | T | 20 | 1.0e-04 [9.2e-05–2.1e-04] | 5.9e-05 [1.4e-05–6.6e-05] |
| single_barrier | T | 40 | 1.4e-04 [5.6e-05–1.6e-04] | 5.6e-05 [1.3e-05–6.0e-05] |
| single_barrier | T | 80 | 1.3e-04 [5.6e-05–3.0e-04] | 5.6e-05 [1.4e-05–6.1e-05] |
| single_barrier | T | 160 | 1.3e-04 [5.7e-05–1.9e-04] | 5.7e-05 [1.5e-05–7.0e-05] |
| single_barrier | T | 320 | 5.8e-05 [1.5e-05–1.6e-04] | 5.8e-05 [1.5e-05–6.9e-05] |
| double_barrier | amplitude | 10 | 1.9e-01 [1.9e-01–2.0e-01] | 1.9e-01 [1.9e-01–1.9e-01] |
| double_barrier | amplitude | 20 | 8.3e-02 [8.3e-02–9.6e-02] | 9.4e-02 [8.5e-02–9.7e-02] |
| double_barrier | amplitude | 40 | 4.5e-02 [4.4e-02–6.2e-02] | 4.4e-02 [4.2e-02–4.4e-02] |
| double_barrier | amplitude | 80 | 4.7e-02 [4.4e-02–5.1e-02] | 5.3e-02 [4.7e-02–5.7e-02] |
| double_barrier | amplitude | 160 | 2.8e-02 [2.5e-02–3.4e-02] | 2.7e-02 [2.5e-02–3.1e-02] |
| double_barrier | amplitude | 320 | 2.5e-02 [2.5e-02–3.0e-02] | 2.5e-02 [2.4e-02–2.5e-02] |
| double_barrier | T | 10 | 1.7e-01 [1.7e-01–1.7e-01] | 1.7e-01 [1.6e-01–1.7e-01] |
| double_barrier | T | 20 | 8.5e-02 [8.2e-02–1.0e-01] | 8.7e-02 [8.4e-02–9.9e-02] |
| double_barrier | T | 40 | 5.9e-02 [5.7e-02–6.4e-02] | 5.9e-02 [5.7e-02–6.0e-02] |
| double_barrier | T | 80 | 4.1e-02 [3.7e-02–4.5e-02] | 3.9e-02 [3.5e-02–4.4e-02] |
| double_barrier | T | 160 | 6.5e-02 [6.1e-02–6.8e-02] | 2.5e-02 [2.4e-02–3.5e-02] |
| double_barrier | T | 320 | 2.7e-02 [2.4e-02–6.8e-02] | 2.4e-02 [2.3e-02–3.6e-02] |
