# Reproduction of headline numbers (scripts/reproduce_key_results.py)

| quantity | stated in report | recomputed | status |
|---|---|---|---|
| Stage A, cubic spline of T, N=10 | 6.2e-05 | 6.199e-05 | PASS |
| Stage B, PCHIP of t, uniform N=640 | 0.018 | 0.01804 | PASS |
| Stage B, PCHIP of t, uniform N=160 | 0.028 | 0.02784 | PASS |
| Stage B, AAA of t(k), uniform N=20 | 1.8e-07 | 1.825e-07 | PASS |
| Stage B, AAA of t(k), uniform N=40 | 3.2e-12 | 3.216e-12 | PASS |
| resonance E_n = 0.993817 | 0.9938 | 0.9938 | PASS |
|   FWHM | 0.00747 | 0.007471 | PASS |
|   pole width -2 Im E_pole | 0.00747 | 0.00747 | PASS |
| resonance E_n = 3.886801 | 3.887 | 3.887 | PASS |
|   FWHM | 0.0874 | 0.08742 | PASS |
|   pole width -2 Im E_pole | 0.08729 | 0.08729 | PASS |
| resonance E_n = 8.38404 | 8.384 | 8.384 | PASS |
|   FWHM | 0.532 | 0.5319 | PASS |
|   pole width -2 Im E_pole | 0.5185 | 0.5185 | PASS |
| Stage B unitarity max|T+R-1| | 6.8e-14 | 6.806e-14 | PASS |
| Exp.4 hybrid, B=2048, median test RMSE | 4.5e-05 | 4.521e-05 | PASS |
| Exp.4 per-configuration AAA, 16 evals, median test RMSE | 6.3e-10 | 6.328e-10 | PASS |
