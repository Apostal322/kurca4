# Experiment 5 — independent stress test

Systems never used for method development (seed 20261013): asymmetric double barriers, coupled-well triple barriers (resonance doublets), smooth double-Gaussian barriers (4000-layer staircase, 1.6e-6 from the smooth-potential ODE solution), wide intervals with above-barrier resonances, and solver noise.
Equal exact-evaluation budgets N. Reconstructed = max|T̂−T| within ±2Γ of the pole ≤ 10 % of the true peak. Spurious = peak of T̂ (prominence > 0.05) farther than 3Γ from every exact pole.

## asymmetric double barrier (15 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 37/37 · 3.4e-09 · 0 | 37/37 · 4.6e-14 · 0 | 37/37 · 5.2e-15 · 0 | 37/37 · 1.4e-15 · 0 |
| AAA greedy (look-ahead) | 37/37 · 2.5e-11 · 0 | 37/37 · 2.8e-12 · 0 | 37/37 · 2.8e-12 · 0 | 37/37 · 2.8e-12 · 0 |
| segment(E,t) + PCHIP(t) | 15/37 · 2.5e-02 · 0 | 27/37 · 8.8e-03 · 0 | 31/37 · 4.6e-04 · 0 | 35/37 · 4.9e-05 · 0 |
| uniform + cubic spline(T) | 2/37 · 5.7e-02 · 0 | 11/37 · 3.4e-02 · 1 | 15/37 · 2.2e-02 · 0 | 21/37 · 7.4e-03 · 1 |
| uniform + GP(t) | 2/37 · 5.9e-02 · 0 | 10/37 · 4.1e-02 · 0 | 13/37 · 1.9e-02 · 0 | 21/37 · 5.4e-03 · 0 |
| hybrid: block S-matrices + exact composition | 24/37 · 4.7e-03 · 0 | 30/37 · 9.6e-04 · 0 | 37/37 · 1.6e-05 · 0 | 37/37 · 1.9e-07 · 0 |
| uniform + MLP(t) | — | 3/11 · 3.3e-02 · 0 | — | 6/11 · 1.3e-02 · 0 |
| uniform + Fourier-feature MLP(t) | — | 4/11 · 2.8e-02 · 0 | — | 6/11 · 4.1e-03 · 0 |

## triple barrier (resonance doublets) (4 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 11/16 · 1.1e-02 · 0 | 16/16 · 2.4e-10 · 0 | 16/16 · 9.0e-13 · 0 | 16/16 · 1.4e-14 · 0 |
| AAA greedy (look-ahead) | 16/16 · 5.3e-07 · 0 | 16/16 · 1.2e-11 · 0 | 16/16 · 1.2e-11 · 0 | 16/16 · 1.2e-11 · 0 |
| segment(E,t) + PCHIP(t) | 6/16 · 5.1e-02 · 0 | 8/16 · 4.8e-02 · 0 | 8/16 · 4.8e-02 · 0 | 11/16 · 2.9e-02 · 0 |
| uniform + cubic spline(T) | 0/16 · 1.1e-01 · 0 | 0/16 · 7.0e-02 · 1 | 2/16 · 5.2e-02 · 0 | 8/16 · 3.8e-02 · 0 |
| uniform + GP(t) | 0/16 · 1.2e-01 · 0 | 0/16 · 7.8e-02 · 0 | 1/16 · 5.6e-02 · 0 | 3/16 · 3.4e-02 · 0 |
| hybrid: block S-matrices + exact composition | 8/16 · 4.8e-02 · 0 | 10/16 · 3.8e-03 · 0 | 16/16 · 2.1e-05 · 0 | 16/16 · 5.0e-07 · 0 |
| uniform + MLP(t) | — | 0/8 · 7.1e-02 · 0 | — | 2/8 · 3.8e-02 · 0 |
| uniform + Fourier-feature MLP(t) | — | 0/8 · 7.2e-02 · 0 | — | 2/8 · 3.0e-02 · 0 |

## smooth double-Gaussian barrier (2 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 5/5 · 2.1e-06 · 0 | 5/5 · 3.7e-11 · 0 | 5/5 · 6.1e-13 · 0 | 5/5 · 1.8e-13 · 0 |
| AAA greedy (look-ahead) | 5/5 · 1.9e-08 · 0 | 5/5 · 1.3e-11 · 0 | 5/5 · 1.3e-11 · 0 | 5/5 · 1.3e-11 · 0 |
| segment(E,t) + PCHIP(t) | 2/5 · 3.7e-02 · 0 | 2/5 · 3.7e-02 · 0 | 3/5 · 6.1e-03 · 0 | 4/5 · 3.6e-03 · 0 |
| uniform + cubic spline(T) | 0/5 · 9.4e-02 · 0 | 0/5 · 4.8e-02 · 0 | 2/5 · 6.0e-02 · 0 | 2/5 · 2.9e-02 · 0 |
| uniform + GP(t) | 0/5 · 1.1e-01 · 0 | 0/5 · 5.1e-02 · 0 | 2/5 · 5.0e-02 · 0 | 2/5 · 3.2e-02 · 1 |
| hybrid: block S-matrices + exact composition | — | — | — | — |
| uniform + MLP(t) | — | 0/5 · 5.2e-02 · 0 | — | 1/5 · 3.6e-02 · 0 |
| uniform + Fourier-feature MLP(t) | — | 0/5 · 4.8e-02 · 0 | — | 2/5 · 2.8e-02 · 0 |

## wide interval (above-barrier resonances) (2 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 2/13 · 9.4e-02 · 0 | 12/13 · 5.8e-04 · 0 | 13/13 · 3.8e-08 · 0 | 13/13 · 9.3e-12 · 0 |
| AAA greedy (look-ahead) | 9/13 · 2.5e-02 · 0 | 13/13 · 3.0e-10 · 0 | 13/13 · 3.0e-10 · 0 | 13/13 · 3.0e-10 · 0 |
| segment(E,t) + PCHIP(t) | 3/13 · 8.4e-02 · 0 | 8/13 · 4.3e-02 · 0 | 9/13 · 2.5e-02 · 0 | 11/13 · 1.2e-02 · 0 |
| uniform + cubic spline(T) | 0/13 · 1.1e-01 · 0 | 4/13 · 7.4e-02 · 0 | 4/13 · 8.7e-02 · 0 | 8/13 · 3.7e-02 · 0 |
| uniform + GP(t) | 0/13 · 1.2e-01 · 0 | 4/13 · 7.6e-02 · 0 | 4/13 · 6.8e-02 · 0 | 8/13 · 3.1e-02 · 0 |
| hybrid: block S-matrices + exact composition | 8/13 · 2.0e-02 · 0 | 11/13 · 9.7e-03 · 0 | 12/13 · 4.1e-03 · 0 | 13/13 · 3.6e-04 · 0 |
| uniform + MLP(t) | — | 0/4 · 5.5e-02 · 0 | — | 1/4 · 4.8e-02 · 0 |
| uniform + Fourier-feature MLP(t) | — | 1/4 · 6.7e-02 · 0 | — | 2/4 · 2.1e-02 · 0 |

## asymmetric + noise 1e-08 (2 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 5/5 · 1.1e-06 · 0 | 5/5 · 5.7e-09 · 0 | 5/5 · 4.5e-09 · 0 | 5/5 · 5.4e-09 · 0 |
| AAA greedy (look-ahead) | 5/5 · 6.5e-07 · 0 | 5/5 · 6.5e-07 · 0 | 5/5 · 6.5e-07 · 0 | 5/5 · 6.5e-07 · 0 |
| segment(E,t) + PCHIP(t) | 1/5 · 2.0e-02 · 0 | 4/5 · 6.1e-03 · 0 | 5/5 · 4.9e-04 · 0 | 5/5 · 7.0e-05 · 0 |
| uniform + cubic spline(T) | 0/5 · 1.0e-01 · 0 | 2/5 · 3.3e-02 · 0 | 3/5 · 2.5e-02 · 0 | 4/5 · 6.0e-03 · 0 |
| uniform + GP(t) | 0/5 · 7.7e-02 · 0 | 2/5 · 3.1e-02 · 0 | 2/5 · 3.3e-02 · 0 | 4/5 · 4.8e-03 · 0 |
| hybrid: block S-matrices + exact composition | — | — | — | — |
| uniform + MLP(t) | — | — | — | — |
| uniform + Fourier-feature MLP(t) | — | — | — | — |

## asymmetric + noise 0.0001 (2 systems)

| method | N=20: resonances / median RMSE(T) / spurious | N=40: resonances / median RMSE(T) / spurious | N=80: resonances / median RMSE(T) / spurious | N=160: resonances / median RMSE(T) / spurious |
|---|---|---|---|---|
| uniform + AAA(t) | 4/5 · 4.7e-04 · 0 | 5/5 · 4.1e-05 · 0 | 5/5 · 4.1e-05 · 0 | 5/5 · 3.8e-05 · 0 |
| AAA greedy (look-ahead) | 5/5 · 2.2e-03 · 0 | 5/5 · 2.2e-03 · 0 | 5/5 · 2.2e-03 · 0 | 5/5 · 2.2e-03 · 0 |
| segment(E,t) + PCHIP(t) | 1/5 · 2.0e-02 · 0 | 4/5 · 6.1e-03 · 0 | 5/5 · 4.9e-04 · 0 | 5/5 · 7.5e-05 · 0 |
| uniform + cubic spline(T) | 0/5 · 1.0e-01 · 0 | 2/5 · 3.3e-02 · 0 | 3/5 · 2.5e-02 · 0 | 4/5 · 6.0e-03 · 0 |
| uniform + GP(t) | 0/5 · 7.7e-02 · 0 | 2/5 · 3.1e-02 · 0 | 2/5 · 3.3e-02 · 0 | 4/5 · 4.8e-03 · 0 |
| hybrid: block S-matrices + exact composition | — | — | — | — |
| uniform + MLP(t) | — | — | — | — |
| uniform + Fourier-feature MLP(t) | — | — | — | — |

## Rational approximant diagnostics (uniform AAA and greedy AAA, all systems)

Poles with Re k inside the window. matched = within 0.1Γ of an exact pole; Froissart = residue < 1e-10·max|t| or a zero within 1e-3·|Im pole|; upper-half = Im k > 0 (non-physical here, no bound states); spurious near axis = unmatched, |Im E| < 0.05·W and Re E in the interval; missed = true resonance without a matching pole. Sensitivity = max change of T̂ when the data are perturbed by 1e-10 (relative).

| method | N | fits | poles matched | Froissart | upper-half | spurious near axis | missed resonances | median pole E error / Γ | median pole Γ rel. error | median / max sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform + AAA(t) | 20 | 27 | 71 | 10 | 4 | 7 | 12 | 4.4e-09 | 1.5e-08 | 1.8e-06 / 2.2e-03 |
| uniform + AAA(t) | 40 | 27 | 83 | 2 | 1 | 0 | 0 | 3.0e-13 | 6.7e-13 | 1.5e-09 / 1.1e+00 |
| uniform + AAA(t) | 80 | 27 | 83 | 5 | 1 | 0 | 0 | 6.4e-14 | 1.3e-13 | 7.3e-10 / 1.4e-04 |
| uniform + AAA(t) | 160 | 27 | 83 | 23 | 2 | 1 | 0 | 5.2e-14 | 7.6e-14 | 1.2e-09 / 5.8e-07 |
| AAA greedy (look-ahead) | 20 | 27 | 77 | 14 | 5 | 1 | 6 | 9.8e-10 | 1.7e-09 | 1.2e-09 / 9.2e-07 |
| AAA greedy (look-ahead) | 40 | 27 | 83 | 9 | 4 | 0 | 0 | 1.7e-12 | 1.0e-11 | 7.3e-10 / 4.9e-08 |
| AAA greedy (look-ahead) | 80 | 27 | 83 | 9 | 4 | 0 | 0 | 1.7e-12 | 1.0e-11 | 7.3e-10 / 4.9e-08 |
| AAA greedy (look-ahead) | 160 | 27 | 83 | 9 | 4 | 0 | 0 | 1.7e-12 | 1.0e-11 | 7.3e-10 / 4.9e-08 |

Fits with at least one spurious near-axis pole: 6; of these, with a spurious *peak* in T̂: 0; median RMSE of these fits: 1.8e-02.

## Greedy self-certification (budget 160)

Certified in 27/27 systems. At certification: all resonances reconstructed in 27/27; max true RMSE(T) 4.5e-03; median evaluations used 22.

- asym_00: certified at 21, RMSE 7.4e-11, 3/3 resonances
- asym_00_noise0.0001: certified at 16, RMSE 2.4e-05, 3/3 resonances
- asym_00_noise1e-08: certified at 18, RMSE 7.9e-08, 3/3 resonances
- asym_01: certified at 23, RMSE 2.2e-11, 2/2 resonances
- asym_01_noise0.0001: certified at 13, RMSE 4.5e-03, 2/2 resonances
- asym_01_noise1e-08: certified at 17, RMSE 1.2e-06, 2/2 resonances
- asym_02: certified at 24, RMSE 2.5e-13, 3/3 resonances
- asym_03: certified at 27, RMSE 1.7e-12, 3/3 resonances
- asym_04: certified at 18, RMSE 1.7e-11, 2/2 resonances
- asym_05: certified at 26, RMSE 4.2e-14, 3/3 resonances
- asym_06: certified at 22, RMSE 3.8e-12, 3/3 resonances
- asym_07: certified at 23, RMSE 4.1e-14, 2/2 resonances
- asym_08: certified at 22, RMSE 5.3e-14, 2/2 resonances
- asym_09: certified at 20, RMSE 2.8e-10, 3/3 resonances
- asym_10: certified at 21, RMSE 2.8e-12, 2/2 resonances
- asym_11: certified at 21, RMSE 1.1e-11, 2/2 resonances
- asym_12: certified at 22, RMSE 7.8e-14, 2/2 resonances
- asym_13: certified at 24, RMSE 8.6e-11, 3/3 resonances
- asym_14: certified at 21, RMSE 6.5e-13, 2/2 resonances
- smooth_gauss_A: certified at 20, RMSE 2.3e-12, 2/2 resonances
- smooth_gauss_B: certified at 24, RMSE 2.3e-11, 3/3 resonances
- triple_amid0.5: certified at 28, RMSE 1.1e-13, 4/4 resonances
- triple_amid0.9: certified at 27, RMSE 2.4e-11, 4/4 resonances
- triple_amid1.3: certified at 28, RMSE 7.0e-11, 4/4 resonances
- triple_amid1.7: certified at 29, RMSE 5.5e-13, 4/4 resonances
- wide_0.1_30: certified at 29, RMSE 5.9e-10, 4/4 resonances
- wide_0.1_60: certified at 39, RMSE 4.5e-12, 9/9 resonances

## Measured cost per fit (median over systems, N = 160; MLPs N = 160 where run)

| method | solver [s] | acquisition [s] | fit [s] | predict on ref grid [s] |
|---|---|---|---|---|
| uniform + AAA(t) | 0.000463 | 0 | 0.00573 | 0.241 |
| AAA greedy (look-ahead) | 0.0057 | 0.102 | 0.00303 | 0.0221 |
| segment(E,t) + PCHIP(t) | 0.0469 | 0.0115 | 0.000696 | 0.000838 |
| uniform + cubic spline(T) | 0.000398 | 0 | 0.00035 | 0.000339 |
| uniform + GP(t) | 0.0004 | 0 | 3.61 | 0.806 |
| hybrid: block S-matrices + exact composition | 0 | 0 | 0.0137 | 0 |
| uniform + MLP(t) | 0.000499 | 0 | 16.4 | 0.0667 |
| uniform + Fourier-feature MLP(t) | 0.000483 | 0 | 40.1 | 0.163 |

## Pole width vs FWHM (all resonances of all stress-test systems)

73 resonances. |FWHM/Γ_pole − 1| grows with x = max(Γ/D, Γ/E_r) with fitted exponent 1.68 (log-log least squares). For x < 1e-2 the median deviation is 7.8e-05; overall median 1.0e-02, max 1.6e+00. Peak offset (E_peak − Re E_pole)/Γ: median 2.6e-02, max 1.80.

| system | E_pole | Γ_pole | FWHM | FWHM/Γ_pole | (E_peak−E_pole)/Γ | T_peak | Γ/D | Γ/E |
|---|---|---|---|---|---|---|---|---|
| triple_amid1.3 | 1.3863 | 1.389e-02 | 3.600e-02 | 2.5907 | +1.798 | 1.000 | 5.14e-01 | 1.00e-02 |
| triple_amid1.3 | 1.4133 | 1.455e-02 | 3.600e-02 | 2.4748 | -0.139 | 1.000 | 5.39e-01 | 1.03e-02 |
| triple_amid1.3 | 5.2848 | 1.488e-01 | 2.765e-01 | 1.8589 | +1.053 | 1.000 | 7.58e-01 | 2.81e-02 |
| triple_amid1.3 | 5.4810 | 1.692e-01 | 2.765e-01 | 1.6342 | -0.234 | 1.000 | 8.62e-01 | 3.09e-02 |
| triple_amid0.9 | 5.1616 | 1.406e-01 | 1.681e-01 | 1.1955 | +0.097 | 1.000 | 3.02e-01 | 2.72e-02 |
| triple_amid1.7 | 5.4216 | 1.632e-01 | 1.367e-01 | 0.8379 | -0.244 | 0.679 | 1.97e+00 | 3.01e-02 |
| triple_amid0.9 | 5.6269 | 1.829e-01 | 2.104e-01 | 1.1504 | -0.073 | 1.000 | 3.93e-01 | 3.25e-02 |
| triple_amid1.7 | 5.3387 | 1.534e-01 | 1.367e-01 | 0.8916 | +0.281 | 0.679 | 1.85e+00 | 2.87e-02 |
| asym_12 | 5.8761 | 8.887e-01 | 9.769e-01 | 1.0993 | +0.077 | 0.069 | 2.02e-01 | 1.51e-01 |
| triple_amid1.7 | 1.4041 | 1.432e-02 | 1.305e-02 | 0.9108 | -0.290 | 0.763 | 1.71e+00 | 1.02e-02 |
| triple_amid1.7 | 1.3957 | 1.411e-02 | 1.305e-02 | 0.9243 | +0.298 | 0.763 | 1.69e+00 | 1.01e-02 |
| asym_14 | 8.2382 | 1.095e+00 | 1.175e+00 | 1.0730 | +0.067 | 0.440 | 1.79e-01 | 1.33e-01 |

