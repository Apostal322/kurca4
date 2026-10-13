# Changelog

All results are kept in the repository history. This file summarises the main stages, including corrections of
earlier, wrong conclusions.

## Unreleased: repository publication

- Professional README, figure gallery, experiment, reproducibility and provenance documentation.
- GitHub Actions CI: ruff, the test suite, and smoke tests on Ubuntu and Windows; a scientific reproduction check.
- `python -m qscatter.quickstart`, a one-minute end-to-end demo.
- Distribution renamed to `quantum-scattering-surrogates`; the import name stays `qscatter`.
- Correct minimum dependency versions (NumPy ≥ 2.0, SciPy ≥ 1.15).
- The build artifact `qscatter.egg-info/` is no longer tracked.
- Reference caches are rebuilt automatically in a fresh clone.

## Final scientific review

- Corrected the statement that resonance poles lie on the physical sheet. They lie on the second (unphysical)
  sheet, Im k < 0.
- Clarified the Γ vs FWHM definition, including the Fano caveat. Added the 1D S-matrix and the 1D vs 3D
  distinction.
- Added `scripts/reproduce_key_results.py` (17/17 reproduced) and a second, independent timing run.

## Stress test and parametric surrogates

- Experiment 4 (parametric surrogate) and Experiment 5 (27 unseen systems). Rational-approximant diagnostics.
- **Correction:** the Experiment 4 spurious-peak metric counted near-duplicate grid points as peaks. It was
  recomputed where possible and withdrawn otherwise; a regression test now covers it.
- **Correction:** FWHM read off grid points gave a spurious 2 % deviation from the pole width. With root
  refinement it is 8·10⁻⁵.
- **Correction:** greedy AAA self-certification is reliable only for noise-free data.

## AAA rational approximation and robustness

- Added AAA rational approximation of t(k), exact complex poles, an independent audit (10 checks), and Experiment 3
  (38 systems).
- **Correction:** the earlier claims "the decisive factor is where the samples are, not the model class" and "an
  interpolant cannot recover an unseen peak" are wrong for rational approximation. Amplitude-driven adaptive
  sampling was demoted from the main result to a secondary, convention-dependent effect.

## Initial study

- Validated S-matrix solver, classical baselines, PyTorch MLP, Experiments 1–2.
- **Correction during the work:** the GP baseline collapsed to a degenerate length scale. It is now bounded by the
  sample spacing.
