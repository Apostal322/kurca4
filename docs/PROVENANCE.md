# Provenance: published algorithms, our implementation choices, and new empirical findings

This file separates three things that are easy to blur in a report:

* **(P) Published.** Algorithms and facts taken from the literature or textbooks.
* **(I) Implementation.** Choices made in this project: parameters, adaptations, glue code. They are
  defensible, but they are not results.
* **(F) Findings.** New empirical observations made here. They are valid only for the systems tested
  and should not be read as general theorems.

## 1. Physics and numerics

| item | type | source / note |
|---|---|---|
| Stationary Schrödinger equation, amplitudes t and r, T + R = 1, rectangular-barrier formula | P | textbooks (Landau–Lifshitz QM; Griffiths) |
| Breit–Wigner form; resonances as poles of the S-matrix at E_r − iΓ/2 | P | scattering theory (Taylor 1972; Landau–Lifshitz) |
| Redheffer star product of scattering matrices | P | Redheffer (1962); its use for stable multilayer tunnelling, e.g. Ko & Inkson (1988) |
| Per-layer propagator written in entire functions of q², with sin(qd)/q = d·sinc | I | Our formulation; it avoids a branch choice and the turning-point degeneracy |
| Analytic continuation to complex E; Newton refinement of 1/t(E) for exact poles | I | Standard numerical technique, our implementation |
| DOP853 ODE cross-check, integrated right to left | P (method) / I (use) | Hairer–Nørsett–Wanner; `scipy.integrate.solve_ivp` |
| Staircase approximation of a smooth potential: O(h²) convergence measured (1.0e-4 → 4.0e-7 for 500 → 8000 layers) | I + F | Our measurement (Experiment 5 setup) |

## 2. Surrogate and sampling methods

| item | type | source / note |
|---|---|---|
| Linear, PCHIP, cubic spline | P | Fritsch & Carlson (1980); de Boor (1978); SciPy |
| Gaussian-process regression, marginal-likelihood kernel selection | P | Rasmussen & Williams (2006); scikit-learn |
| GP length-scale lower bound of half the mean sample spacing | I | Our fix for a degenerate likelihood optimum we observed (F) |
| AAA rational approximation | P | Nakatsukasa, Sète & Trefethen (2018); `scipy.interpolate.AAA` |
| Approximating t as a function of **k** rather than E, and approximating the **complex amplitude** rather than T | I | Motivated by the textbook analytic structure (t meromorphic in k). The choice is ours. |
| Discarding Froissart doublets (negligible residue, or pole–zero pairs) | P (concept) / I (thresholds) | Spurious poles are discussed in the AAA paper; our thresholds are 1e-10·max\|t\| and 1e-3·\|Im pole\| |
| Greedy sampling at argmin \|barycentric denominator\| with a look-ahead error check | P (idea) / I (adaptation) | The greedy MRI loop (gMRI) in the code of Pradovera, Hiptmair & Perugia. Their MRI weights are not determined for scalar outputs, so we use AAA least-squares weights. Our parameters: tolerance 1e-8, three consecutive passes, then bisection of the largest gap. |
| Calibrated error estimate driving a greedy choice | P (principle) | Greedy emulators of Giri et al. (2026). We use only the principle; their Galerkin / residual machinery needs solution vectors, which our scalar-output setting does not provide. |
| Segment-length adaptive sampling √(ΔÊ² + Δy²) | P | Default loss of 1D adaptive-sampling libraries (`adaptive`, Nijholt et al.) |
| Segment sampling on the complex amplitude, or on log T | I | Our variants |
| MLP, Adam, cosine schedule, low-data protocol (interleaved validation, early stopping with refit) | P (components) / I (protocol) | Kingma & Ba; PyTorch. The protocol and its ablation are ours. |
| Fourier features, with σ selected on validation data | P (method) / I (selection) | Tancik et al. (2020) |
| Deep-ensemble disagreement as an acquisition function | P (method) / I (use) | Lakshminarayanan et al. (2017) |
| POD + MLP of reduced coefficients; compression error reported separately from coefficient error | P (method) / I (setup) | Hesthaven & Ubbiali (2018); the error separation follows the spirit of Dölz & Henríquez |
| Dissipative pretraining with an absorbing well −iW (W = 0.05), half the budget lossy and half lossless, same geometries | P (idea) / I (transfer to 1D) | Nam, Park & Jang (2026). They broaden electromagnetic resonances with material loss; the transfer to 1D and the value of W are ours. |
| Hybrid surrogate: learn the smooth single-barrier S-matrix, compose exactly with the star product | I | Our structure-exploiting model. It requires knowing the block decomposition. |

## 3. New empirical findings (F), with their scope

| finding | evidence | scope / caveat |
|---|---|---|
| Local methods on uniform grids miss resonances narrower than the grid spacing | Exp. 1–2 | Expected behaviour; quantified here |
| AAA on t(k) recovers every resonance from 15–30 uniform evaluations | Exp. 3 (38 systems), Exp. 5 (unseen families) | Piecewise-constant and smooth finite-range 1D potentials; exact or low-noise data |
| The amplitude is a better AAA target than T (about half as many points) | Exp. 1, 3 | Consistent with the pole count of \|t\|²; observed, not proven in general |
| Greedy look-ahead self-certification gave no false certification | Exp. 3 (38/38), Exp. 5 | Empirical; a theoretical guarantee would need error bounds we do not have |
| Detectability limit: Γ/h ≳ C·max(data error, background-approximation error) | Exp. 3b map | Heuristic; C is not universal |
| Segment sampling on t beats segment sampling on T for local interpolants | Exp. 2–3 | Convention-dependent; disappears for phase reference X = 50 |
| Stationary GPs degrade on multi-scale adaptive designs; MLPs miss narrow peaks even with data (spectral bias) | Exp. 2 | Small models, our protocols |
| Hybrid S-matrix composition is the most sample-efficient parametric surrogate | Exp. 4 | Requires a known decomposition; 1D |

## 4. Reference verification log

The network policy of the execution environment blocked arxiv.org, doi.org, publisher sites, Crossref, OpenAlex
and Semantic Scholar (all requests timed out). Only github.com was reachable. Verification was therefore limited
to what the authors state in their own repositories:

| reference | what was verified | how |
|---|---|---|
| Pradovera, Hiptmair, Perugia. "Surrogate modeling of resonant behavior in scattering problems through adaptive rational approximation and sketching", J. Sci. Comput. (2025) | title, authors, year, DOI 10.1007/s10915-025-03020-x, preprint arXiv:2503.10194 | README of github.com/pradovera/hybridScatteringAmplification |
| Giri, Kim, Drischler, Elster, Furnstahl. "Active learning emulators for nuclear two-body scattering in momentum space", Phys. Rev. C 113, 044001 (2026) | full citation, DOI 10.1103/s6my-pqs9, arXiv:2512.17842 | BibTeX in the README of github.com/buqeye/ms_greedy_emulator |
| Nam, Park, Jang. "Data-efficient electromagnetic surrogate solver through dissipative relaxation transfer learning", Adv. Opt. Mater. (2026), DOI 10.1002/adom.71366 | title and arXiv:2601.18235 only | The repository README says "currently under review in Advanced Optical Materials". **The DOI and the publication status could not be verified.** |
| Dölz, Henríquez. "Fully discrete analysis of the Galerkin POD neural network approximation with application to 3D acoustic wave scattering", arXiv:2502.01859 | nothing | No reachable source. **Unverified**; cited as supplied. |
| Textbook and classic references (AAA 2018, Fritsch–Carlson 1980, Tancik 2020, …) | not checked online | Cited from established bibliographic knowledge; check against the originals before submission |

**The full texts of the four recent papers were not read** in this environment. The methods were implemented from
the authors' public code and abstracts (summarised in the session log), and the reports do not attribute to these
papers any result they were not checked to contain.
