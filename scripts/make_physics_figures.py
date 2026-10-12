"""Stage A / Stage B physics validation figures and numbers (no machine learning).

Outputs to results/physics/: PNG figures and physics_summary.json.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qscatter.config import STAGE_A, STAGE_B
from qscatter.physics.analytic import single_barrier_T
from qscatter.physics.ode_solver import solve_ode
from qscatter.physics.resonances import find_resonances, lorentzian
from qscatter.physics.smatrix import compute_scattering, compute_transmission
from qscatter.plotting import INK2, NEUTRAL, SLOTS, setup

OUT = Path("results/physics")
OUT.mkdir(parents=True, exist_ok=True)
setup()
summary = {}

# ---------------------------------------------------------------- Stage A
pA = STAGE_A.potential()
V0, a = pA.params["height"], pA.params["width"]
E = np.linspace(*STAGE_A.E_range, STAGE_A.n_ref)
rA = compute_scattering(E, pA)
T_an = single_barrier_T(E, V0, a)
summary["stage_A"] = dict(
    potential=pA.to_dict(), E_range=STAGE_A.E_range, n_ref=E.size,
    max_abs_err_vs_analytic=float(np.max(np.abs(rA.T - T_an))),
    max_unitarity_defect=float(np.max(rA.unitarity_defect)),
    T_at_barrier_top=float(compute_transmission(V0, pA)[0]),
    T_at_barrier_top_exact=1 / (1 + V0 * a**2 / 4),
    first_above_barrier_resonance=V0 + (np.pi / a) ** 2,
)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(E, rA.T, color=SLOTS[0], label="T(E) numerical")
ax[0].plot(E, rA.R, color=SLOTS[1], label="R(E) numerical")
idx = np.arange(0, E.size, 800)
ax[0].plot(E[idx], T_an[idx], "o", ms=5, mfc="none", mec=INK2, mew=1.2, label="T(E) analytic")
ax[0].axvline(V0, color=NEUTRAL, lw=1, ls="--")
ax[0].text(V0 + 0.15, 0.97, "E = V$_0$", color=INK2, fontsize=9, va="top")
ax[0].set(xlabel="E", ylabel="probability", title=f"Single barrier (V$_0$={V0:g}, a={a:g})")
ax[0].legend(loc="center right")
sub = slice(None, None, 20)  # every 20th point keeps the scatter readable
for y, c, lab in [(np.abs(rA.T - T_an), SLOTS[0], "|T - T$_{analytic}$|"), (rA.unitarity_defect, SLOTS[1], "|T + R - 1|")]:
    ys, xs = y[sub], E[sub]
    ax[1].semilogy(xs[ys > 0], ys[ys > 0], ".", ms=2.5, color=c, label=lab)
ax[1].axhline(np.finfo(float).eps, color=NEUTRAL, ls=":", lw=1)
ax[1].text(STAGE_A.E_range[0] + 0.1, np.finfo(float).eps * 1.6, "machine eps", color=INK2, fontsize=8)
ax[1].set(xlabel="E", ylabel="absolute error", title="Validation vs analytic formula (exact zeros not shown)", ylim=(1e-18, 1e-13))
ax[1].legend(loc="upper right", markerscale=4)
fig.savefig(OUT / "stageA_transmission.png")
plt.close(fig)

# ---------------------------------------------------------------- Stage B
pB = STAGE_B.potential()
EB = np.linspace(*STAGE_B.E_range, STAGE_B.n_ref)
rB = compute_scattering(EB, pB)
fT = lambda e: compute_transmission(e, pB)
res = find_resonances(fT, np.linspace(*STAGE_B.E_range, 200001))
summary["stage_B"] = dict(
    potential=pB.to_dict(), E_range=STAGE_B.E_range, n_ref=EB.size,
    ref_grid_spacing=float(EB[1] - EB[0]),
    max_unitarity_defect=float(np.max(rB.unitarity_defect)),
    resonances=[r.to_dict() for r in res],
    points_per_fwhm_on_reference_grid=[float(r.fwhm / (EB[1] - EB[0])) for r in res],
)
# ODE confirmation of each resonance + Lorentzian quality
ode_checks = []
for r in res:
    o = solve_ode(r.E_res, pB)
    win = np.linspace(r.E_res - 0.5 * r.fwhm, r.E_res + 0.5 * r.fwhm, 201)
    lor_err = float(np.max(np.abs(fT(win) - lorentzian(win, r.E_res, r.fwhm, r.T_peak))))
    ode_checks.append(dict(E_res=r.E_res, T_ode=float(abs(o.t) ** 2),
                           T_smatrix=r.T_peak, max_dev_from_lorentzian_within_fwhm=lor_err))
summary["stage_B"]["ode_and_lorentzian_checks"] = ode_checks

# reference-grid convergence: resonances found on coarser grids
conv = []
for n in (2001, 5001, 20001, 99001, 400001):
    try:
        rr = find_resonances(fT, np.linspace(*STAGE_B.E_range, n))
    except Exception:
        rr = []
    conv.append(dict(n_grid=n, n_found=len(rr), E_res=[x.E_res for x in rr], fwhm=[x.fwhm for x in rr]))
summary["stage_B"]["grid_convergence"] = conv
# sampled-T convergence of a uniform grid: does the reference grid resolve the narrowest peak?
for n in (25001, 99001, 396001):
    Eg = np.linspace(*STAGE_B.E_range, n)
    summary["stage_B"].setdefault("max_T_on_grid", {})[n] = float(np.max(fT(Eg)[np.abs(Eg - res[0].E_res) < res[0].fwhm]))

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(EB, rB.T, color=SLOTS[0], lw=1.5)
for i, r in enumerate(res):
    ax[0].annotate(f"E$_{i+1}$={r.E_res:.4f}\nΓ={r.fwhm:.2g}", (r.E_res, r.T_peak), xytext=(6, -28),
                   textcoords="offset points", fontsize=8, color=INK2)
ax[0].set(ylabel="T(E)", title="Symmetric double barrier: resonant tunnelling "
          f"(V={pB.params['height_left']:g}, b={pB.params['width_left']:g}, w={pB.params['well_width']:g})")
ax[1].semilogy(EB, rB.T, color=SLOTS[0], lw=1.2)
ax[1].set(xlabel="E", ylabel="T(E) (log)")
fig.savefig(OUT / "stageB_transmission.png")
plt.close(fig)

# narrow vs broad resonance, with Breit-Wigner overlay and ODE points
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for i, r in enumerate(res):
    w = np.linspace(r.E_res - 4 * r.fwhm, r.E_res + 4 * r.fwhm, 2001)
    ax[i].plot((w - r.E_res) / r.fwhm, fT(w), color=SLOTS[0], label="S-matrix")
    ax[i].plot((w - r.E_res) / r.fwhm, lorentzian(w, r.E_res, r.fwhm, r.T_peak), color=SLOTS[1], lw=1.2,
               ls="--", label="Breit–Wigner")
    pts = np.linspace(r.E_res - 3 * r.fwhm, r.E_res + 3 * r.fwhm, 9)
    ax[i].plot((pts - r.E_res) / r.fwhm, [abs(solve_ode(p, pB).t) ** 2 for p in pts], "o", mfc="none",
               mec=INK2, mew=1.2, label="ODE (independent)")
    ax[i].set(xlabel="(E - E$_n$) / Γ$_n$", title=f"n={i+1}: E={r.E_res:.4f}, Γ={r.fwhm:.2e}")
ax[0].set_ylabel("T(E)")
ax[0].legend(loc="upper left", fontsize=8)
fig.savefig(OUT / "stageB_resonance_zoom.png")
plt.close(fig)

# complex amplitude: phase winds by ~pi across each resonance
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(EB, rB.t.real, color=SLOTS[0], lw=1.2, label="Re t")
ax[0].plot(EB, rB.t.imag, color=SLOTS[1], lw=1.2, label="Im t")
ax[0].set(xlabel="E", ylabel="t(E)", title="Complex transmission amplitude")
ax[0].legend()
ax[1].plot(EB, np.unwrap(np.angle(rB.t)) / np.pi, color=SLOTS[2], lw=1.5)
for r in res:
    ax[1].axvline(r.E_res, color=NEUTRAL, lw=0.8, ls=":")
ax[1].set(xlabel="E", ylabel="arg t / π (unwrapped)", title="Phase of t: rises by ≈π across each resonance")
fig.savefig(OUT / "stageB_amplitude_phase.png")
plt.close(fig)

# wavefunction density at and away from resonance
x = np.linspace(pB.x_left - 3, pB.x_right + 3, 3001)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for j, (Ev, lab) in enumerate([(res[0].E_res, "on resonance E$_1$"), (0.5 * (res[0].E_res + res[1].E_res), "off resonance")]):
    o = solve_ode(Ev, pB, x_grid=x)
    ax[j].plot(x, np.abs(o.psi) ** 2, color=SLOTS[0], label="|ψ(x)|²")
    ax2 = ax[j].twinx()  # potential drawn as context, its scale labelled separately
    ax2.fill_between(x, pB(x), color=NEUTRAL, alpha=0.25, lw=0)
    ax2.set_ylabel("V(x)", color=INK2)
    ax2.grid(False)
    ax[j].set(xlabel="x", title=f"{lab}: E={Ev:.4f}, T={abs(o.t)**2:.3g}")
ax[0].set_ylabel("|ψ|² (incident amplitude 1)")
fig.savefig(OUT / "stageB_wavefunction.png")
plt.close(fig)

# cost of the solvers (for the efficiency discussion)
Et = np.linspace(*STAGE_B.E_range, 10000)
t0 = time.perf_counter(); compute_scattering(Et, pB); ts = (time.perf_counter() - t0) / Et.size
t0 = time.perf_counter()
for e in Et[:200]:
    solve_ode(e, pB)
to = (time.perf_counter() - t0) / 200
summary["solver_cost_per_energy_s"] = dict(smatrix_vectorised=ts, ode_dop853=to)

(OUT / "physics_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=2, default=float))
