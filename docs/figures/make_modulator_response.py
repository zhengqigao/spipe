"""Draw docs/figures/modulator_response.png: the mzm phase response to a drive step, for several
filter orders n and time constants tau.

Every curve is computed by SPIPE's own modulator model (MZM._drive, the code path a simulation
uses), and checked against the analytic step response of n cascaded identical first-order stages,
1 - exp(-t/tau) * sum_{k<n} (t/tau)^k / k!, before it is drawn. On the time grid the drive
value at a sample is taken to hold over the interval before it, so the discrete response leads
the continuous one by exactly one sample: with that allowed for, n = 1 is exact (to 1e-15), and
n >= 2 agrees to about 0.2-0.35 * dt / tau -- the scheme is first-order accurate in dt/tau there, so
the samples must be much finer than tau (here dt = 0.05 ps).

Run:  python docs/figures/make_modulator_response.py
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from spipe.photonic.model.mzm import MZM

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.05e-12                      # 0.05 ps: fine enough that the curves are the continuous ones
T_STOP = 60e-12
STEP_AT = 5e-12                    # the drive steps from 0 to vpi here
VPI = 2.0


def response(tau, order):
    """Normalised phase Δφ(t) / Δφ_final after a step in the drive, from SPIPE's model."""
    t = torch.arange(0.0, T_STOP, DT, dtype=torch.float64)
    drive = torch.where(t >= STEP_AT, torch.tensor(VPI, dtype=torch.float64),
                        torch.tensor(0.0, dtype=torch.float64))
    model = MZM(time=t, omega=torch.tensor([2 * math.pi * 193.1e12], dtype=torch.float64),
                act=drive, vpi=VPI, tau=tau, order=order)
    dphi = math.pi * model._drive().detach() / VPI           # the filtered Δφ, in rad
    return (t - STEP_AT).numpy() * 1e12, (dphi / math.pi).numpy()


def analytic(t_ps, tau, order):
    x = [max(v, 0.0) * 1e-12 / tau for v in t_ps]
    return [0.0 if v <= 0 else 1 - math.exp(-v) * sum(v ** k / math.factorial(k)
                                                        for k in range(order)) for v in x]


def bandwidth_tau(f3db, order):
    """tau giving a 3 dB bandwidth f3db with n cascaded identical first-order stages."""
    return math.sqrt(2 ** (1 / order) - 1) / (2 * math.pi * f3db)


# (label, tau, order) per panel
TAU = 8e-12
PANELS = [
    ("(a) same τ = 8 ps, order n = 1…4",
     [(f"n = {n}", TAU, n) for n in (1, 2, 3, 4)]),
    ("(b) order n = 1, τ = 4, 8, 16 ps",
     [(f"τ = {int(tau * 1e12)} ps", tau, 1) for tau in (4e-12, 8e-12, 16e-12)]),
    ("(c) same bandwidth, 20 GHz: n = 1…4",
     [(f"n = {n} (τ = {bandwidth_tau(20e9, n) * 1e12:.2f} ps)", bandwidth_tau(20e9, n), n)
      for n in (1, 2, 3, 4)]),
]
COLORS = ['#1b4f9c', '#2a9d8f', '#e9a03b', '#c8553d']        # distinct in colour and lightness

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
for ax, (title, curves) in zip(axes, PANELS):
    for (label, tau, order), color in zip(curves, COLORS):
        t_ps, y = response(tau, order)
        ref = analytic([v + DT * 1e12 for v in t_ps], tau, order)      # one sample ahead
        err = max(abs(a - b) for a, b in zip(y, ref))
        bound = 1e-12 if order == 1 else 0.5 * DT / tau
        assert err < bound, f"{label}: {err:.2e} from the analytic response (bound {bound:.1e})"
        ax.plot(t_ps, y, color=color, linewidth=2, label=label)
    ax.axvline(0, color='0.6', linewidth=0.8, linestyle=':')
    ax.axhline(1, color='0.85', linewidth=0.8)
    ax.axhline(0.5, color='0.85', linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('time after the drive step (ps)')
    ax.set_xlim(-3, 45)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
axes[0].set_ylabel('Δφ / Δφ_final')
fig.suptitle('mzm phase response to a step in the drive:  (τ·d/dt + 1)ⁿ Δφ = π·(V − vbias)/vpi',
             fontsize=11.5)
fig.tight_layout()
out = os.path.join(HERE, 'modulator_response.png')
fig.savefig(out, dpi=150)
print(f"wrote {out}; every curve matches the analytic step response of n cascaded first-order stages")
