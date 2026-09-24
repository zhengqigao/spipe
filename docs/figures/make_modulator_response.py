"""Draw docs/figures/modulator_response.png: the MZM phase change after a step in the drive,
for several response times tau.

Every curve is computed by SPIPE's own modulator model (MZM._drive, the code path a simulation
uses) and checked against the analytic step response 1 - exp(-t/tau) before it is drawn. On the
time grid the drive value at a sample is taken to hold over the interval before it, so the
discrete response leads the continuous one by exactly one sample; the check allows for that.

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
T_STOP = 70e-12
STEP_AT = 5e-12                    # the drive steps from 0 to vpi here
VPI = 2.0


def response(tau):
    """Normalised phase Δφ(t) / Δφ_final after a step in the drive, from SPIPE's model."""
    t = torch.arange(0.0, T_STOP, DT, dtype=torch.float64)
    drive = torch.where(t >= STEP_AT, torch.tensor(VPI, dtype=torch.float64),
                        torch.tensor(0.0, dtype=torch.float64))
    model = MZM(time=t, omega=torch.tensor([2 * math.pi * 193.1e12], dtype=torch.float64),
                act=drive, vpi=VPI, tau=tau)
    dphi = math.pi * model._drive().detach() / VPI           # the filtered Δφ, in rad
    return (t - STEP_AT).numpy() * 1e12, (dphi / math.pi).numpy()


def analytic(t_ps, tau):
    if tau == 0:
        return [1.0 if v >= 0 else 0.0 for v in t_ps]
    return [0.0 if v <= 0 else 1 - math.exp(-v * 1e-12 / tau) for v in t_ps]


CURVES = [("τ = 0 (instantaneous)", 0.0, '0.35', (0, (4, 3))),
          ("τ = 4 ps  (40 GHz)", 4e-12, '#1b4f9c', '-'),
          ("τ = 8 ps  (20 GHz)", 8e-12, '#2a9d8f', '-'),
          ("τ = 16 ps  (10 GHz)", 16e-12, '#e9a03b', '-')]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
for label, tau, color, style in CURVES:
    t_ps, y = response(tau)
    shift = DT * 1e12 if tau > 0 else 0.0                       # the filter leads by one sample
    ref = analytic([v + shift for v in t_ps], tau)
    err = max(abs(a - b) for a, b in zip(y, ref))
    assert err < 1e-12, f"{label}: {err:.2e} from the analytic step response"
    ax.plot(t_ps, y, color=color, linestyle=style, linewidth=2, label=label)
ax.axhline(1, color='0.85', linewidth=0.8)
ax.axhline(0.63, color='0.85', linewidth=0.8)
ax.text(66, 0.645, '63 %', color='0.45', fontsize=9, ha='right')
ax.set_xlabel('time after the drive step (ps)')
ax.set_ylabel('Δφ / Δφ_final')
ax.set_xlim(-3, 66)
ax.set_ylim(-0.03, 1.05)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.set_title('MZM phase change after a step in the drive:  τ·dΔφ/dt + Δφ = π·(V − vbias)/vpi',
             fontsize=10.5)
fig.tight_layout()
out = os.path.join(HERE, 'modulator_response.png')
fig.savefig(out, dpi=150)
print(f"wrote {out}; every curve matches the analytic step response")
