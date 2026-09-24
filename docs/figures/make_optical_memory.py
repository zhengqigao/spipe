"""Draw docs/figures/optical_memory.png: what the steady-state photonic solve misses, and what
mode='envelope' recovers.

A modulator switches the light on at 210 ps and a detector reads the output of
(a) a 100 ps delay line, and (b) a ring resonator with a 10 ps round trip.

Both runs use SPIPE itself (the circuits of examples/envelope_delay_ring.py), reading the field at
the centre carrier. The exact answers are drawn next to them, and every envelope sample is
checked against them before anything is drawn:
- delay line: the input, 100 ps later;
- ring: the round-trip recursion. Each trip adds the light coupled in to what survived the
  previous trip, so after m trips the through field is  t - kappa^2 * a * sum_{k<m} (a t)^k.

Run:  python docs/figures/make_optical_memory.py
"""
import math
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from spipe import Photonic

HERE = os.path.dirname(os.path.abspath(__file__))
C = 299792458.0
DT, N = 10e-12, 120                                    # 10 ps samples, 0 .. 1190 ps
t = torch.arange(N, dtype=torch.float64) * DT
drive = (t <= 200e-12).to(torch.float64).reshape(-1, 1) * 2.0    # light switches on at 210 ps
ON = 21                                                # first sample with the light on


def run(lines, node, k):
    ph = Photonic([l + "\n" for l in lines])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # edge-carrier notice; we read the centre
        _, qs, _ = ph.simulate(t, drive)
        _, env, _ = ph.simulate(t, drive, mode='envelope')
    power = lambda p: (p[node][:, k, 1].abs() ** 2).numpy()
    return power(qs), power(env)


# (a) 100 ps delay line
DELAY = 100e-12
qs_a, env_a = run([".mode neff=4.0 ng=4.0 wl=1552.5e-9", ".freq 192.9e12 193.3e12 401",
                   ".source 1.0@a1 0.0@a2", "mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9",
                   f"wg0 b2 c1 l={DELAY * C / 4.0!r}", "pd1 c1 v1 level1 r0=1.0",
                   "pd2 b1 v2 level1 r0=1.0", ".prob c1"], 'c1', 200)
exact_a = (np.arange(N) >= ON + round(DELAY / DT)).astype(float)

# (b) ring resonator: round trip T_RT, through coupling T_C, field kept per trip A
T_RT, T_C, A = 10e-12, 0.95, 0.99
qs_b, env_b = run([".mode neff=4.0 ng=4.0 wl=1552.5e-9", ".freq 192.9e12 193.3e12 1601",
                   ".source 1.0@a1 0.0@a2", "mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9",
                   f"mzi0 b2 r2 c1 r1 theta={math.acos(T_C)!r}",
                   f"wg0 r1 r2 l={T_RT * C / 4.0!r} alpha={A!r}", "pd1 c1 v1 level1 r0=1.0",
                   "pd2 b1 v2 level1 r0=1.0", ".prob c1"], 'c1', 800)
kappa2 = 1 - T_C ** 2
exact_b = np.array([0.0 if n < ON else
                    (T_C - kappa2 * A * sum((A * T_C) ** j for j in range(n - ON))) ** 2
                    for n in range(N)])
lifetime = -T_RT / math.log(A * T_C)

for name, env, exact, tol in (("delay line", env_a, exact_a, 2e-3), ("ring", env_b, exact_b, 2e-3)):
    err = float(np.abs(env - exact).max())
    assert err < tol, f"{name}: envelope differs from the exact answer by {err:.2e}"

ps = t.numpy() * 1e12
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
panels = [(axes[0], qs_a, env_a, exact_a, "(a) 100 ps delay line",
           "light reaches the detector\n100 ps after it is switched on"),
          (axes[1], qs_b, env_b, exact_b,
           f"(b) ring resonator (10 ps round trip, photon lifetime {lifetime * 1e12:.0f} ps)",
           "light passing the ring arrives first;\nthen the ring fills and cancels part\nof it, over several photon lifetimes")]
for ax, qs, env, exact, title, note in panels:
    ax.axvline(ps[ON], color='0.8', linewidth=1)
    ax.plot(ps, qs, color='#d1495b', linewidth=2, linestyle=(0, (5, 3)),
            label="steady state at every sample (the original assumption)")
    ax.plot(ps, env, color='#1b4f9c', linestyle='none', marker='o', markersize=4,
            label="mode='envelope' (samples)")
    ax.plot(ps, exact, color='0.15', linewidth=1, label='exact')
    ax.text(ps[ON] + 6, 1.04, 'light switched on', color='0.45', fontsize=8.5, va='bottom')
    ax.text(1180, 0.62, note, color='0.35', fontsize=8.5, ha='right')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('time (ps)')
    ax.set_xlim(100, 1190)
    ax.set_ylim(-0.03, 1.13)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel('power at the detector (per unit launched)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(HERE, 'optical_memory.png')
fig.savefig(out, dpi=150)
print(f"wrote {out}; envelope matches the exact delay-line and ring answers")
