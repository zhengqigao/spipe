"""Draw docs/figures/coherent_detection.png: two optical carriers 10 GHz apart on one detector,
summed incoherently (the default) and coherently (coherent=1).

Both carriers have unit amplitude and reach the detector in phase at t = 0. Summed in field,
they beat at their spacing Df:  |e^{+jπ·Df·t} + e^{−jπ·Df·t}|² = 2 + 2·cos(2π·Df·t).
A detector with a single-pole bandwidth bw passes that beat with gain |H| = 1/√(1 + (Df/bw)²).
The exact filtered current, starting from the sample value 4 at t = 0, is

    y(t) = 2 + 2·Re[H·e^{j2π·Df·t}] + (2 − 2·Re H)·e^{−t/τ},     H = 1/(1 + j·Df/bw),  τ = 1/(2π·bw)

Every SPIPE curve is checked against it before anything is drawn.

Run:  python docs/figures/make_coherent_detection.py
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
F0, DF = 193.1e12, 10e9                                # two carriers, 10 GHz apart


def detect(bw, coherent, dt, n):
    """Photocurrent of a unit-responsivity detector on two unit carriers, shape (n,)."""
    deck = [".mode neff=2.35 ng=4.0 wl=1550e-9",
            f".freq {F0!r} {F0 + DF!r} 2",
            ".source 1.0@a1",
            "wg0 a1 b1 l=0",
            f"pd1 b1 v1 level1 r0=1.0 bw={bw!r} noise=0 coherent={coherent}"]
    t = torch.arange(n, dtype=torch.float64) * dt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current = Photonic([l + "\n" for l in deck]).simulate(t)[0][:, 0]
    return t.numpy(), current.numpy()


def exact(t, bw):
    h = 1.0 / (1.0 + 1j * DF / bw)
    tau = 1.0 / (2 * math.pi * bw)
    return 2 + 2 * np.real(h * np.exp(2j * math.pi * DF * t)) + (2 - 2 * h.real) * np.exp(-t / tau)


def beat_left(bw, dt):
    """Measured beat amplitude after the filter has settled, as a fraction of the mean (2)."""
    tau = 1.0 / (2 * math.pi * bw)
    n = int((8 * tau + 3 / DF) / dt)
    t, y = detect(bw, 1, dt, n)
    tail = y[t > 8 * tau]
    return 0.5 * (tail.max() - tail.min()) / 2.0


panels = []
for bw, dt, n in ((50e9, 0.1e-12, 3000), (0.5e9, 1e-12, 2000)):
    t, coh = detect(bw, 1, dt, n)
    _, inc = detect(bw, 0, dt, n)
    ref = exact(t, bw)
    err = float(np.abs(coh - ref).max())
    assert err < 0.02, f"bw = {bw:g}: coherent current differs from the exact answer by {err:.3g}"
    assert float(np.abs(inc - 2).max()) < 1e-12
    panels.append((bw, t, coh, inc, ref))

ratios = np.logspace(-1, 3, 200)
points = [0.2, 1, 5, 20, 100, 500]
measured = [beat_left(DF / r, 1 / (DF * 40)) for r in points]
for r, m in zip(points, measured):
    assert abs(m - 1 / math.sqrt(1 + r * r)) < 0.02 * max(1 / math.sqrt(1 + r * r), 0.05), (r, m)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={'width_ratios': [1, 1, 0.9]})
titles = ["(a) fast detector, bw = 50 GHz: the beat is the signal",
          "(b) slow detector, bw = 0.5 GHz: the beat is filtered out"]
windows = [300, 2000]
for ax, (bw, t, coh, inc, ref), title, win in zip(axes, panels, titles, windows):
    ps = t * 1e12
    ax.plot(ps, inc, color='#d1495b', linewidth=2, linestyle=(0, (5, 3)),
            label='coherent=0: incoherent sum (the default, and the original model)')
    ax.plot(ps, coh, color='#8fb3e0', linewidth=4, label='coherent=1: carriers summed in field')
    ax.plot(ps, ref, color='0.1', linewidth=0.9, label='exact')
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel('time (ps)')
    ax.set_xlim(0, win)
    ax.set_ylim(-0.1, 4.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel('photocurrent (A, r0 = 1, two unit carriers)')
axes[1].annotate('starts in phase at t = 0,\nthen settles with τ = 1/(2π·bw)',
                 xy=(120, 3.3), xytext=(500, 3.5), fontsize=8.5, color='0.35',
                 arrowprops=dict(arrowstyle='->', color='0.5', linewidth=0.8))

ax = axes[2]
ax.loglog(ratios, 1 / np.sqrt(1 + ratios ** 2), color='0.15', linewidth=1, label='1/√(1 + (Δf/bw)²)')
ax.loglog(points, measured, 'o', color='#1b4f9c', markersize=6, label='SPIPE, coherent=1')
ax.axvline(1, color='0.85', linewidth=0.8)
ax.text(1.15, 1.15, 'Δf = bw', color='0.45', fontsize=8.5)
ax.set_title('(c) beat left after the detector', fontsize=9.5)
ax.set_xlabel('carrier spacing / detector bandwidth  (Δf / bw)')
ax.set_ylabel('beat amplitude / mean current')
ax.set_ylim(1e-3, 1.5)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc='lower left')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(0.36, 0))
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(HERE, 'coherent_detection.png')
fig.savefig(out, dpi=150)
print(f"wrote {out}; every curve matches the exact answer")
