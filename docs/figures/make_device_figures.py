"""Draw the figures of docs/photonic_device.md: a schematic of every photonic device, and, where
a device has a tunable response, that response as SPIPE computes it.

Every response curve is simulated with SPIPE's Photonic solver and checked against the formula
the document gives for it, so the figures and the equations cannot drift apart.

Run:  python docs/figures/make_device_figures.py
"""
import math
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import torch

from spipe import Photonic

warnings.simplefilter('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))

WG = '#1b4f9c'          # waveguides
ACT = '#c8553d'         # electrodes / electrical nodes
BOX = '#e9eef6'         # device bodies
TXT = '#222222'
C1, C2 = '#1b4f9c', '#e9a03b'


# ------------------------------------------------------------------ drawing helpers

def canvas(ax, w=10, h=4):
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.axis('off')


def wire(ax, x0, y0, x1, y1, color=WG, lw=2.5):
    ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, solid_capstyle='round')


def s_bend(ax, x0, y0, x1, y1, color=WG, lw=2.5):
    t = torch.linspace(0, 1, 50)
    ax.plot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * (1 - torch.cos(math.pi * t)) / 2,
            color=color, linewidth=lw)


def port(ax, x, y, name, side='left', color=TXT):
    ax.plot([x], [y], 'o', color=color, markersize=5)
    dx = -0.15 if side == 'left' else 0.15
    ax.text(x + dx, y, name, ha='right' if side == 'left' else 'left', va='center',
            fontsize=11, color=color, family='monospace')


def body(ax, x, y, w, h, label=None, color=BOX):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', facecolor=color,
                                edgecolor='#8a9bb5', linewidth=1))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10, color=TXT)


def electrode(ax, x, y, w, node='V', h=0.25):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=ACT, alpha=0.85, edgecolor='none'))
    ax.plot([x + w / 2, x + w / 2], [y + h, y + h + 0.45], color=ACT, linewidth=1.5)
    ax.plot([x + w / 2], [y + h + 0.45], 'o', color=ACT, markersize=5)
    ax.text(x + w / 2 + 0.12, y + h + 0.5, node, color=ACT, fontsize=11, va='bottom',
            family='monospace')


def coupler(ax, x, ytop, ybot, length=1.0, gap=0.25):
    """Two waveguides bending together, running side by side, bending apart."""
    mid = (ytop + ybot) / 2
    s_bend(ax, x, ytop, x + 0.6, mid + gap / 2)
    s_bend(ax, x, ybot, x + 0.6, mid - gap / 2)
    wire(ax, x + 0.6, mid + gap / 2, x + 0.6 + length, mid + gap / 2)
    wire(ax, x + 0.6, mid - gap / 2, x + 0.6 + length, mid - gap / 2)
    s_bend(ax, x + 0.6 + length, mid + gap / 2, x + 1.2 + length, ytop)
    s_bend(ax, x + 0.6 + length, mid - gap / 2, x + 1.2 + length, ybot)
    return x + 1.2 + length


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ simulation helpers

HEAD = ['.mode neff=2.35 ng=4.0 wl=1550e-9', '.freq 193.1e12 193.1e12 1']


def powers(lines, t=None, drive=None, head=HEAD):
    ph = Photonic([x + '\n' for x in head + lines])
    if t is None:
        return ph.simulate()[0][0]
    return ph.simulate(t, drive)[0]


def check(name, got, want, tol=1e-9):
    err = max(abs(float(a) - float(b)) for a, b in zip(got, want))
    assert err < tol, f"{name}: {err:.2e} from the documented formula"


def response_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.28) if ylabel.startswith('power') else None
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ 1. waveguide
fig, ax = plt.subplots(figsize=(7, 1.8))
canvas(ax, 10, 2)
port(ax, 1, 1, 'a1'); port(ax, 9, 1, 'b1', 'right')
wire(ax, 1, 1, 9, 1)
ax.annotate('', xy=(8.6, 1.45), xytext=(1.4, 1.45),
            arrowprops=dict(arrowstyle='<->', color='0.4'))
ax.text(5, 1.55, 'length l,  field transmission alpha', ha='center', va='bottom', fontsize=10)
ax.text(5, 0.45, 'out = alpha · exp(j·β·l) · in', ha='center', fontsize=10, color='0.3',
        family='monospace')
save(fig, 'device_wg.png')

# ------------------------------------------------------------------ 2. phase shifter
fig, ax = plt.subplots(figsize=(7, 1.8))
canvas(ax, 10, 2)
port(ax, 1, 1, 'a1'); port(ax, 9, 1, 'b1', 'right')
wire(ax, 1, 1, 9, 1)
body(ax, 4, 0.7, 2, 0.6, 'ps')
ax.text(5, 0.25, 'out = exp(j·ps) · in', ha='center', fontsize=10, color='0.3',
        family='monospace')
save(fig, 'device_ps.png')

# ------------------------------------------------------------------ 3. mzi (a coupler) + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 4)
port(ax, 1.5, 3, 'a1'); port(ax, 1.5, 1, 'a2')
wire(ax, 1.5, 3, 3, 3); wire(ax, 1.5, 1, 3, 1)
xe = coupler(ax, 3, 3, 1, length=2.0)
wire(ax, xe, 3, 8.5, 3); wire(ax, xe, 1, 8.5, 1)
port(ax, 8.5, 3, 'b1', 'right'); port(ax, 8.5, 1, 'b2', 'right')
ax.text(5.2, 2.55, 'θ', ha='center', va='center', fontsize=13, color=TXT)
ax.text(5.2, 0.2, 'through: cos θ     cross: j·sin θ', ha='center', fontsize=10, color='0.3',
        family='monospace')
thetas = torch.linspace(0, math.pi / 2, 41)
through, cross = [], []
for th in thetas:
    p = powers(['.source 1.0@a1', f'mzi0 a1 a2 b1 b2 theta={float(th)!r}',
                'pd1 b1 v1 level1 r0=1', 'pd2 b2 v2 level1 r0=1'])
    through.append(float(p[0])); cross.append(float(p[1]))
check('mzi', through, [math.cos(float(th)) ** 2 for th in thetas])
bx.plot(thetas / math.pi, through, color=C1, linewidth=2, label='through  b1 = cos²θ')
bx.plot(thetas / math.pi, cross, color=C2, linewidth=2, label='cross  b2 = sin²θ')
response_axes(bx, 'theta (units of π)', 'power (light into a1)')
bx.legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
save(fig, 'device_mzi.png')

# ------------------------------------------------------------------ 4. pbum + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 4)
port(ax, 0.8, 3, 'a1'); port(ax, 0.8, 1, 'a2')
x = coupler(ax, 0.8, 3, 1, length=0.6)
body(ax, x + 0.4, 2.7, 1.5, 0.6, 'θ'); body(ax, x + 0.4, 0.7, 1.5, 0.6, 'φ')
wire(ax, x, 3, x + 0.4, 3); wire(ax, x, 1, x + 0.4, 1)
wire(ax, x + 1.9, 3, x + 2.3, 3); wire(ax, x + 1.9, 1, x + 2.3, 1)
xe = coupler(ax, x + 2.3, 3, 1, length=0.6)
port(ax, xe, 3, 'b1', 'right'); port(ax, xe, 1, 'b2', 'right')
ax.text(1.6, 0.25, 'cp_left', ha='center', fontsize=9, color='0.4')
ax.text(xe - 1.0, 0.25, 'cp_right', ha='center', fontsize=9, color='0.4')
diffs = torch.linspace(0, 2 * math.pi, 61)
cross, through = [], []
for d in diffs:
    p = powers(['.source 1.0@a1', f'pbum0 a1 a2 b1 b2 theta={float(d)!r} phi=0 l=0',
                'pd1 b1 v1 level1 r0=1', 'pd2 b2 v2 level1 r0=1'])
    through.append(float(p[0])); cross.append(float(p[1]))
check('pbum', through, [math.sin(float(d) / 2) ** 2 for d in diffs])
bx.plot(diffs / math.pi, through, color=C1, linewidth=2, label='b1 = sin²((θ−φ)/2)')
bx.plot(diffs / math.pi, cross, color=C2, linewidth=2, label='b2 = cos²((θ−φ)/2)')
response_axes(bx, 'theta − phi (units of π)', 'power (light into a1)')
bx.legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
save(fig, 'device_pbum.png')

# ------------------------------------------------------------------ 5. splitter and wdm
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 2.8))
for axis, title, labels in ((ax, 'splitter1to4', ['1/4 power'] * 4),
                            (bx, 'wdm1to4', ['channel f1', 'channel f2', 'channel f3', 'channel f4'])):
    canvas(axis, 10, 5)
    port(axis, 1, 2.5, 'a1')
    wire(axis, 1, 2.5, 2.9, 2.5)
    body(axis, 2.9, 1.9, 2.2, 1.2, title)
    for i, y in enumerate((4.3, 3.1, 1.9, 0.7)):
        s_bend(axis, 5.1, 2.5, 7.3, y)
        port(axis, 7.3, y, f'b{i + 1}', 'right')
        axis.text(8.1, y, labels[i], va='center', fontsize=9, color='0.35')
save(fig, 'device_splitter_wdm.png')

# ------------------------------------------------------------------ 6. mzm + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 4.6)
port(ax, 0.6, 3, 'a1'); port(ax, 0.6, 1, 'a2')
x = coupler(ax, 0.6, 3, 1, length=0.5)
wire(ax, x, 3, x + 3.2, 3); wire(ax, x, 1, x + 3.2, 1)
electrode(ax, x + 0.5, 3.18, 2.2, node='V')
ax.add_patch(Rectangle((x + 0.5, 0.57), 2.2, 0.25, facecolor=ACT, alpha=0.45, edgecolor='none'))
ax.text(x + 1.6, 2.35, 'arm 1: φ1, a1', ha='center', fontsize=9, color='0.35')
ax.text(x + 1.6, 1.45, 'arm 2: φ2, a2', ha='center', fontsize=9, color='0.35')
xe = coupler(ax, x + 3.2, 3, 1, length=0.5)
port(ax, xe, 3, 'b1', 'right'); port(ax, xe, 1, 'b2', 'right')
ax.text(1.5, 0.3, 'κ1', ha='center', fontsize=10); ax.text(xe - 1.0, 0.3, 'κ2', ha='center', fontsize=10)
volts = torch.linspace(-1, 3, 81, dtype=torch.float64)
p = powers(['.source 1.0@a1 0.0@a2', 'mzm0 a1 a2 b1 b2 v level1 vpi=2.0 vbias=0.0 act_l=1e-9',
            'pd1 b1 v1 level1 r0=1', 'pd2 b2 v2 level1 r0=1'],
           t=torch.arange(len(volts), dtype=torch.float64) * 1e-9, drive=volts.reshape(-1, 1))
check('mzm', p[:, 0], [math.sin(math.pi * float(v) / 2.0 / 2) ** 2 for v in volts])
bx.plot(volts, p[:, 0], color=C1, linewidth=2, label='b1 = sin²(Δφ/2)')
bx.plot(volts, p[:, 1], color=C2, linewidth=2, label='b2 = cos²(Δφ/2)')
bx.axvline(0, color='0.8', linewidth=1); bx.axvline(2, color='0.8', linewidth=1)
bx.text(0.05, 0.03, 'vbias', fontsize=8, color='0.4'); bx.text(2.05, 0.03, 'vbias + vpi', fontsize=8, color='0.4')
response_axes(bx, 'drive V (volts), vpi = 2, vbias = 0', 'power (light into a1)')
bx.legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
save(fig, 'device_mzm.png')

# ------------------------------------------------------------------ 7. modm + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 4.6)
port(ax, 1.2, 3, 'a1'); port(ax, 1.2, 1, 'a2')
wire(ax, 1.2, 3, 2.4, 3); wire(ax, 1.2, 1, 2.4, 1)
ax.text(1.9, 3.3, 'wgu_l', ha='center', fontsize=8, color='0.4')
ax.text(1.9, 0.55, 'wgl_l', ha='center', fontsize=8, color='0.4')
xe = coupler(ax, 2.4, 3, 1, length=2.4)
electrode(ax, 3.3, 2.2, 1.8, node='V')
wire(ax, xe, 3, 8.6, 3); wire(ax, xe, 1, 8.6, 1)
port(ax, 8.6, 3, 'b1', 'right'); port(ax, 8.6, 1, 'b2', 'right')
ax.text(4.2, 0.25, 'variable coupler: angle φ(V)', ha='center', fontsize=9, color='0.35')
rate = 2 * math.pi * 193.1e12 / 299792458.0 * 2.35 * 200e-6 * 1e-3
volts = torch.linspace(0, 1.7, 69, dtype=torch.float64)
p = powers(['.source 1.0@a1 0.0@a2', 'modm0 a1 a2 b1 b2 v level1 coeff1=1e-3 act_l=200e-6',
            'pd1 b1 v1 level1 r0=1', 'pd2 b2 v2 level1 r0=1'],
           t=torch.arange(len(volts), dtype=torch.float64) * 1e-9, drive=volts.reshape(-1, 1),
           head=['.mode neff=2.35', '.freq 193.1e12 193.1e12 1'])
check('modm', p[:, 0], [math.cos(rate * float(v)) ** 2 for v in volts])
bx.plot(volts, p[:, 0], color=C1, linewidth=2, label='through b1 = cos²φ')
bx.plot(volts, p[:, 1], color=C2, linewidth=2, label='cross b2 = sin²φ')
response_axes(bx, 'drive V (volts); coeff1 = 1e-3, act_l = 200 µm', 'power (light into a1)')
bx.legend(frameon=False, fontsize=9, loc='upper center', ncol=2)
save(fig, 'device_modm.png')

# ------------------------------------------------------------------ 8. modp + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 2.8), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 3)
port(ax, 1, 1.2, 'a1'); port(ax, 9, 1.2, 'b1', 'right')
wire(ax, 1, 1.2, 9, 1.2)
electrode(ax, 3.5, 1.33, 2.5, node='V')
ax.text(4.75, 0.55, 'act_l (active)', ha='center', fontsize=9, color='0.35')
ax.text(7.5, 0.55, 'wg_l (passive)', ha='center', fontsize=9, color='0.35')
volts = torch.linspace(0, 2, 41, dtype=torch.float64)
ph = Photonic([x + '\n' for x in ['.mode neff=2.35', '.freq 193.1e12 193.1e12 1', '.source 1.0@a1',
                                  'modp0 a1 b1 v level1 coeff1=1e-3 act_l=200e-6',
                                  'pd1 b1 v1 level1 r0=1', '.prob b1']])
out = ph.simulate(torch.arange(len(volts), dtype=torch.float64) * 1e-9, volts.reshape(-1, 1))
phase = torch.angle(out[1]['b1'][:, 0, 1]).numpy()
unwrapped = [float(x) for x in torch.tensor(phase).numpy()]
import numpy as np
unwrapped = np.unwrap(unwrapped)
check('modp', unwrapped, [rate * float(v) for v in volts], tol=1e-9)
bx.plot(volts, unwrapped / math.pi, color=C1, linewidth=2, label='phase of b1 = β·act_l·coeff1·V')
response_axes(bx, 'drive V (volts); coeff1 = 1e-3, act_l = 200 µm', 'added phase (units of π)')
bx.legend(frameon=False, fontsize=9)
save(fig, 'device_modp.png')

# ------------------------------------------------------------------ 9. photodetector + response
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 2.8), gridspec_kw={'width_ratios': [1.25, 1]})
canvas(ax, 10, 3)
port(ax, 1, 1.5, 'b1')
wire(ax, 1, 1.5, 5, 1.5)
body(ax, 5, 0.9, 1.6, 1.2, 'pd')
wire(ax, 6.6, 1.5, 8.2, 1.5, color=ACT, lw=1.5)
port(ax, 8.2, 1.5, 'vo1', 'right', color=ACT)
ax.text(3.0, 2.0, 'optical', ha='center', fontsize=9, color='0.35')
ax.text(7.4, 1.8, 'current', ha='center', fontsize=9, color=ACT)
ax.text(5.8, 0.35, 'I = r0 · |E|²  (+ idark, bandwidth, noise)', ha='center', fontsize=9,
        color='0.3', family='monospace')
T = 400
t = torch.arange(T, dtype=torch.float64) * 2e-12
nl = ['.source 1.0@a1 0.0@a2', 'mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9',
      'pd1 b1 v1 level1 r0=0.8', 'pd2 b2 v2 level1 r0=0.8']
drive = torch.where(t >= 100e-12, 2.0, 0.0).to(torch.float64).reshape(-1, 1)
ideal = powers(nl, t=t, drive=drive)[:, 0]
nl_bw = ['.source 1.0@a1 0.0@a2', 'mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9',
         'pd1 b1 v1 level1 r0=0.8 bw=10e9 noise=0', 'pd2 b2 v2 level1 r0=0.8']
slow = powers(nl_bw, t=t, drive=drive)[:, 0]
tau = 1 / (2 * math.pi * 10e9)
# the low-pass follows 1 - exp(-t/tau) from the step, one sample ahead (drive held over the interval)
ref = [0.0 if tt < 100e-12 else 0.8 * (1 - math.exp(-(tt - 100e-12 + 2e-12) / tau)) for tt in t.tolist()]
check('pd bandwidth', slow, ref, tol=1e-9)
bx.plot(t * 1e12, ideal, color=C2, linewidth=2, label='r0 = 0.8 only (ideal)')
bx.plot(t * 1e12, slow, color=C1, linewidth=2, label='bw = 10 GHz, noise = 0')
response_axes(bx, 'time (ps); 1 W of light switched on at 100 ps', 'photocurrent (A)')
bx.set_xlim(0, 300)
bx.legend(frameon=False, fontsize=9, loc='lower right')
save(fig, 'device_pd.png')

print("wrote the device figures; every response curve matches its documented formula")
