"""Optical memory: a delay line and a ring's ring-up, with mode='envelope'.

WHAT THIS DEMONSTRATES
----------------------
The default photonic solver is quasi-static: at every time sample it solves the network in steady
state, so light takes no time to cross it. mode='envelope' keeps the optical memory. This script
switches a modulator on at 200 ps and watches the light arrive:

1. through a 100 ps delay line -- quasi-static sees it at once, envelope 100 ps later;
2. inside a ring resonator -- envelope fills it with the ring's photon lifetime, compared with the
   analytic value -T_rt / ln(a t).

Two things about sizing the .freq grid, both explained in docs/envelope.md:
- the band must be wide compared with 1/dt (here 400 GHz against 100 GHz), and the carrier you
  read must have a full window -- the centre one does;
- the spacing df must make 1/(2 df) longer than the circuit's memory. The ring uses 0.25 GHz
  (2 ns) for a 163 ps lifetime; at 1 GHz (500 ps) the fitted lifetime is badly wrong.

Run:  python examples/envelope_delay_ring.py      (a few seconds)
"""
import math
import os as _os
import sys as _sys
import warnings

# Make `import spipe` work from a source checkout without installing.
try:
    import spipe as _probe  # noqa: F401
except ImportError:
    _here = _os.path.dirname(_os.path.abspath(__file__))
    for _ in range(6):
        _cand = _os.path.join(_here, 'src')
        if _os.path.isdir(_os.path.join(_cand, 'spipe')):
            _sys.path.insert(0, _cand)
            break
        _here = _os.path.dirname(_here)

import numpy as np
import torch
from spipe import Photonic

c = 299792458.0
dt, T = 10e-12, 120                                   # 10 ps samples, 1.2 ns
t = torch.arange(T, dtype=torch.float64) * dt
drive = (t <= 200e-12).to(torch.float64).reshape(-1, 1) * 2.0    # 2 V (off) until 200 ps, then 0 V (on)

# --- 1. a 100 ps delay line -------------------------------------------------------------
tau = 100e-12
L = tau * c / 4.0                                      # ng = 4
delay = [l + "\n" for l in [
    ".mode neff=4.0 ng=4.0 wl=1552.5e-9",
    ".freq 192.9e12 193.3e12 401",                     # 1 GHz spacing, 400 GHz band
    ".source 1.0@a1 0.0@a2",
    "mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9",
    f"wg0 b2 c1 l={L!r}",
    "pd1 c1 v1 level1 r0=1.0", "pd2 b1 v2 level1 r0=1.0", ".prob c1"]]
ph = Photonic(delay)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                    # edge-carrier notice; we read the centre carrier
    _, qs, _ = ph.simulate(t, drive)
    _, env, _ = ph.simulate(t, drive, mode='envelope')
k = 200                                                # centre carrier: a full window
p_qs = qs['c1'][:, k, 1].abs() ** 2
p_env = env['c1'][:, k, 1].abs() ** 2
cross = lambda p: float(t[int((p > 0.5).nonzero()[0])])
print(f"delay line: quasistatic 50% at {cross(p_qs)*1e12:.0f} ps, envelope at {cross(p_env)*1e12:.0f} ps "
      f"(drive at 200 ps, delay {tau*1e12:.0f} ps)")

# --- 2. a ring's ring-up -----------------------------------------------------------------
T_rt, t_c, a = 10e-12, 0.95, 0.99                      # round trip, through coupling, loss per trip
ring = [l + "\n" for l in [
    ".mode neff=4.0 ng=4.0 wl=1552.5e-9",
    ".freq 192.9e12 193.3e12 1601",                    # 0.25 GHz: +/- 2 ns of memory; centre on resonance
    ".source 1.0@a1 0.0@a2",
    "mzm0 a1 a2 b1 b2 v level1 vpi=2.0 act_l=1e-9",
    f"mzi0 b2 r2 c1 r1 theta={math.acos(t_c)!r}",
    f"wg0 r1 r2 l={T_rt * c / 4.0!r} alpha={a!r}",
    "pd1 c1 v1 level1 r0=1.0", "pd2 b1 v2 level1 r0=1.0", ".prob r1"]]
ph = Photonic(ring)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, qs, _ = ph.simulate(t, drive)
    _, env, _ = ph.simulate(t, drive, mode='envelope')
k = 800                                                # centre carrier of the finer grid
steady = qs['r1'][-1, k, 1]                            # the quasistatic answer is the steady state
field = env['r1'][:, k, 1]                             # inside the ring
gap = (field - steady).abs()[22:90]                    # 220 .. 900 ps
fit = -1.0 / np.polyfit(t[22:90].numpy(), np.log(gap.numpy()), 1)[0]
print(f"ring: build-up {float(steady.abs()):.4f}x (analytic {math.sqrt(1 - t_c**2) / (1 - a * t_c):.4f}x), lifetime fitted {fit*1e12:.1f} ps, "
      f"analytic {-T_rt / math.log(a * t_c) * 1e12:.1f} ps")
