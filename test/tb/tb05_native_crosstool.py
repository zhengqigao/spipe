"""TB-05 : native simulator vs HSPICE and Xyce on identical netlists.

The SAME netlist body goes to all three engines; only the control cards differ.
Written from SPEC_E1.md only.
"""
import sys, os, subprocess, tempfile, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, main, REPO

sys.path.insert(0, REPO)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

WRK = tempfile.mkdtemp(prefix='tb05_')
# Site-specific setup for the licensed simulators. Set SPIPE_CAD_SETUP to any
# shell snippet that puts `hspice` and `Xyce` on PATH (e.g. an environment-modules
# `module load`). If unset, both are assumed to be on PATH already.
_CAD_SETUP = os.environ.get('SPIPE_CAD_SETUP', '')
MODLOAD = (_CAD_SETUP + '; ') if _CAD_SETUP else ''


def _modload(mod=None):
    return MODLOAD

# ---------------------------------------------------------------- circuits
NMOS = ".model nch NMOS (LEVEL=1 VTO=0.7 KP=120u LAMBDA=0 CGSO=0 CGDO=0)\n"
PMOS = ".model pch PMOS (LEVEL=1 VTO=-0.7 KP=40u LAMBDA=0 CGSO=0 CGDO=0)\n"

CIRCUITS = {
    'rc_ladder': dict(
        body="""V1 in 0 PULSE(0 1 1n 10p 10p 100n 200n)
R1 in a 1k
C1 a 0 10p
R2 a b 2k
C2 b 0 5p
R3 b out 4k
C3 out 0 2p
""", probe='out', tstep=2e-10, tstop=3e-7),

    'series_rlc': dict(
        body="""V1 in 0 PULSE(0 1 1n 10p 10p 500n 1u)
L1 in mid 1m
R1 mid out 100
C1 out 0 1n
""", probe='out', tstep=2e-9, tstop=4e-6),

    'diode_rect': dict(
        body=""".model dmod D(IS=1e-14 N=1.0 RS=10 CJO=1f VJ=0.7 M=0.5)
V1 in 0 SIN(0 2 10meg)
D1 in out dmod
R1 out 0 5k
C1 out 0 10p
""", probe='out', tstep=1e-9, tstop=4e-7),

    'common_source': dict(
        body=NMOS + """Vdd vdd 0 3.0
Vin g 0 PULSE(0.9 1.3 1n 1n 1n 20n 40n)
M1 d g 0 0 nch W=20u L=1u
RL vdd d 20k
CL d 0 20f
""", probe='d', tstep=2e-10, tstop=1.2e-7),

    'diff_pair': dict(
        body=NMOS + """Vdd vdd 0 3.0
Vcm cm 0 1.2
Vin  inp 0 PULSE(1.2 1.3 1n 1n 1n 20n 40n)
Vinn inn 0 1.2
M1 op inp s 0 nch W=20u L=1u
M2 on inn s 0 nch W=20u L=1u
Rt s 0 10k
R1 vdd op 15k
R2 vdd on 15k
C1 op 0 10f
C2 on 0 10f
""", probe='op', tstep=2e-10, tstop=1.2e-7),

    'inverter_chain': dict(
        body=NMOS + PMOS + """Vdd vdd 0 3.0
Vin a 0 PULSE(0 3 1n 50p 50p 10n 20n)
M1 b a 0 0 nch W=4u L=0.5u
M2 b a vdd vdd pch W=8u L=0.5u
C1 b 0 20f
M3 c b 0 0 nch W=4u L=0.5u
M4 c b vdd vdd pch W=8u L=0.5u
C2 c 0 20f
M5 out c 0 0 nch W=4u L=0.5u
M6 out c vdd vdd pch W=8u L=0.5u
C3 out 0 20f
""", probe='out', tstep=1e-11, tstop=4e-8),
}

# Tolerances calibrated from how much HSPICE and Xyce disagree with EACH OTHER on
# these exact decks (see validate_tb05_refs.py): tol = 1.5x their disagreement, with a
# 1% floor on p99 and 2% on max.  Demanding that our engine match either tool more
# closely than the two commercial tools match each other would not be a fair test.
# Measured 2026-09-21, HSPICE X-2025.06 vs Xyce 7.10:
#   rc_ladder      p99 3.56e-03  max 3.71e-03
#   series_rlc     p99 6.32e-03  max 6.46e-03
#   diode_rect     p99 3.03e-02  max 4.18e-02
#   common_source  p99 4.97e-02  max 2.47e-01   (single-sample undershoot at first edge)
#   diff_pair      p99 3.87e-02  max 4.13e-02
#   inverter_chain p99 9.07e-05  max 1.94e-01   (edge timing only; waveform tracks to 1e-4)
TOL = {
    'rc_ladder':      (0.010,  0.020),
    'series_rlc':     (0.010,  0.020),
    'diode_rect':     (0.0455, 0.0628),
    'common_source':  (0.0745, 0.370),
    'diff_pair':      (0.0580, 0.0620),
    'inverter_chain': (0.010,  0.291),
}
RING_TOL = 0.0753      # hspice 2.3002 GHz vs xyce 2.4157 GHz = 5.02% apart

RING = dict(
    body=NMOS + PMOS + """Vdd vdd 0 3.0
M1 n2 n1 0 0 nch W=4u L=0.5u
M2 n2 n1 vdd vdd pch W=8u L=0.5u
C1 n2 0 50f
M3 n3 n2 0 0 nch W=4u L=0.5u
M4 n3 n2 vdd vdd pch W=8u L=0.5u
C2 n3 0 50f
M5 n1 n3 0 0 nch W=4u L=0.5u
M6 n1 n3 vdd vdd pch W=8u L=0.5u
C3 n1 0 50f
.ic v(n1)=3.0 v(n2)=0 v(n3)=3.0
""", probe='n1', tstep=2e-12, tstop=2e-8)


# ---------------------------------------------------------------- runners
def run_hspice(name, body, probe, tstep, tstop, uic=False):
    deck = (f"* {name}\n" + body +
            f".options post probe reltol=1e-6 abstol=1e-12 vntol=1e-9\n"
            f".print tran v({probe})\n"
            f".tran {tstep:g} {tstop:g}{' uic' if uic else ''}\n.end\n")
    p = os.path.join(WRK, f'hs_{name}.sp')
    open(p, 'w').write(deck)
    cmd = MODLOAD + f'cd {WRK} && hspice {p} -o {p}'
    subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=1200)
    lis = p + '.lis'
    if not os.path.exists(lis):
        return None, None
    return parse_hspice_lis(lis)


def parse_hspice_lis(path):
    import re
    mult = {'': 1, 's': 1, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15,
            'a': 1e-18, 'k': 1e3, 'x': 1e6, 'g': 1e9, 'v': 1.0}
    def cv(tok, unit):
        m = re.match(r'^([+-]?[\d.]+(?:[eE][+-]?\d+)?)([a-zA-Z]*)$', tok)
        if not m:
            return float('nan')
        val = float(m.group(1)); suf = (m.group(2) or unit).lower()
        return val * mult.get(suf[:1] if suf[:1] in mult else '', 1.0)
    t, v, parsing = [], [], False
    for line in open(path, errors='ignore'):
        s = line.strip()
        if s == 'x':
            parsing = True; continue
        if s == 'y':
            parsing = False; continue
        if parsing and s and s[0].isdigit():
            p = s.split()
            if len(p) >= 2:
                t.append(cv(p[0], 's')); v.append(cv(p[1], 'v'))
    if not t:
        return None, None
    return np.array(t), np.array(v)


def run_xyce(name, body, probe, tstep, tstop, uic=False):
    deck = (f"* {name}\n" + body +
            f".print tran format=noindex v({probe})\n"
            f".options timeint reltol=1e-6 abstol=1e-12\n"
            f".tran {tstep:g} {tstop:g}{' uic' if uic else ''}\n.end\n")
    p = os.path.join(WRK, f'xy_{name}.cir')
    open(p, 'w').write(deck)
    cmd = MODLOAD + f'cd {WRK} && Xyce {p}'
    subprocess.run(['bash', '-c', cmd], capture_output=True, text=True, timeout=1200)
    prn = p + '.prn'
    if not os.path.exists(prn):
        return None, None
    t, v = [], []
    for line in open(prn, errors='ignore'):
        p2 = line.split()
        if len(p2) >= 2:
            try:
                t.append(float(p2[0])); v.append(float(p2[1]))
            except ValueError:
                continue
    if not t:
        return None, None
    return np.array(t), np.array(v)


def run_native(name, body, probe, tstep, tstop, uic=False):
    from spipe.electronic.native import Netlist
    deck = f"* {name}\n" + body + f".tran {tstep:g} {tstop:g}{' uic' if uic else ''}\n.end\n"
    ckt = Netlist.from_string(deck)
    r = ckt.tran(tstep, tstop, uic=uic)
    t = np.asarray(r.t.detach().cpu().numpy(), dtype=float)
    v = np.asarray(r.v(probe).detach().cpu().numpy(), dtype=float)
    return t, v


def compare(t1, v1, t2, v2, tstop):
    """Deviation on a common grid, normalised by the reference swing.

    Returns (p99, pmax).  Both are reported because HSPICE and Xyce themselves
    differ by a single-sample undershoot at the first switching edge (measured:
    HSPICE dips to 0.051 V where Xyce reaches 0.110 V on common_source), which a
    plain max-over-window metric lets dominate.  p99 measures whether the whole
    waveform tracks; pmax still catches a genuinely wrong trace.
    """
    g = np.linspace(0.05 * tstop, 0.98 * tstop, 400)
    a = np.interp(g, t1, v1)
    b = np.interp(g, t2, v2)
    swing = max(np.max(np.abs(b)), np.ptp(b), 1e-9)
    d = np.abs(a - b) / swing
    return float(np.percentile(d, 99)), float(np.max(d))


def osc_freq(t, v):
    m = t > 0.4 * t[-1]
    tt, vv = t[m], v[m]
    mid = 0.5 * (np.max(vv) + np.min(vv))
    s = vv - mid
    zc = np.where((np.sign(s[:-1]) < 0) & (np.sign(s[1:]) >= 0))[0]
    if len(zc) < 3:
        return None
    tz = tt[zc]
    return 1.0 / float(np.mean(np.diff(tz)))


def build():
    tb = TB('TB-05  native simulator vs HSPICE / Xyce')
    try:
        import spipe.electronic.native  # noqa
    except Exception as e:
        tb.ok('E1.import', False, f"cannot import: {e!r}")
        return tb

    for name, spec in CIRCUITS.items():
        b, pr, ts, tp = spec['body'], spec['probe'], spec['tstep'], spec['tstop']
        okn, rn = tb.no_raise(f'{name}.native_runs',
                              lambda: run_native(name, b, pr, ts, tp))
        if not okn or rn is None or rn[0] is None:
            continue
        tn, vn = rn

        tol99, tolmax = TOL[name]
        for tool, runner in (('hspice', run_hspice), ('xyce', run_xyce)):
            tt, vv = runner(name, b, pr, ts, tp)
            if tt is None:
                tb.skip(f'{name}.vs_{tool}', f'{tool} produced no output')
                continue
            p99, pmax = compare(tn, vn, tt, vv, tp)
            tb.lt(f'{name}.vs_{tool}', p99, tol99,
                  f"p99 deviation (max={pmax:.2e}, tol_max={tolmax:.2e}); "
                  f"{tool} swing={np.ptp(vv):.4g} V")
            tb.lt(f'{name}.vs_{tool}.max', pmax, tolmax, 'worst single sample')

    # ---- ring oscillator: compare FREQUENCY, not waveform (phase drifts) ----
    b, pr, ts, tp = RING['body'], RING['probe'], RING['tstep'], RING['tstop']
    okn, rn = tb.no_raise('ring.native_runs', lambda: run_native('ring', b, pr, ts, tp, uic=True))
    if okn and rn is not None and rn[0] is not None:
        fn = osc_freq(*rn)
        tb.ok('ring.native_oscillates', fn is not None,
              f"native frequency={fn/1e9 if fn else float('nan'):.4f} GHz")
        th, vh = run_hspice('ring', b, pr, ts, tp, uic=True)
        if th is None or fn is None:
            tb.skip('ring.vs_hspice', 'HSPICE produced no output or native did not oscillate')
        else:
            fh = osc_freq(th, vh)
            if fh is None:
                tb.skip('ring.vs_hspice', 'HSPICE trace did not oscillate')
            else:
                tb.close('ring.vs_hspice', fn, fh, RING_TOL,
                         detail=f"native={fn/1e9:.4f} GHz  hspice={fh/1e9:.4f} GHz "
                                f"(hspice and xyce are themselves 5.02% apart)")

    print(f"\n[tb05 work dir: {WRK}]")
    return tb


if __name__ == '__main__':
    main(build)
