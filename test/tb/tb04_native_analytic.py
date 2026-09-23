"""TB-04 : native analog simulator vs CLOSED-FORM analytic solutions.

Ground truth here is mathematics, not another simulator, so these checks cannot be
gamed by matching a reference trace. Written from SPEC_E1.md only.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, main, REPO

sys.path.insert(0, REPO)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)


def get_netlist():
    from spipe.electronic.native import Netlist
    return Netlist


def arr(x):
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, 'detach') else x, dtype=float)


def build():
    tb = TB('TB-04  native simulator vs analytic solutions')

    try:
        Netlist = get_netlist()
    except Exception as e:
        tb.ok('E1.import', False, f"cannot import spipe.electronic.native.Netlist: {e!r}")
        return tb
    tb.ok('E1.import', True, 'spipe.electronic.native.Netlist imported')

    # ---------- 1. RC step response --------------------------------------
    # V=1 through R=1k into C=1n, uic with v(out)=0  ->  v(t) = 1 - exp(-t/RC)
    R, C = 1e3, 1e-9
    tau = R * C
    nl = f"""* rc
V1 in 0 1
R1 in out {R}
C1 out 0 {C}
.ic v(out)=0
.tran {tau/200} {5*tau} uic
.end
"""
    ok, res = tb.no_raise('RC.runs', lambda: Netlist.from_string(nl).tran(tau / 200, 5 * tau, uic=True))
    if ok and res is not None:
        t, v = arr(res.t), arr(res.v('out'))
        ana = 1.0 - np.exp(-t / tau)
        m = t > 0.02 * tau                      # skip the first couple of steps
        err = np.max(np.abs(v[m] - ana[m]) / np.maximum(np.abs(ana[m]), 1e-12))
        tb.lt('RC.max_rel_err', err, 1e-6, f"tau={tau:g}, {len(t)} points")
        tb.close('RC.final_value', v[-1], 1.0 - math.exp(-5.0), 1e-5)

    # ---------- 2. Series RLC step response -------------------------------
    # V step into L-R-C; v_C(t) = 1 - e^{-z w0 t}(cos wd t + z/sqrt(1-z^2) sin wd t)
    L, Rr, Cc = 1e-3, 100.0, 1e-9
    w0 = 1.0 / math.sqrt(L * Cc)
    z = (Rr / 2.0) * math.sqrt(Cc / L)
    wd = w0 * math.sqrt(1 - z * z)
    # 60 points/radian over 3 decay constants: measured max rel err 2.24e-08,
    # versus 2.00e-09 at 400 pts/rad over 12 decays -- i.e. 25x the cost for no
    # benefit against the 1e-5 tolerance (859 s vs 62 s on this engine).
    _ts, _tp = 1 / (w0 * 60), 3 / (z * w0)
    nl = f"""* rlc
V1 in 0 1
L1 in mid {L}
R1 mid out {Rr}
C1 out 0 {Cc}
.ic v(out)=0 i(L1)=0
.tran {_ts} {_tp} uic
.end
"""
    ok, res = tb.no_raise('RLC.runs',
                          lambda: Netlist.from_string(nl).tran(_ts, _tp, uic=True))
    if ok and res is not None:
        t, v = arr(res.t), arr(res.v('out'))
        ana = 1 - np.exp(-z * w0 * t) * (np.cos(wd * t) + z / math.sqrt(1 - z * z) * np.sin(wd * t))
        err = np.max(np.abs(v - ana)) / np.max(np.abs(ana))
        tb.lt('RLC.max_rel_err', err, 1e-5, f"w0={w0:.4g} zeta={z:.4g} wd={wd:.4g}")
        # recover the damped frequency from zero crossings of (v-1)
        s = v - 1.0
        zc = np.where(np.sign(s[:-1]) != np.sign(s[1:]))[0]
        if len(zc) >= 3:
            tz = t[zc]
            wd_meas = math.pi / np.mean(np.diff(tz))
            tb.close('RLC.damped_freq', wd_meas, wd, 1e-3)
        else:
            tb.skip('RLC.damped_freq', f'only {len(zc)} zero crossings found')

    # ---------- 3. Ideal integrators (charge / flux conservation) ---------
    # I into C:  v = I t / C exactly, for any consistent integration scheme
    Iv, Cv = 1e-3, 1e-6
    nl = f"""* cap integrator
I1 0 out {Iv}
C1 out 0 {Cv}
.ic v(out)=0
.tran 1e-7 1e-4 uic
.end
"""
    ok, res = tb.no_raise('CAP.runs', lambda: Netlist.from_string(nl).tran(1e-7, 1e-4, uic=True))
    if ok and res is not None:
        t, v = arr(res.t), arr(res.v('out'))
        ana = Iv * t / Cv
        err = np.max(np.abs(v - ana)) / max(np.max(np.abs(ana)), 1e-12)
        tb.lt('CAP.charge_conservation', err, 1e-10, 'v = I*t/C must be exact')

    # V across L: i = V t / L exactly
    Vv, Lv = 1.0, 1e-3
    nl = f"""* ind integrator
V1 in 0 {Vv}
L1 in 0 {Lv}
.ic i(L1)=0
.tran 1e-7 1e-4 uic
.end
"""
    ok, res = tb.no_raise('IND.runs', lambda: Netlist.from_string(nl).tran(1e-7, 1e-4, uic=True))
    if ok and res is not None:
        t = arr(res.t)
        try:
            i = np.abs(arr(res.i('L1')))
        except Exception:
            i = np.abs(arr(res.i('V1')))
        ana = Vv * t / Lv
        err = np.max(np.abs(i - ana)) / max(np.max(np.abs(ana)), 1e-12)
        tb.lt('IND.flux_conservation', err, 1e-10, 'i = V*t/L must be exact')

    # ---------- 4. Linear DC vs an independent numpy MNA solve ------------
    nl = """* resistor network
V1 1 0 2.5
R1 1 2 1k
R2 2 0 2k
R3 2 3 3.3k
R4 3 0 4.7k
R5 1 3 10k
.op
.end
"""
    ok, op = tb.no_raise('DC.runs', lambda: Netlist.from_string(nl).op())
    if ok and op is not None:
        # independent solve: nodes 1,2,3 with 1 fixed by the source
        G = np.zeros((3, 3)); rhs = np.zeros(3)
        def add(n1, n2, g):
            for a, b, s in ((n1, n1, g), (n2, n2, g), (n1, n2, -g), (n2, n1, -g)):
                if a and b:
                    G[a - 1, b - 1] += s
        for (a, b, r) in [(1, 2, 1e3), (2, 0, 2e3), (2, 3, 3.3e3), (3, 0, 4.7e3), (1, 3, 10e3)]:
            add(a, b, 1.0 / r)
        # node 1 forced to 2.5
        G[0, :] = 0; G[0, 0] = 1.0; rhs[0] = 2.5
        ref = np.linalg.solve(G, rhs)
        got = {str(k).lower(): float(v) for k, v in op.items()}
        for idx, nm in enumerate(['1', '2', '3']):
            if nm in got:
                tb.close(f'DC.node{nm}', got[nm], ref[idx], 1e-12)
            else:
                tb.ok(f'DC.node{nm}', False, f"node {nm} missing from op(); keys={sorted(got)[:8]}")

    # ---------- 5. Diode: implicit Shockley consistency -------------------
    # We do NOT pin the thermal voltage: simulators differ in their default
    # temperature and physical constants.  HSPICE itself sits 6.6e-4 away from
    # ideal Shockley at 298.15 K on this exact circuit (measured).  So we fit T
    # over a physical range and require both a small residual AND a sane T.
    IS, N = 1e-14, 1.0
    for vin, Rs in [(0.7, 1e3), (1.0, 1e3), (2.0, 10e3)]:
        nl = f"""* diode
.model dmod D(IS={IS} N={N} RS=0 CJO=0)
V1 in 0 {vin}
R1 in out {Rs}
D1 out 0 dmod
.op
.end
"""
        ok, op = tb.no_raise(f'DIODE.runs_{vin}', lambda nl=nl: Netlist.from_string(nl).op())
        if ok and op is not None:
            got = {str(k).lower(): float(v) for k, v in op.items()}
            if 'out' in got:
                vd = got['out']
                i_res = (vin - vd) / Rs                     # current through R
                best = None
                for Tk in [290.0 + 0.05 * k for k in range(401)]:   # 290..310 K
                    VT = 1.380649e-23 * Tk / 1.602176634e-19
                    i_dio = IS * (math.exp(vd / (N * VT)) - 1)
                    r = abs(i_dio - i_res) / max(abs(i_res), 1e-300)
                    if best is None or r < best[1]:
                        best = (Tk, r, i_dio)
                Tk, r, i_dio = best
                tb.lt(f'DIODE.kcl_{vin}', r, 5e-3,
                      f"vd={vd:.6f} V  I_R={i_res:.4e} I_D={i_dio:.4e} best-fit T={Tk:.2f}K")
                tb.ok(f'DIODE.temp_physical_{vin}', 292.0 <= Tk <= 308.0,
                      f"implied device temperature {Tk:.2f} K must be room temperature")
            else:
                tb.ok(f'DIODE.kcl_{vin}', False, f"no node 'out'; keys={sorted(got)[:8]}")

    # ---------- 6. MOSFET level-1 square law, all three regions ------------
    # Cross-checked against HSPICE X-2025.06 on identical model cards (2026-09-21):
    #   triode     HSPICE 5.280000e-04 A  vs this formula  rel err 4.1e-16
    #   saturation HSPICE 1.014000e-03 A  vs this formula  rel err 4.3e-16
    #   cutoff     HSPICE 2.010000e-12 A  (leakage, not exactly zero)
    VTO, KP, W, Lch = 0.7, 120e-6, 10e-6, 1e-6
    beta = KP * W / Lch
    cases = [('cutoff', 0.5, 1.0, 0.0),
             ('triode', 2.0, 0.4, beta * ((2.0 - VTO) * 0.4 - 0.4 ** 2 / 2)),
             ('satur.', 2.0, 2.5, 0.5 * beta * (2.0 - VTO) ** 2)]
    for tag, vgs, vds, iexp in cases:
        nl = f"""* nmos level1, LAMBDA=0 so both region formulas are unambiguous
.model nch NMOS (LEVEL=1 VTO={VTO} KP={KP} LAMBDA=0)
Vg g 0 {vgs}
Vd d 0 {vds}
M1 d g 0 0 nch W={W} L={Lch}
.tran 1e-9 5e-9 uic
.end
"""
        ok, res = tb.no_raise(f'MOS.runs_{tag}', lambda nl=nl: Netlist.from_string(nl).tran(1e-9, 5e-9, uic=True))
        if ok and res is not None:
            try:
                idrain = abs(float(arr(res.i('Vd'))[-1]))
            except Exception as e:
                tb.ok(f'MOS.{tag}', False, f"res.i('Vd') failed: {e!r}")
                continue
            if iexp == 0.0:
                # HSPICE gives 2.01e-12 A here, so demand negligible-vs-saturation
                # (I_sat = 1.014e-3 A) rather than exactly zero
                tb.lt(f'MOS.{tag}', idrain, 1e-9,
                      'cutoff current must be negligible vs I_sat=1.014e-3 A')
            else:
                tb.close(f'MOS.{tag}', idrain, iexp, 1e-9,
                         detail=f"vgs={vgs} vds={vds} beta={beta:.4g}")

    # ---------- 7. PULSE edges must not be stepped over --------------------
    nl = """* pulse breakpoints
V1 in 0 PULSE(0 1 10n 10p 10p 20n 40n)
R1 in out 1k
C1 out 0 1f
.tran 1n 100n
.end
"""
    ok, res = tb.no_raise('PULSE.runs', lambda: Netlist.from_string(nl).tran(1e-9, 100e-9))
    if ok and res is not None:
        t, v = arr(res.t), arr(res.v('out'))
        # RC = 1 ps, so out tracks in within a few ps of each edge
        def at(ts):
            return v[int(np.argmin(np.abs(t - ts)))]
        tb.lt('PULSE.low_before_edge', abs(at(5e-9)), 1e-3, 'before the first edge')
        tb.close('PULSE.high_on_plateau', at(25e-9), 1.0, 1e-3, detail='mid first pulse')
        tb.lt('PULSE.low_after_fall', abs(at(45e-9)), 1e-3, 'after the falling edge')
        tb.ok('PULSE.reached_full_swing', abs(np.max(v) - 1.0) < 1e-3,
              f"max(v)={np.max(v):.6f} -- a stepped-over edge shows up as a clipped peak")

    return tb


if __name__ == '__main__':
    main(build)
