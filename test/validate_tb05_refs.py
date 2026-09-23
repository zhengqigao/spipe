"""Calibrate TB-05: run every reference circuit on HSPICE and Xyce and compare them
to EACH OTHER.  Two commercial tools disagreeing sets the floor on what we can fairly
demand of our own engine, so this prints the tolerance constants for TB-05 to use.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tb'))
import numpy as np
import tb05_native_crosstool as T

print(f"work dir: {T.WRK}\n")
print(f"{'circuit':<18}{'hs pts':>8}{'xy pts':>8}{'swing':>9}{'p99':>10}{'max':>10}   -> TOL entry")
out = {}
for name, spec in T.CIRCUITS.items():
    b, pr, ts, tp = spec['body'], spec['probe'], spec['tstep'], spec['tstop']
    th, vh = T.run_hspice(name, b, pr, ts, tp)
    tx, vx = T.run_xyce(name, b, pr, ts, tp)
    if th is None or tx is None:
        print(f"{name:<18}{'FAIL' if th is None else len(th):>8}"
              f"{'FAIL' if tx is None else len(tx):>8}   TOOL FAILED")
        continue
    p99, pmax = T.compare(th, vh, tx, vx, tp)
    # our engine must agree with each tool at least as well as they agree with
    # each other, with 1.5x headroom and a 1% floor
    t99 = max(0.01, 1.5 * p99)
    tmx = max(0.02, 1.5 * pmax)
    out[name] = (t99, tmx)
    print(f"{name:<18}{len(th):>8}{len(tx):>8}{np.ptp(vh):>9.4f}{p99:>10.2e}{pmax:>10.2e}"
          f"   ({t99:.3g}, {tmx:.3g})")

b, pr, ts, tp = T.RING['body'], T.RING['probe'], T.RING['tstep'], T.RING['tstop']
th, vh = T.run_hspice('ring', b, pr, ts, tp, uic=True)
tx, vx = T.run_xyce('ring', b, pr, ts, tp, uic=True)
fh = T.osc_freq(th, vh) if th is not None else None
fx = T.osc_freq(tx, vx) if tx is not None else None
if fh and fx:
    d = abs(fh - fx) / fh
    print(f"\nring: hspice={fh/1e9:.4f} GHz  xyce={fx/1e9:.4f} GHz  disagree={d:.2%}"
          f"   -> RING_TOL = {max(0.02, 1.5*d):.3g}")

print("\nTOL = {")
for k, (a, c) in out.items():
    print(f"    '{k}': ({a:.4g}, {c:.4g}),")
print("}")
