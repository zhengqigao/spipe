"""TB-06 : gradients in the native simulator.

Three independent routes must agree: autograd through the solve, the explicit
time-domain adjoint, and central finite differences. Agreement of all three is
very hard to fake -- a wrong Jacobian breaks at least one of them.
Written from SPEC_E1.md only.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, main, REPO

sys.path.insert(0, REPO)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)


LINEAR = """* rc divider, linear
V1 in 0 PULSE(0 1 0 1p 1p 1 2)
R1 in mid 1k
R2 mid out 2k
C1 out 0 1n
.tran 2e-8 4e-6
.end
"""

NONLIN = """* common-source amplifier, nonlinear
.model nch NMOS (LEVEL=1 VTO=0.7 KP=120u LAMBDA=0.02)
Vdd vdd 0 3.0
Vin g 0 PULSE(0.9 1.3 0 1n 1n 20n 40n)
M1 d g 0 0 nch W=20u L=1u
RL vdd d 20k
CL d 0 20f
.tran 2e-10 6e-8
.end
"""

DIODE = """* diode rectifier
.model dmod D(IS=1e-14 N=1.0 RS=10 CJO=1f VJ=0.7 M=0.5)
V1 in 0 SIN(0 2 10meg)
D1 in out dmod
R1 out 0 5k
C1 out 0 10p
.tran 1e-9 3e-7
.end
"""


def objective_factory(node):
    def obj(res):
        return (res.v(node) ** 2).sum()
    return obj


def build():
    tb = TB('TB-06  native simulator gradients (autograd / adjoint / finite difference)')

    try:
        from spipe.electronic.native import Netlist
    except Exception as e:
        tb.ok('E1.import', False, f"cannot import: {e!r}")
        return tb

    def run_and_grad(netlist, params, node, tstep, tstop):
        """Returns (loss, autograd dict, adjoint dict)."""
        ckt = Netlist.from_string(netlist)
        handles = {}
        for dev, pn in params:
            p = ckt.param(dev, pn)
            p.requires_grad_(True)
            handles[(dev, pn)] = p
        res = ckt.tran(tstep, tstop)
        loss = objective_factory(node)(res)
        loss.backward()
        auto = {k: (None if v.grad is None else float(v.grad)) for k, v in handles.items()}

        ckt2 = Netlist.from_string(netlist)
        adj = ckt2.adjoint_grad(objective_factory(node), params=list(params))
        adj = {k: float(v) for k, v in adj.items()}
        return float(loss), auto, adj

    def fd(netlist, dev, pn, node, tstep, tstop, rel=1e-5):
        ckt = Netlist.from_string(netlist)
        p0 = float(ckt.param(dev, pn))
        h = abs(p0) * rel if p0 != 0 else rel
        out = []
        for s in (+1, -1):
            c = Netlist.from_string(netlist)
            with torch.no_grad():
                c.param(dev, pn).copy_(torch.tensor(p0 + s * h))
            r = c.tran(tstep, tstop)
            out.append(float(objective_factory(node)(r)))
        return (out[0] - out[1]) / (2 * h), p0

    # ------------------------------------------------------------------
    suites = [
        ('LIN', LINEAR, [('R1', 'R'), ('R2', 'R'), ('C1', 'C')], 'out', 2e-8, 4e-6),
        ('MOS', NONLIN, [('M1', 'W'), ('M1', 'L'), ('RL', 'R')], 'd', 2e-10, 6e-8),
        ('DIO', DIODE, [('R1', 'R'), ('C1', 'C')], 'out', 1e-9, 3e-7),
    ]

    for tag, nl, params, node, ts, tp in suites:
        ok, got = tb.no_raise(f'{tag}.runs', lambda nl=nl, params=params, node=node, ts=ts, tp=tp:
                              run_and_grad(nl, params, node, ts, tp))
        if not ok or got is None:
            continue
        loss, auto, adj = got
        tb.ok(f'{tag}.loss_finite', np.isfinite(loss), f"loss={loss:.8g}")

        for (dev, pn) in params:
            key = (dev, pn)
            a = auto.get(key)
            d = adj.get(key)
            if a is None:
                tb.ok(f'{tag}.{dev}.{pn}.autograd_exists', False,
                      'autograd produced no .grad -- the solve is not differentiable')
                continue
            tb.ok(f'{tag}.{dev}.{pn}.autograd_exists', True, f"d(loss)/d({pn})={a:.8g}")

            okf, r = tb.no_raise(f'{tag}.{dev}.{pn}.fd_runs',
                                 lambda dev=dev, pn=pn, nl=nl, node=node, ts=ts, tp=tp:
                                 fd(nl, dev, pn, node, ts, tp))
            if okf and r is not None:
                g_fd, p0 = r
                scale = max(abs(g_fd), abs(a), 1e-300)
                tb.lt(f'{tag}.{dev}.{pn}.autograd_vs_fd', abs(a - g_fd) / scale, 1e-6,
                      f"autograd={a:.10g} fd={g_fd:.10g} (param={p0:.6g})")
                # a gradient that is identically zero would trivially "agree" with
                # nothing -- require it to be genuinely non-trivial
                tb.ok(f'{tag}.{dev}.{pn}.nontrivial', abs(g_fd) > 1e-14,
                      f"|d(loss)/d({pn})|={abs(g_fd):.3e} must be measurably non-zero")

            if d is None:
                tb.ok(f'{tag}.{dev}.{pn}.adjoint_exists', False, 'adjoint_grad returned no entry')
            else:
                tb.ok(f'{tag}.{dev}.{pn}.adjoint_exists', True, f"adjoint={d:.8g}")
                scale = max(abs(a), abs(d), 1e-300)
                tb.lt(f'{tag}.{dev}.{pn}.adjoint_vs_autograd', abs(a - d) / scale, 1e-6,
                      f"autograd={a:.10g} adjoint={d:.10g}")

    # ---- the adjoint must be a single backward sweep, not N forward solves ----
    # Scaling test: cost of adjoint_grad must be ~flat in the number of parameters.
    try:
        from spipe.electronic.native import Netlist as NL2
        many = [('M1', 'W'), ('M1', 'L'), ('RL', 'R'), ('CL', 'C'), ('Vdd', 'DC')]
        avail = []
        c = NL2.from_string(NONLIN)
        for p in many:
            try:
                c.param(*p); avail.append(p)
            except Exception:
                pass
        if len(avail) >= 3:
            def timed(ps):
                c = NL2.from_string(NONLIN)
                t0 = time.time()
                c.adjoint_grad(objective_factory('d'), params=ps)
                return time.time() - t0
            t1 = timed(avail[:1]); tn = timed(avail)
            ratio = tn / max(t1, 1e-6)
            tb.lt('ADJ.cost_scaling', ratio, 2.0,
                  f"{len(avail)} params took {ratio:.2f}x the time of 1 param "
                  f"({t1:.3f}s -> {tn:.3f}s); a true adjoint is ~flat, "
                  f"repeated forward solves would scale ~{len(avail)}x")
        else:
            tb.skip('ADJ.cost_scaling', f'only {len(avail)} parameters resolvable')
    except Exception as e:
        tb.ok('ADJ.cost_scaling', False, f"raised {e!r}")

    # ---- float64 throughout --------------------------------------------
    try:
        c = Netlist.from_string(LINEAR)
        r = c.tran(2e-8, 4e-6)
        tb.ok('E1.float64_time', r.t.dtype == torch.float64, f"res.t.dtype={r.t.dtype}")
        tb.ok('E1.float64_volt', r.v('out').dtype == torch.float64, f"res.v dtype={r.v('out').dtype}")
        tb.ok('E1.param_float64', c.param('R1', 'R').dtype == torch.float64,
              f"param dtype={c.param('R1','R').dtype}")
    except Exception as e:
        tb.ok('E1.float64_time', False, f"raised {e!r}")

    # ---- external-block hook exists and is wired through the solver ------
    try:
        c = Netlist.from_string(LINEAR)
        tb.ok('E1.add_external_block_exists', hasattr(c, 'add_external_block'),
              'required for the later unified electronic-photonic DAE')
    except Exception as e:
        tb.ok('E1.add_external_block_exists', False, f"raised {e!r}")

    return tb


if __name__ == '__main__':
    main(build)
