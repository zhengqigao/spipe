"""L.2.c -- automatic bias control: negative feedback that holds an MZM at quadrature.

What this demonstrates
----------------------
This is the "feedback works and is boring" case, and it is the one whose answer is checkable in
closed form.

    laser -> MZM -> bar port -> monitor photodiode -> transimpedance R
                 ^                                        |
                 |                                        v
                 +------ buffer <- integrator <- (v_ref - v_monitor)

The monitor photocurrent is compared with a reference and the error is *integrated*, so the
loop has infinite DC gain and the only possible steady state is the one with zero error::

    v_monitor = R * r0 * P_in * 10^(-il/10) * sin^2(pi * (v_rf - vbias) / (2 * vpi))
    v_ref     = R * r0 * P_in * 10^(-il/10) / 2          (half of full scale, by construction)

    =>  sin^2(...) = 1/2  =>  pi (v_rf - vbias) / (2 vpi) = pi/4  =>  v_rf* = vbias + vpi / 2

i.e. **the setpoint is the bias plus a quarter wave, exactly, whatever the loop gain is** --
which is the point of an integrator, and what makes this a real closed-form check rather than
a plausibility argument.  The example runs the circuit for three different MZM ``vbias`` values
(standing in for fabrication spread or a thermal drift) and shows the loop tracking each one.

Feedback in the sense of Definition 1 of the paper is present: ``I_pd`` -> ``v(nmon)`` ->
``v(nint)`` -> ``v(nrf)`` -> the drive of ``mzm0``.

Run it::

    python examples/bias_control.py
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch

from _common import (add_common_arguments, banner, report_convergence, report_failure,
                     resolve_spice_exe, set_config, spice_options, settling_time, work_dir, write_netlist)

from spipe import Circuit, config
from spipe.core.core import FixedPointError

# ---------------------------------------------------------------------------- design point
VPI = 2.0             # MZM half-wave voltage [V]
IL_DB = 3.0           # insertion loss [dB]
ER_DB = 1e9           # extinction ratio [dB] -- ideal, so the closed form is exact
P_IN = 1.0e-3         # laser power [W]
R0 = 0.9              # monitor responsivity [A/W]
R_MON = 2.0e3         # transimpedance load [Ohm]
GM = 1.0e-3           # integrator transconductance [S]
C_INT = 1.0e-12       # integrating capacitor [F]

#: Full-scale monitor voltage, i.e. the value of ``v(nmon)`` at peak bar transmission.
K_FULL_SCALE = R_MON * R0 * P_IN * 10 ** (-IL_DB / 10)
#: Half of it: the reference that puts the modulator at quadrature.
V_REF = 0.5 * K_FULL_SCALE

#: Small-signal loop crossover, ``omega_c = gm * dv_mon/dv_rf / C``, evaluated at quadrature
#: where ``dv_mon/dv_rf = K * pi / (2 * vpi)``.
OMEGA_C = GM * K_FULL_SCALE * math.pi / (2 * VPI) / C_INT

T_STOP = 20e-9
NUM_T = 101

#: MZM bias offsets to correct for -- fabrication spread / thermal drift, in volts.
DRIFTS = (0.0, 0.3, -0.4)


def build_netlist(path: str, vbias: float) -> str:
    electronic = f"""* L.2.c -- automatic bias control loop (negative feedback, integrating controller)
{spice_options(T_STOP, NUM_T).rstrip()}
* Monitor photocurrent -> transimpedance load.
Rmon nmon 0 {R_MON:g}
* Reference: half of full-scale monitor voltage == quadrature.
Vref nref 0 {V_REF:.12g}
* Integrating error amplifier: i(nint) = gm * (v(nref) - v(nmon)).
Gint 0 nint nref nmon {GM:g}
Cint nint 0 {C_INT:g}
Rint nint 0 100meg
* Unity-gain buffer onto the modulator RF port.
Edrv nrf 0 nint 0 1.0
.tran 0 {T_STOP} {NUM_T}
.print tran v(nmon) v(nint) v(nrf)
"""

    photonic = f"""# MZM whose bar output is monitored; the bias is the drift the loop absorbs.
mzm0 nin nin_unused nbar ncross nrf level1 vpi={VPI} vbias={vbias} er={ER_DB:.0f} il={IL_DB} act_l=1e-3
pd1 nbar nmon level1 r0={R0}
    .mode neff=2.35
    .freq 193.5e12 193.5e12 1
    .source {math.sqrt(P_IN):.12g}@nin power=0.05w eff=0.2
    .prob nbar ncross
    .end
"""
    return write_netlist(path, electronic, photonic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()

    directory = work_dir(args, 'bias_control')
    set_config(args, max_iter=120, rtol=1e-5, atol=1e-5)
    spice_exe = resolve_spice_exe(args)

    banner('L.2.c  automatic bias control loop (MZM held at quadrature)')
    print(f"  full-scale monitor voltage K = R*r0*P_in*10^(-il/10) = {K_FULL_SCALE:.9f} V")
    print(f"  reference                v_ref = K/2                 = {V_REF:.9f} V")
    print(f"  loop crossover           omega_c/2pi                 = {OMEGA_C / (2 * math.pi):.6g} Hz")
    print(f"  closed-form setpoint     v_rf* = vbias + vpi/2       = vbias + {VPI / 2:g} V")

    rows: List[Tuple[float, float, float, float, Dict]] = []
    for drift in DRIFTS:
        netlist = build_netlist(os.path.join(directory, f'bias_control_{drift:+.2f}.sp'), drift)
        circuit = Circuit(netlist, spice_exe=spice_exe,
                          spice_wrk_dir=os.path.join(directory, 'tmp'),
                          spice_file_name='bias_control.sp')
        print(f"\n  -- MZM vbias = {drift:+.3f} V ------------------------------------------------")
        try:
            prob_e, _, param_e, param_p, _ = circuit.simulate(x0=torch.zeros(1, dtype=torch.float64))
        except FixedPointError as error:
            report_failure(error)
            continue
        info = circuit.fixed_point_info
        report_convergence(info)

        drive = param_p[:, 0]
        monitor = prob_e['v(nmon)']
        settled = float(drive[-1])
        expected = drift + VPI / 2
        transmission = math.sin(math.pi * (settled - drift) / (2 * VPI)) ** 2
        t_settle = settling_time(drive, circuit.time, tolerance=0.01)
        rows.append((drift, settled, expected, t_settle, info))

        print(f"      settled drive   v_rf   = {settled:.9f} V")
        print(f"      closed form     v_rf*  = {expected:.9f} V   "
              f"(error {settled - expected:+.3e} V, {abs(settled - expected) / VPI * 100:.4f}% of vpi)")
        print(f"      monitor voltage v(nmon)= {float(monitor[-1]):.9f} V   "
              f"(reference {V_REF:.9f} V, error {float(monitor[-1]) - V_REF:+.3e} V)")
        print(f"      bar transmission       = {transmission:.9f}   (quadrature is 0.5)")
        print(f"      monitor photocurrent   = {float(param_e[-1, 0]) * 1e6:.6f} uA")
        print(f"      1% settling time       = {t_settle * 1e9:.4f} ns "
              f"(1/omega_c = {1e9 / OMEGA_C:.4f} ns)")

    print('\n  -- summary ------------------------------------------------------------------')
    print('      vbias [V]   v_rf simulated [V]   v_rf closed form [V]   error [V]   iters')
    for drift, settled, expected, _, info in rows:
        print(f"      {drift:+8.3f}   {settled:18.9f}   {expected:20.9f}   "
              f"{settled - expected:+9.2e}   {info['iters']:5d}")

    if args.plot:
        _plot(directory, rows)
    return 0


def _plot(directory: str, rows) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    drifts = [row[0] for row in rows]
    measured = [row[1] for row in rows]
    expected = [row[2] for row in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(drifts, expected, 'o--', label='closed form: vbias + vpi/2', markersize=10)
    plt.plot(drifts, measured, 'x', label='simulated', markersize=12, markeredgewidth=2)
    plt.xlabel('MZM vbias (V)')
    plt.ylabel('settled MZM drive (V)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('L.2.c bias control: settled drive vs closed-form setpoint')
    plt.tight_layout()
    path = os.path.join(directory, 'bias_control.png')
    plt.savefig(path, dpi=120)
    print(f"      plot written to {path}")


if __name__ == '__main__':
    raise SystemExit(main())
