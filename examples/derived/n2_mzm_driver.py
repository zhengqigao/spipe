"""N2 -- a Level-1 CMOS MZM driver into the realistic ``level3`` RC modulator load.

This is the circuit the whole differentiable-co-simulation argument is about, so it is the
cleanest one in the set: one driver, one modulator, one detector, nothing else.

    data -> inverter chain (level=1 CMOS) -> mzm RF port (mod_level3 RC load) -> pd

Why ``level3`` matters
----------------------
``mod_level1``, which every shipped example uses, is ``1e9`` Ohm in parallel with ``1e-18`` F:
an open circuit and one attofarad.  A driver terminated in that sees *no load at all*, so the
dominant electro-optic bandwidth limit of a real link -- the driver pushing charge into the
junction capacitance of the phase shifter through the access resistance of the doped slab --
is simply missing from the simulation, and the answer to "how wide should this transistor be"
is "it does not matter".  ``mod_level3`` is a two-section RC ladder for a depletion-mode
silicon MZM segment (200 fF junction, 10 Ohm series, 20 fF pad), which is a *finite* load, and
then the width of the driver decides the optical eye.

d(optical) / dW
---------------
With that load in place there is a real derivative of an *optical* figure of merit with respect
to an *electronic* design variable, and that is what this example measures:

    W          NMOS width of the output stage (the PMOS is 3W, for a symmetric drive)
    eye        optical eye height at the detector, i.e. the photocurrent difference between the
               end of a '1' bit and the end of a '0' bit, in microamps

The example sweeps W, reports ``eye(W)``, and forms ``d eye / dW`` by central differences.

A note on where the analytic gradient would come from: SPIPE's photonic solver already supplies
``d(optical)/d(drive)`` in closed form (``Simulate.backward`` in ``spipe.photonic.photonic``),
and the Xyce back end can supply ``d(drive)/d(parameter)`` through ``.SENS``
(``SimulateXyce.backward``).  The HSPICE back end raises ``NotImplementedError`` for the
backward pass, so with ``--sim hspice`` -- the default, and the only engine these models were
calibrated on -- the chain rule cannot be closed inside the tool, and this example measures the
derivative by finite differences instead.  It does not pretend otherwise.

Run it::

    python examples/derived/n2_mzm_driver.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import os as _os, sys as _sys
# Make `import spipe` work from a source checkout without installing.
# Walks up to the repo root and puts `src/` on the path; a pip-installed
# spipe takes precedence because this only appends if the import fails.
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

import torch

from _common import (add_common_arguments, banner, report_convergence, resolve_spice_exe,
                     set_config, work_dir, write_netlist)

from spipe import Circuit, config

HERE = os.path.dirname(os.path.abspath(__file__))

VDD = 3.3
VPI = 3.3            # so a rail-to-rail drive is exactly one half-wave voltage
VBIAS = 0.0
IL_DB = 3.0
ER_DB = 30.0         # 30 dB -- a realistic silicon MZM, not an ideal one
P_IN = 1.0e-3
R0 = 0.9

# level3 modulator load: depletion-mode segment.  These are the *defaults* of
# spipe.electronic.electronic.ModModel's 'level3' entry, repeated here only so the report can
# print them.  They are deliberately not passed on the mzm netlist line: keyword arguments there
# go to the *photonic* model, which (correctly) rejects attributes it does not have.
CJ = 200e-15
RS = 10.0
RSH = 1e6
CPAD = 20e-15

BIT_RATE = 10e9
NUM_BITS = 8
PATTERN = '01101001'          # balanced, with both isolated and repeated bits
T_BIT = 1.0 / BIT_RATE
T_STOP = NUM_BITS * T_BIT
NUM_T = 401

#: Driver output-stage NMOS widths to sweep [um].
WIDTHS = (4.0, 8.0, 16.0, 32.0, 64.0)
#: The width the derivative is reported at.
W_NOMINAL = 16.0


def build_netlist(path: str, width_um: float) -> str:
    edges: List[Tuple[float, float]] = [(0.0, 0.0)]
    for index, bit in enumerate(PATTERN):
        level = VDD if bit == '1' else 0.0
        start = index * T_BIT
        edges.append((start, edges[-1][1]))
        edges.append((start + 10e-12, level))
    edges.append((T_STOP, edges[-1][1]))
    pwl = 'PWL (' + ' '.join(f'{t:.12g} {v:.6g}' for t, v in edges) + ')'

    # Two-stage inverter chain, fan-out 4: a W/4 pre-driver and the W output stage.
    electronic = f"""* N2 -- Level-1 CMOS driver into the level3 RC modulator load
.include {os.path.join(HERE, 'models_level1.spice')}
.option numdgt=7 delmax={T_BIT / 40:.6g} relv=1e-6 reli=1e-9 absv=1e-9 absi=1e-15
Vdd vdd 0 {VDD}
Vdata ndata 0 {pwl}
* pre-driver, W/4
mp0 npre ndata vdd vdd pmos1 w={3 * width_um / 4:.6g}u l=0.5u
mn0 npre ndata 0   0   nmos1 w={width_um / 4:.6g}u l=0.5u
* output stage: this is the W the optical eye is differentiated with respect to
mp1 nrf npre vdd vdd pmos1 w={3 * width_um:.6g}u l=0.5u
mn1 nrf npre 0   0   nmos1 w={width_um:.6g}u l=0.5u
* detector load
Rpd npd 0 1000
.tran 0 {T_STOP:.12g} {NUM_T}
.print tran v(ndata) v(npre) v(nrf) v(npd)
"""

    # act_l is 5 um, not the 1 mm of a travelling-wave segment: it is inert here (all the
    # dacoeff* loss-modulation coefficients are zero, so act_l only enters the *estimate* of the
    # optical group delay) and a 1 mm device at 2 ps sampling would trip the quasi-static check
    # for a reason that has nothing to do with what this example measures.
    photonic = f"""# MZM driven rail to rail; mod_level3 is the electrical load the driver sees.
mzm0 nin nin_unused nbar ncross nrf level3 vpi={VPI} vbias={VBIAS} er={ER_DB:g} il={IL_DB} act_l=5e-6
pd1 nbar npd level1 r0={R0}
    .mode neff=2.35
    .freq 193.5e12 193.5e12 1
    .source {math.sqrt(P_IN):.12g}@nin power=0.05w eff=0.2
    .prob nbar ncross
    .end
"""
    return write_netlist(path, electronic, photonic)


def eye_height(photocurrent: torch.Tensor, time: torch.Tensor) -> Tuple[float, float, float]:
    """``(eye, mean one level, mean zero level)`` in amps, sampled at 90% of each bit."""
    ones, zeros = [], []
    for index, bit in enumerate(PATTERN):
        target = (index + 0.9) * T_BIT
        sample = int(torch.argmin((time - target).abs()))
        (ones if bit == '1' else zeros).append(float(photocurrent[sample]))
    mean_one = sum(ones) / len(ones)
    mean_zero = sum(zeros) / len(zeros)
    return mean_one - mean_zero, mean_one, mean_zero


def rise_time(signal: torch.Tensor, time: torch.Tensor, lo: float = 0.2,
              hi: float = 0.8) -> float:
    """20-80% rise time of the first 0 -> 1 transition in the record."""
    low, high = float(signal.min()), float(signal.max())
    if high - low <= 0:
        return float('nan')
    threshold_lo = low + lo * (high - low)
    threshold_hi = low + hi * (high - low)
    start = end = None
    for k in range(len(signal) - 1):
        if start is None and float(signal[k]) <= threshold_lo < float(signal[k + 1]):
            start = float(time[k])
        if start is not None and float(signal[k]) <= threshold_hi < float(signal[k + 1]):
            end = float(time[k])
            break
    if start is None or end is None:
        return float('nan')
    return end - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()

    directory = work_dir(args, 'n2_mzm_driver')
    spice_exe = resolve_spice_exe(args)
    set_config(args, max_iter=20)

    banner('N2  Level-1 CMOS MZM driver into the level3 RC modulator load')
    tau_load = RS * CJ
    print(f"  modulator load : cj = {CJ * 1e15:g} fF, rs = {RS:g} Ohm, rsh = {RSH:g} Ohm, "
          f"cpad = {CPAD * 1e15:g} fF")
    print(f"                   intrinsic rs*cj = {tau_load * 1e12:.4f} ps "
          f"({1 / (2 * math.pi * tau_load) / 1e9:.2f} GHz)")
    print(f"  modulator      : vpi = {VPI:g} V (== the supply, so a rail-to-rail drive is a "
          f"full half wave), er = {ER_DB:g} dB, il = {IL_DB:g} dB")
    print(f"  data           : {PATTERN} at {BIT_RATE / 1e9:g} Gb/s, "
          f"{NUM_T} samples over {T_STOP * 1e9:g} ns")

    results: Dict[float, Dict[str, float]] = {}
    print('\n  -- width sweep ---------------------------------------------------------------')
    for width in WIDTHS:
        netlist = build_netlist(os.path.join(directory, f'n2_w{width:g}u.sp'), width)
        circuit = Circuit(netlist, spice_exe=spice_exe,
                          spice_wrk_dir=os.path.join(directory, 'tmp'),
                          spice_file_name='n2.sp')
        prob_e, _, param_e, param_p, _ = circuit.simulate(
            x0=torch.zeros(1, dtype=torch.float64))
        info = circuit.fixed_point_info
        photocurrent = param_e[:, 0]
        drive = param_p[:, 0]
        eye, one, zero = eye_height(photocurrent, circuit.time)
        swing = float(drive.max() - drive.min())
        extinction = 10 * math.log10(one / zero) if zero > 0 else float('inf')
        results[width] = {'eye': eye, 'one': one, 'zero': zero, 'swing': swing,
                          'extinction': extinction, 'iters': info['iters']}
        print(f"      W = {width:5.1f} um : eye = {eye * 1e6:8.4f} uA   "
              f"(1 level {one * 1e6:8.4f} uA, 0 level {zero * 1e6:7.4f} uA, ER "
              f"{extinction:6.2f} dB)   drive swing = {swing:6.4f} V of {VPI:g} V   "
              f"fixed point in {info['iters']} iterations")

    print('\n  -- d(optical) / dW -----------------------------------------------------------')
    widths = sorted(results)
    index = widths.index(W_NOMINAL)
    if 0 < index < len(widths) - 1:
        w_lo, w_hi = widths[index - 1], widths[index + 1]
        slope = ((results[w_hi]['eye'] - results[w_lo]['eye']) / (w_hi - w_lo))
        print(f"      central difference about W = {W_NOMINAL:g} um "
              f"(using {w_lo:g} and {w_hi:g} um):")
        print(f"      d(eye)/dW = {slope * 1e6:.6f} uA / um "
              f"= {slope * 1e6 / results[W_NOMINAL]['eye'] / 1e6 * 100:.4f} % of the eye per um")
        for lo, hi in zip(widths[:-1], widths[1:]):
            local = (results[hi]['eye'] - results[lo]['eye']) / (hi - lo)
            print(f"        {lo:5.1f} -> {hi:5.1f} um : {local * 1e6:10.6f} uA/um")
    print('      This is a finite difference.  The analytic path needs '
          'd(drive)/dW from the SPICE\n      back end, which exists for Xyce (.SENS) and not '
          'for HSPICE; see the module docstring.')

    if args.plot:
        _plot(directory, results)
    return 0


def _plot(directory: str, results) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    widths = sorted(results)
    plt.figure(figsize=(7, 5))
    plt.plot(widths, [results[w]['eye'] * 1e6 for w in widths], 'o-', linewidth=2)
    plt.xlabel('output stage NMOS width W (um)')
    plt.ylabel('optical eye height (uA)')
    plt.title('N2: optical eye vs driver width, into the level3 RC load')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(directory, 'n2_eye_vs_width.png')
    plt.savefig(path, dpi=120)
    print(f"      plot written to {path}")


if __name__ == '__main__':
    raise SystemExit(main())
