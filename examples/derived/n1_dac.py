"""N1 -- a Level-1 7-bit R-2R DAC, calibrated against the measured sky130 reference.

Why
---
Every DAC in ``test2/`` instantiates the sky130 foundry models, which are BSIM4 (``level=54``).
That is a compact model with several hundred parameters and a hand-written implementation; it
is not something a small native circuit engine can host, so those circuits only run on a
licensed commercial simulator.  This is the same *function* -- an 8-bit-class voltage DAC
driving a modulator -- built out of ``level=1`` MOSFETs and resistors, which every SPICE-family
engine, including a native one, can evaluate.

What is measured, and against what
----------------------------------
The reference was captured from HSPICE on the real sky130 PDK, on the 7-bit subcircuit
(``test2/dac_model/8bit_DAC/7bit_DAC.sub``), and is quoted in :data:`SKY130`:

    output swing   3.27418 V
    LSB            25.781 mV
    INL / DNL      2.517 / 2.517 LSB   (endpoint method)
    monotonic      no
    1% settling    2.73 ns

Swing and LSB are matched *by construction*: an R-2R ladder with ``vrefh = 3.3 V`` and 7 bits
has a full scale of ``3.3 * 127/128 = 3.27422 V`` and an LSB of ``3.3/128 = 25.781 mV``, and
those are exactly the reference numbers, because they are properties of the reference voltage
and the bit count rather than of the topology.  Settling is matched by calibrating the output
capacitance ``cload`` of the ladder.  INL and DNL are *not* expected to match: the reference
is non-monotonic with a 2.5 LSB error at code 95, which is a property of the interpolating
architecture sky130's DAC uses (two half DACs sharing an undriven node), and an R-2R ladder
does not have that failure mode.  This example reports what it got and says so.

Run it::

    python examples/derived/n1_dac.py
    python examples/derived/n1_dac.py --calibrate     # sweep cload to hit the 2.73 ns target
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from typing import Dict, List, Sequence, Tuple

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

from _common import (add_common_arguments, banner, resolve_spice_exe, run_spice, settling_time,
                     work_dir)

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

#: Measured on HSPICE X-2025.06 with the real sky130_fd_pr BSIM4 (level=54) models, on the
#: 7-bit subcircuit.  INL/DNL are the endpoint-line figures (first and last code define the
#: straight line), which is what reproduces the reference numbers exactly.
SKY130: Dict[str, float] = {
    'nbits': 7,
    'vdd': 3.3,
    'vout_min': 1.89792e-05,
    'vout_max': 3.2742,
    'swing': 3.2741810208,
    'lsb': 0.02578095291968504,
    'inl_max_lsb': 2.516993597252719,
    'dnl_max_lsb': 2.5166235368338583,
    'monotonic': False,
    'settle_1pct_s': 2.730e-09,
}

NBITS = 7
NCODES = 1 << NBITS
VDD = 3.3
VREFH = 3.3
VREFL = 0.0
VLOGIC = 3.3          # digital swing driving the switch gates

T_CODE = 20e-9        # dwell per code in the DC-transfer staircase
T_EDGE = 100e-12      # digital edge

#: Output capacitance of the ladder, in farads.  This is the one calibrated number: it sets the
#: settling time and nothing else (the ladder's Thevenin resistance is ``rlad`` whatever the
#: code, so the pole is ``rlad * cload``).  Calibrated with ``--calibrate``.
CLOAD = 2.742538e-13
RLAD = 2.0e3


def _pwl(points: Sequence[Tuple[float, float]]) -> str:
    return 'PWL (' + ' '.join(f'{t:.12g} {v:.6g}' for t, v in points) + ')'


def _staircase_pwl(bit: int) -> str:
    """PWL waveform of one bit while the code walks 0, 1, ... , 127."""
    points: List[Tuple[float, float]] = [(0.0, VLOGIC if (0 >> bit) & 1 else 0.0)]
    for code in range(1, NCODES):
        level = VLOGIC if (code >> bit) & 1 else 0.0
        start = code * T_CODE
        points.append((start, points[-1][1]))
        points.append((start + T_EDGE, level))
    points.append((NCODES * T_CODE, points[-1][1]))
    return _pwl(points)


def _step_pwl(bit: int, t_step: float) -> str:
    """PWL waveform of one bit for a 0 -> 127 full-scale step at ``t_step``."""
    return _pwl([(0.0, 0.0), (t_step, 0.0), (t_step + T_EDGE, VLOGIC), (10 * t_step, VLOGIC)])


def _header(cload: float) -> str:
    return f"""* N1 -- Level-1 7-bit R-2R DAC characterisation
.include {os.path.join(HERE, 'models_level1.spice')}
.include {os.path.join(HERE, 'switch_l1.sub')}
.include {os.path.join(HERE, 'dac7_r2r_l1.sub')}
.option numdgt=7 relv=1e-6 reli=1e-9 absv=1e-9 absi=1e-15
Vdd vdd 0 {VDD}
Vrefh vrefh 0 {VREFH}
Vrefl vrefl 0 {VREFL}
xdac vrefh vrefl d0 d1 d2 d3 d4 d5 d6 vdd vout s7bit_DAC_l1 rlad={RLAD:g} cload={cload:.6g}
"""


def build_transfer_deck(path: str, cload: float) -> str:
    body = _header(cload)
    for bit in range(NBITS):
        body += f"Vd{bit} d{bit} 0 {_staircase_pwl(bit)}\n"
    body += f".tran {T_CODE / 10:.6g} {NCODES * T_CODE:.6g}\n"
    body += ".print tran v(vout)\n.end\n"
    with open(path, 'w') as handle:
        handle.write(body)
    return path


def build_step_deck(path: str, cload: float, t_step: float = 2e-9,
                    t_stop: float = 20e-9) -> str:
    body = _header(cload)
    for bit in range(NBITS):
        body += f"Vd{bit} d{bit} 0 {_step_pwl(bit, t_step)}\n"
    body += f".option delmax=2p\n.tran 2p {t_stop:.6g}\n"
    body += ".print tran v(vout)\n.end\n"
    with open(path, 'w') as handle:
        handle.write(body)
    return path


def sample_codes(time: Sequence[float], vout: Sequence[float]) -> List[float]:
    """The settled output of every code, read at 95% of the way through its dwell."""
    values: List[float] = []
    index = 0
    for code in range(NCODES):
        target = code * T_CODE + 0.95 * T_CODE
        while index + 1 < len(time) and time[index + 1] <= target:
            index += 1
        values.append(vout[index])
    return values


def linearity(values: Sequence[float]) -> Dict[str, float]:
    """Endpoint-line INL and DNL in LSB, plus swing, LSB and monotonicity."""
    n = len(values)
    swing = values[-1] - values[0]
    lsb = swing / (n - 1)
    inl = [(values[k] - (values[0] + k * lsb)) / lsb for k in range(n)]
    dnl = [((values[k + 1] - values[k]) / lsb - 1.0) for k in range(n - 1)]
    worst_inl = max(range(n), key=lambda k: abs(inl[k]))
    worst_dnl = max(range(n - 1), key=lambda k: abs(dnl[k]))
    return {
        'vout_min': values[0],
        'vout_max': values[-1],
        'swing': swing,
        'lsb': lsb,
        'inl_max_lsb': abs(inl[worst_inl]),
        'inl_code': worst_inl,
        'dnl_max_lsb': abs(dnl[worst_dnl]),
        'dnl_code': worst_dnl,
        'monotonic': all(values[k + 1] >= values[k] for k in range(n - 1)),
    }


def measure_settling(time: Sequence[float], vout: Sequence[float], t_step: float) -> float:
    """1% settling time of the full-scale step, measured from the edge."""
    start = next(k for k in range(len(time)) if time[k] >= t_step)
    tail_t = torch.tensor(time[start:], dtype=torch.float64)
    tail_v = torch.tensor(vout[start:], dtype=torch.float64)
    return float(settling_time(tail_v, tail_t - tail_t[0], tolerance=0.01))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    parser.add_argument('--calibrate', action='store_true',
                        help='sweep cload until the 1%% settling matches the sky130 reference')
    parser.add_argument('--cload', type=float, default=CLOAD,
                        help='output capacitance of the ladder [F]')
    parser.add_argument('--native', action='store_true',
                        help="also run the DC transfer on SPIPE's native engine "
                             "(spipe.electronic.native) and compare it with SPICE")
    args = parser.parse_args()

    directory = work_dir(args, 'n1_dac')
    spice_exe = resolve_spice_exe(args)
    cload = args.cload

    banner('N1  Level-1 7-bit R-2R DAC vs the sky130 reference')
    print(f"  ladder R = {RLAD:g} Ohm, legs 2R = {2 * RLAD:g} Ohm, cload = {cload * 1e15:.3f} fF")
    print(f"  ideal full scale = vrefh * {NCODES - 1}/{NCODES} = "
          f"{VREFH * (NCODES - 1) / NCODES:.6f} V,  ideal LSB = vrefh/{NCODES} = "
          f"{VREFH / NCODES * 1e3:.6f} mV")

    if args.calibrate:
        cload = _calibrate(directory, spice_exe, cload)
        print(f"\n  calibrated cload = {cload * 1e15:.4f} fF")

    step_deck = build_step_deck(os.path.join(directory, 'n1_step.sp'), cload)
    time, columns = run_spice(step_deck, spice_exe)
    vout = columns['v(vout)'] if 'v(vout)' in columns else columns['vout']
    settle = measure_settling(time, vout, 2e-9)

    transfer_deck = build_transfer_deck(os.path.join(directory, 'n1_transfer.sp'), cload)
    t2, c2 = run_spice(transfer_deck, spice_exe)
    v2 = c2['v(vout)'] if 'v(vout)' in c2 else c2['vout']
    codes = sample_codes(t2, v2)
    metrics = linearity(codes)

    with open(os.path.join(directory, 'n1_dc_transfer.csv'), 'w') as handle:
        handle.write('code,vout\n')
        for code, value in enumerate(codes):
            handle.write(f'{code},{value:.9g}\n')

    print('\n  -- measured ------------------------------------------------------------------')
    rows = [
        ('output swing [V]', metrics['swing'], SKY130['swing']),
        ('LSB [mV]', metrics['lsb'] * 1e3, SKY130['lsb'] * 1e3),
        ('vout at code 0 [V]', metrics['vout_min'], SKY130['vout_min']),
        ('vout at code 127 [V]', metrics['vout_max'], SKY130['vout_max']),
        ('INL, endpoint [LSB]', metrics['inl_max_lsb'], SKY130['inl_max_lsb']),
        ('DNL, endpoint [LSB]', metrics['dnl_max_lsb'], SKY130['dnl_max_lsb']),
        ('1% settling [ns]', settle * 1e9, SKY130['settle_1pct_s'] * 1e9),
    ]
    print(f"      {'metric':24s} {'this (level=1)':>16s} {'sky130 (BSIM4)':>16s} {'ratio':>10s}")
    for name, got, ref in rows:
        ratio = got / ref if ref else float('nan')
        print(f"      {name:24s} {got:16.6f} {ref:16.6f} {ratio:10.4f}")
    print(f"      {'monotonic':24s} {str(metrics['monotonic']):>16s} "
          f"{str(SKY130['monotonic']):>16s}")
    print(f"      worst INL at code {metrics['inl_code']}, worst DNL at code "
          f"{metrics['dnl_code']} (sky130: both at code 95)")
    print(f"\n      DC transfer written to {os.path.join(directory, 'n1_dc_transfer.csv')}")
    print('      Note: INL/DNL are NOT expected to agree.  The sky130 part is a recursive '
          'interpolating\n      converter whose two halves share an undriven node, which is '
          'where its 2.5 LSB error at\n      code 95 and its non-monotonicity come from; an '
          'R-2R ladder has no such node.')

    if args.native:
        _native_transfer(directory, cload, codes)
    return 0


def _native_deck(cload: float, code: int) -> str:
    """The same circuit as a flat deck with DC bit sources, for the native engine.

    ``.include`` is expanded by hand because the point of the exercise is the *devices*, not
    the file handling: every card below is a resistor, a capacitor, a voltage source, a
    subcircuit call or a ``level=1`` MOSFET.
    """
    body = []
    for line in _header(cload).splitlines():
        if line.strip().lower().startswith('.include'):
            with open(line.split(None, 1)[1].strip()) as handle:
                body.append(handle.read())
        else:
            body.append(line)
    for bit in range(NBITS):
        body.append(f"Vd{bit} d{bit} 0 {VLOGIC if (code >> bit) & 1 else 0.0}")
    body.append('.end')
    return '\n'.join(body)


def _native_transfer(directory: str, cload: float, spice_codes: Sequence[float]) -> None:
    """Run the 128-code DC transfer on ``spipe.electronic.native`` and compare with SPICE."""
    print('\n  -- native engine (spipe.electronic.native) -----------------------------------')
    try:
        from spipe.electronic.native import Netlist
    except Exception as error:                      # pragma: no cover - depends on the checkout
        print(f"      not available in this checkout ({type(error).__name__}: {error})")
        print('      The Level-1 models and the R-2R ladder are exactly the device set that '
              'engine\n      supports, so this is a packaging question, not a circuit one.')
        return

    values: List[float] = []
    for code in range(NCODES):
        circuit = Netlist.from_string(_native_deck(cload, code))
        operating_point = circuit.op()
        values.append(float(operating_point['vout']))

    metrics = linearity(values)
    worst = max(abs(a - b) for a, b in zip(values, spice_codes))
    print(f"      128 operating points solved natively (no SPICE, no license)")
    print(f"      output swing [V]     {metrics['swing']:12.6f}   "
          f"(sky130 {SKY130['swing']:.6f})")
    print(f"      LSB [mV]             {metrics['lsb'] * 1e3:12.6f}   "
          f"(sky130 {SKY130['lsb'] * 1e3:.6f})")
    print(f"      INL / DNL [LSB]      {metrics['inl_max_lsb']:12.6f} / "
          f"{metrics['dnl_max_lsb']:.6f}")
    print(f"      monotonic            {str(metrics['monotonic']):>12s}")
    print(f"      max |native - SPICE| over the 128 codes = {worst * 1e3:.4f} mV "
          f"= {worst / metrics['lsb']:.4f} LSB")
    with open(os.path.join(directory, 'n1_dc_transfer_native.csv'), 'w') as handle:
        handle.write('code,vout\n')
        for code, value in enumerate(values):
            handle.write(f'{code},{value:.9g}\n')


def _calibrate(directory: str, spice_exe: str, start: float) -> float:
    """Bisect ``cload`` so that the 1% settling time matches the sky130 reference."""
    target = SKY130['settle_1pct_s']

    def settle_for(cload: float) -> float:
        deck = build_step_deck(os.path.join(directory, 'n1_cal.sp'), cload)
        time, columns = run_spice(deck, spice_exe)
        vout = columns['v(vout)'] if 'v(vout)' in columns else columns['vout']
        return measure_settling(time, vout, 2e-9)

    lo, hi = 1e-15, 2e-12
    print(f"      calibrating cload for a {target * 1e9:.3f} ns 1% settling ...")
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        got = settle_for(mid)
        print(f"        cload = {mid * 1e15:9.4f} fF  ->  settling = {got * 1e9:8.4f} ns")
        if got > target:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


if __name__ == '__main__':
    raise SystemExit(main())
