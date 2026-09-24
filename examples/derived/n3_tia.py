"""N3 -- a real transimpedance amplifier, replacing the ideal 70 dB gain block of ``pd_level2``.

What is wrong with ``pd_level2``
--------------------------------
The shipped detector front end is::

    .subckt pd_level2 n ground ...
    Ipd cathode anode           <- the photocurrent
    d1 anode cathode diode
    r1 anode cathode 10k
    c1 anode cathode 'Cj0*(1+V(anode,cathode)/Vj)**(-M)'
    vneg anode ground -1.0
    rf cathode n 10k
    cf cathode n 2fF
    E n ground ground cathode 3162      <- "70 dB gain ideal Opamp"
    .ends pd_level2

``E`` is a voltage-controlled voltage source with a gain of 3162 and *no bandwidth, no supply
rails, no input capacitance, no noise and no power consumption*.  It cannot clip, it cannot
slew, it has infinite output drive, and it costs nothing in the power report.  Everything that
makes a receiver front end hard is therefore absent, and the closed-loop transimpedance is
simply ``rf`` for all frequencies.

What ``pd_level4`` is
---------------------
The same shunt-feedback topology, with the ideal block replaced by a real single-stage
common-source amplifier plus a source-follower output buffer, in level=1 devices::

    photodiode --+-- Rf --+
                 |        |
              cathode   nout -- source follower -- n
                 |        |
                 +-- M1 --+     (common source, resistor loaded)

so the open-loop gain is finite, ``A = gm * rd``, and the closed-loop transimpedance is

    Rt = Rf * A / (1 + A),

which is *measurably* less than ``Rf``; the bandwidth is set by the input pole
``Rf * Cin / (1 + A)`` and by the load; and the amplifier draws real current from a real
supply, so it shows up in the power report.

This example registers ``pd_level4`` with :func:`spipe.electronic_register`, runs the same
small optical link with the old and the new front end, and reports the transimpedance, the step
response and the supply current of each.

Run it::

    python examples/derived/n3_tia.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Tuple

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

# `_common` lives in examples/, one directory up from this one, and is NOT a package.
# Without this the script only imports when the current directory happens to be
# examples/ -- so `python examples/derived/n1_dac.py` from the repo root, which is what
# examples/README.md tells you to run, died on ModuleNotFoundError.
_examples_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _examples_dir not in _sys.path:
    _sys.path.insert(0, _examples_dir)

import torch

from _common import (add_common_arguments, banner, report_convergence, resolve_spice_exe,
                     set_config, work_dir, write_netlist)

from spipe import Circuit, config, electronic_register, electronic_reset

HERE = os.path.dirname(os.path.abspath(__file__))

VPI = 3.3
#: 0.33 mW, i.e. a peak photocurrent of ~150 uA.  Chosen so that the *real* front end stays in
#: its linear range: at a transimpedance of ~10 kOhm that is a 1.5 V output swing, which fits
#: between the 1 V quiescent point and the 3.3 V rail.  The ideal ``pd_level2`` block has no
#: rails at all and would happily report 4.5 V from a 1 mW input, which is precisely the kind of
#: answer this example exists to stop anyone believing.
P_IN = 3.3e-4
R0 = 0.9
IL_DB = 3.0
ER_DB = 30.0

RF = 11.6e3        # feedback resistor [Ohm]
RD = 2.0e3         # common-source load [Ohm]
W_CS = 40.0        # common-source device width [um]
L_CS = 0.5         # device length [um]

T_STOP = 4.0e-9
NUM_T = 201
T_EDGE = 2.0e-9    # when the optical input switches

#: The level=1 TIA front end.  ``Ipd`` must be the first element (the electronic netlist writer
#: finds the line starting with ``Ipd`` and turns it into the PWL photocurrent source), and the
#: subcircuit takes its own ``.model`` cards so that it is self-contained.
PD_LEVEL4 = f""".subckt pd_level4 n ground rf={RF:g} rd={RD:g} wcs={W_CS:g} lcs={L_CS:g}
Ipd cathode anode
d1 anode cathode diode4
.model diode4 D(IS=1e-12 N=1.0 CJO=50f VJ=0.7 M=0.5)
vneg anode ground -1.0
vsup vdd4 ground 3.3
* common-source gain stage
m1 nout cathode ground ground nmos4 w='wcs*1u' l='lcs*1u'
rdl vdd4 nout 'rd'
* shunt feedback
rfb cathode nout 'rf'
cfb cathode nout 2f
* source-follower output buffer
m2 vdd4 nout n ground nmos4 w=20u l='lcs*1u'
rbias n ground 20k
.model nmos4 nmos level=1 vto=0.50 kp=120u gamma=0.40 phi=0.70 lambda=0.05
+ tox=14n cgso=2e-10 cgdo=2e-10 cgbo=1e-10 is=1e-15
.ends pd_level4
"""


def build_netlist(path: str, pd_model: str) -> str:
    electronic = f"""* N3 -- detector front end comparison
.option numdgt=7 delmax={T_STOP / NUM_T / 4:.6g} relv=1e-6 reli=1e-9 absv=1e-9 absi=1e-15
* The modulator is driven by an external step: this link is feedback free on purpose, so that
* the only thing being compared is the front end.
Vdrv nrf 0 PWL (0 0 {T_EDGE:.12g} 0 {T_EDGE + 20e-12:.12g} {VPI:g} {T_STOP:.12g} {VPI:g})
.tran 0 {T_STOP:.12g} {NUM_T}
.print tran v(nrf) v(npd)
"""

    photonic = f"""# One MZM, one detector: the front end is the only thing that changes.
mzm0 nin nin_unused nbar ncross nrf level1 vpi={VPI} vbias=0.0 er={ER_DB:g} il={IL_DB} act_l=5e-6
pd1 nbar npd {pd_model} r0={R0}
    .mode neff=2.35
    .freq 193.5e12 193.5e12 1
    .source {math.sqrt(P_IN):.12g}@nin power=0.05w eff=0.2
    .prob nbar ncross
    .end
"""
    return write_netlist(path, electronic, photonic)


def characterise(circuit_dir: str, spice_exe: str, pd_model: str,
                 name: str) -> Dict[str, float]:
    netlist = build_netlist(os.path.join(circuit_dir, f'n3_{pd_model}.sp'), pd_model)
    circuit = Circuit(netlist, spice_exe=spice_exe,
                      spice_wrk_dir=os.path.join(circuit_dir, 'tmp'),
                      spice_file_name=f'n3_{pd_model}.sp',
                      power_node=[])
    prob_e, _, param_e, _, power = circuit.simulate(x0=torch.zeros(1, dtype=torch.float64))
    info = circuit.fixed_point_info

    time = circuit.time
    photocurrent = param_e[:, 0]
    output = prob_e['v(npd)']

    before = int(0.9 * T_EDGE / float(time[-1]) * (len(time) - 1))
    after = len(time) - 1
    d_current = float(photocurrent[after] - photocurrent[before])
    d_voltage = float(output[after] - output[before])
    transimpedance = d_voltage / d_current if d_current != 0 else float('nan')

    # 10-90% rise time of the front-end output after the optical step.
    low, high = float(output[before]), float(output[after])
    span = high - low
    t10 = t90 = float('nan')
    for k in range(before, after):
        if math.isnan(t10) and (float(output[k]) - low) >= 0.1 * span:
            t10 = float(time[k])
        if math.isnan(t90) and (float(output[k]) - low) >= 0.9 * span:
            t90 = float(time[k])
            break
    rise = t90 - t10 if not (math.isnan(t10) or math.isnan(t90)) else float('nan')

    return {
        'name': name,
        'iters': info['iters'],
        'i_dark': float(photocurrent[before]),
        'i_lit': float(photocurrent[after]),
        'v_dark': float(output[before]),
        'v_lit': float(output[after]),
        'transimpedance': transimpedance,
        'rise_10_90': rise,
        'pd_power': float(power['pd_equiv'][after]) if power.get('pd_equiv') is not None
        else float('nan'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()

    directory = work_dir(args, 'n3_tia')
    spice_exe = resolve_spice_exe(args)
    set_config(args, max_iter=20)

    electronic_reset()
    electronic_register('pd', 'level4', PD_LEVEL4)
    with open(os.path.join(directory, 'pd_level4.sub'), 'w') as handle:
        handle.write(PD_LEVEL4)

    banner('N3  real TIA (pd_level4) vs the ideal 70 dB block (pd_level2)')
    gm = math.sqrt(2 * 120e-6 * (W_CS / L_CS) * 0.5e-3)
    open_loop = gm * RD
    print(f"  common source  : W/L = {W_CS:g}/{L_CS:g} um, rd = {RD:g} Ohm")
    print(f"                   gm ~ {gm * 1e3:.4f} mS at 0.5 mA  ->  open-loop gain A ~ "
          f"{open_loop:.3f}")
    print(f"  feedback       : rf = {RF:g} Ohm  ->  closed-loop Rt ~ rf*A/(1+A) = "
          f"{RF * open_loop / (1 + open_loop) / 1e3:.4f} kOhm")
    print(f"  pd_level2      : rf = 10 kOhm behind an ideal E source of gain 3162 "
          f"(70 dB), Rt = 10 kOhm at every frequency")

    rows = []
    for pd_model, label in (('level2', 'pd_level2 (ideal 70 dB block)'),
                            ('level4', 'pd_level4 (real common-source TIA)')):
        print(f"\n  -- {label} ---------------------------------------")
        result = characterise(directory, spice_exe, pd_model, label)
        rows.append(result)
        print(f"      fixed point converged in {result['iters']} iterations")
        print(f"      photocurrent  {result['i_dark'] * 1e6:10.4f} -> "
              f"{result['i_lit'] * 1e6:10.4f} uA")
        print(f"      output node   {result['v_dark']:10.6f} -> {result['v_lit']:10.6f} V")
        print(f"      transimpedance dV/dI = {result['transimpedance'] / 1e3:10.4f} kOhm")
        print(f"      10-90% rise   {result['rise_10_90'] * 1e12:10.3f} ps")

    print('\n  -- comparison ----------------------------------------------------------------')
    print(f"      {'front end':38s} {'Rt [kOhm]':>12s} {'rise [ps]':>12s} {'iters':>7s}")
    for row in rows:
        print(f"      {row['name']:38s} {row['transimpedance'] / 1e3:12.4f} "
              f"{row['rise_10_90'] * 1e12:12.3f} {row['iters']:7d}")
    if len(rows) == 2 and rows[0]['transimpedance']:
        ratio = rows[1]['transimpedance'] / rows[0]['transimpedance']
        print(f"      real / ideal transimpedance ratio = {ratio:.4f}")
    print('      The real front end is slower and lower gain than the ideal one, which is the '
          'point:\n      pd_level2 hides the gain-bandwidth trade-off that sizes a receiver.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
