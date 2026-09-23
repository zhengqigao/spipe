"""L.2.d -- optical-delay OEO.  This example deliberately **refuses to simulate**.

Why there is no number at the end of this script
------------------------------------------------
A textbook optoelectronic oscillator gets its frequency from an **optical** delay line: light
goes round a long fibre or waveguide, and the loop oscillates on whichever cavity mode

    f_m = m / tau_optical ,   m = 1, 2, 3, ...

falls inside the RF filter's pass band.  The delay *is* the frequency-determining element.

SPIPE solves the photonic network in **steady state at every time sample** (the paper's
Assumption 1, and what makes ``Photonic.simulate`` a per-sample linear solve).  A steady-state
scattering matrix has no memory: ``S(t_k)`` depends on the drive at ``t_k`` and on nothing
else.  A waveguide of length ``L`` contributes the phase ``exp(-j beta L)`` at the optical
carrier -- the *carrier* phase, correct and useful -- but it cannot delay the modulation
envelope, because there is no envelope in the formulation to delay.  The optical delay is
therefore not merely *inaccurate* here, it is **absent**: the model of this circuit is a
delay-free loop, whose oscillation frequency is set by whatever electronics happen to be in
it, and that number would have nothing to do with the device.

So this example measures the delay, shows how far it is outside the quasi-static regime using
the machinery SPIPE already has (:meth:`~spipe.photonic.photonic.Photonic.max_group_delay` and
:meth:`~spipe.photonic.photonic.Photonic.check_quasistatic`), prints the resulting warning
verbatim, and stops.  It does not run the fixed-point solve, and it does not report an
oscillation frequency.

What to use instead
-------------------
``mode='envelope'``: an envelope (slowly-varying-amplitude) formulation, in which each optical
path carries its own group delay and the photonic network becomes a delay system rather than a
memoryless one.  That mode is being added to SPIPE separately; it is what this circuit needs.
This example is written so that it becomes the *correct* netlist for that mode without change --
only the solver underneath it has to be different.

Run it::

    python examples/oeo_optical_delay.py
"""

from __future__ import annotations

import argparse
import math
import os
import warnings

from _common import (add_common_arguments, banner, resolve_spice_exe, work_dir, write_netlist)

from spipe import Circuit, config
from spipe.photonic.func import FreeLightSpeed

# ---------------------------------------------------------------------------- design point
VPI = 2.0
IL_DB = 3.0
ER_DB = 1e9
P_IN = 1.0e-3
R0 = 0.9
R_LOAD = 1.0e3
NEFF = 2.35
NG = 2.35

#: Length of the optical delay line [m].  2 m of waveguide at ng = 2.35 is 15.7 ns, i.e. a
#: 63.8 MHz mode spacing -- a small, on-chip version of the kilometres of fibre a real OEO uses.
DELAY_LENGTH = 2.0

F0 = 2.0e9            # RF band-pass centre frequency [Hz]
Q = 8.0
R_BP = 50.0

T_STOP = 8.0e-9
NUM_T = 321

OMEGA0 = 2.0 * math.pi * F0
L_BP = Q * R_BP / OMEGA0
C_BP = 1.0 / (OMEGA0 ** 2 * L_BP)


def build_netlist(path: str) -> str:
    electronic = f"""* L.2.d -- optical-delay OEO.  The electronics are only the RF filter and the gain;
* the frequency-determining element is the optical delay line in the photonic section.
Rpd npd 0 {R_LOAD:g}
Etia ntia 0 npd 0 10.0
Lbp ntia nlc {L_BP:.12g}
Cbp nlc nbp {C_BP:.12g}
Rbp nbp 0 {R_BP:g}
Vq nq 0 {VPI / 2:g}
Edrv nrf nq nbp 0 1.0
.tran 0 {T_STOP:g} {NUM_T}
.print tran v(npd) v(nbp) v(nrf)
"""

    photonic = f"""# The optical delay line: this is the element that sets the oscillation frequency,
# and it is the element the quasi-static photonic solve cannot represent.
mzm0 nin nin_unused nbar ncross nrf level1 vpi={VPI} vbias=0.0 er={ER_DB:.0f} il={IL_DB} act_l=1e-4
wg1 nbar ndelay l={DELAY_LENGTH:g} alpha=0.98
pd1 ndelay npd level1 r0={R0}
    .mode neff={NEFF} ng={NG}
    .freq 193.5e12 193.5e12 1
    .source {math.sqrt(P_IN):.12g}@nin power=0.05w eff=0.2
    .prob nbar ndelay
    .end
"""
    return write_netlist(path, electronic, photonic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()

    directory = work_dir(args, 'oeo_optical_delay')
    netlist = build_netlist(os.path.join(directory, 'oeo_optical_delay.sp'))

    banner('L.2.d  optical-delay OEO  --  REFUSED, not simulated')
    print(f"  netlist : {netlist}")

    # Building the Circuit is what runs check_quasistatic(); capture its warning verbatim.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        circuit = Circuit(netlist, spice_exe=resolve_spice_exe(args),
                          spice_wrk_dir=os.path.join(directory, 'tmp'),
                          spice_file_name='oeo_optical_delay.sp')

    dt = float(circuit.time[1] - circuit.time[0])
    tau = circuit.p_circuit.max_group_delay()
    expected = NG * DELAY_LENGTH / FreeLightSpeed

    print('\n  -- what the delay line actually is ------------------------------------------')
    print(f"      delay line             : {DELAY_LENGTH:g} m of waveguide at ng = {NG:g}")
    print(f"      group delay ng*L/c     : {expected * 1e9:.6f} ns")
    print(f"      Photonic.max_group_delay(): {tau * 1e9:.6f} ns")
    print(f"      transient time step dt : {dt * 1e12:.4f} ps")
    print(f"      max_group_delay / dt   : {tau / dt:.4f}      "
          f"(quasi-static needs << 1; SPIPE warns above 0.1)")
    print(f"      cavity mode spacing 1/tau : {1.0 / expected / 1e6:.4f} MHz")
    print(f"      RF filter pass band    : {F0 / 1e9:g} GHz, Q = {Q:g}, "
          f"BW = {F0 / Q / 1e6:.3f} MHz")
    print(f"      modes inside that band : "
          f"{max(1, int(round((F0 / Q) * expected)))} "
          f"-- a real OEO picks one of these, and which one is a property of the delay")

    print('\n  -- the warning SPIPE emits (verbatim) ---------------------------------------')
    quasi = [w for w in captured if 'Quasi-static' in str(w.message)]
    if not quasi:
        print('      (none -- check spipe.config["quasistatic_check"])')
    for warning in quasi:
        print(f"      {warning.category.__name__}: {warning.message}")

    import inspect
    signature = inspect.signature(circuit.p_circuit.simulate)
    has_envelope = 'mode' in signature.parameters
    print(f"\n      Photonic.simulate() accepts mode=: {has_envelope}"
          + ("  -- so the envelope formulation this circuit needs is available in this "
             "checkout" if has_envelope else
             "  -- the envelope formulation this circuit needs is not in this checkout yet"))

    print('\n  -- REFUSAL -------------------------------------------------------------------')
    refusal = (
        "SPIPE cannot simulate an optical-delay optoelectronic oscillator, and will not "
        "pretend to.\n"
        "      The oscillation frequency of this circuit is set by the OPTICAL loop delay "
        f"({expected * 1e9:.3f} ns\n"
        f"      here, a mode spacing of {1.0 / expected / 1e6:.3f} MHz).  Under the "
        "quasi-static assumption the\n"
        "      photonic network is solved in steady state at every time sample, so it has "
        "ZERO memory:\n"
        "      the delay line contributes an optical carrier phase but no envelope delay at "
        "all.  The\n"
        "      frequency-selective element of this oscillator is therefore not approximated "
        "badly, it is\n"
        "      structurally absent from the model, and any oscillation SPIPE reported for this "
        "netlist\n"
        "      would be a property of the electronics alone.\n"
        "      Use mode='envelope' instead: the envelope (slowly-varying-amplitude) "
        "formulation gives\n"
        "      every optical path its own group delay, which is exactly what this circuit "
        "needs.  The\n"
        "      netlist above is already the right netlist for that mode.")
    print(f"      {refusal}")
    print('\n      No simulation was run.  No oscillation frequency is reported.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
