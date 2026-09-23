"""L.2.a -- optoelectronic oscillator whose period is set by the *electronic* filter.

What this demonstrates
----------------------
A real feedback loop, in the sense of Definition 1 of the paper (a photocurrent that reaches a
modulator drive), closed around a gain greater than one, so that the circuit has no stable
quiescent point and must oscillate::

    laser -> MZM (bar port) -> photodiode -> transimpedance R -> voltage gain A
                ^                                                       |
                |                                                       v
                +---- +quadrature offset <---- series-RLC band pass <----+

Nothing here needs optical memory: the frequency is set entirely by the electronic band-pass
filter, so the circuit sits inside SPIPE's quasi-static photonic formulation.  (The textbook
OEO, whose frequency is set by an *optical* delay line, does not -- see
``oeo_optical_delay.py``.)

Analytic prediction
-------------------
Around quadrature the MZM bar transmission is ``a^2 sin^2(dphi/2)`` with
``dphi = pi (v - vbias) / vpi``, so its small-signal slope is ``a^2 pi / (2 vpi)`` and the
open-loop gain of the ring at the filter centre frequency is

    G = A * R_load * r0 * P_in * a^2 * pi / (2 * vpi).

The series-RLC band pass is ``H(s) = (w0/Q) s / (s^2 + (w0/Q) s + w0^2)``, unity at ``w0``, so
the closed-loop characteristic equation ``1 - G H(s) = 0`` reduces to

    s^2 + (w0/Q) (1 - G) s + w0^2 = 0,

whose roots are ``s = sigma +- j w_osc`` with

    sigma  = (w0 / (2 Q)) * (G - 1)          > 0 for G > 1, i.e. the loop starts up,
    w_osc  = w0 * sqrt(1 - ((G - 1) / (2 Q))^2).

The amplitude is limited by the modulator itself, not by any added clipper: the fundamental
component of ``sin(theta_a cos wt)`` is ``2 J1(theta_a) cos wt``, so the describing-function
gain of the MZM falls as ``2 J1(theta_a) / theta_a`` and the limit cycle settles where
``G * 2 J1(theta_a) / theta_a = 1``.  Both predictions are printed and compared with the
measurement.

Why this is a genuine test of the solver
----------------------------------------
Within one SPICE run the loop is *open*: SPICE is handed the photocurrent of the previous
iterate as a PWL source.  The loop is closed only by the fixed-point iteration, whose linearised
operator has gain ``|G H(jw)|``, i.e. **greater than one** over the filter pass band.  Plain
Picard therefore diverges on exactly the Fourier modes that make the circuit oscillate; Anderson
acceleration is what makes it converge, because only a handful of modes have gain above one
(the -6 dB width of ``G H`` here is a few hundred MHz, a few bins of the record).

Run it::

    python examples/oeo_electronic.py --plot
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from _common import (add_common_arguments, banner, dominant_frequency, report_convergence,
                     report_failure, resolve_spice_exe, set_config, spice_options, work_dir,
                     write_netlist,
                     zero_crossing_frequency)

from spipe import Circuit, config
from spipe.core.core import FixedPointError

# ---------------------------------------------------------------------------- design point
VPI = 2.0             # MZM half-wave voltage [V]
VBIAS = 0.0           # MZM bias [V]; the loop adds a +vpi/2 offset to sit at quadrature
IL_DB = 3.0           # insertion loss [dB]
ER_DB = 1e9           # extinction ratio [dB] -- ideal, so the closed form is exact
P_IN = 1.0e-3         # laser power [W]
R0 = 0.9              # responsivity [A/W]
R_LOAD = 1.0e3        # transimpedance load [Ohm]

F0 = 2.0e9            # band-pass centre frequency [Hz]
Q = 8.0               # band-pass quality factor
R_BP = 50.0           # band-pass series resistance [Ohm]
LOOP_GAIN = 2.0       # open-loop gain at F0; > 1 is what makes it oscillate

T_STOP = 8.0e-9
NUM_T = 321

#: Series-RLC element values that realise ``F0`` and ``Q`` with ``R_BP``:
#: ``w0 = 1/sqrt(LC)`` and ``Q = sqrt(L/C)/R``.
OMEGA0 = 2.0 * math.pi * F0
L_BP = Q * R_BP / OMEGA0
C_BP = 1.0 / (OMEGA0 ** 2 * L_BP)

#: Small-signal slope of the optical-to-electrical path, ``dv(npd)/dv(nrf)`` at quadrature.
A_SQUARED = 10 ** (-IL_DB / 10)
SLOPE = R_LOAD * R0 * P_IN * A_SQUARED * math.pi / (2.0 * VPI)
#: Amplifier gain that realises ``LOOP_GAIN``.
A_GAIN = LOOP_GAIN / SLOPE

#: Closed-loop pole of the linearised ring.
SIGMA = OMEGA0 / (2.0 * Q) * (LOOP_GAIN - 1.0)
F_ANALYTIC = F0 * math.sqrt(max(0.0, 1.0 - ((LOOP_GAIN - 1.0) / (2.0 * Q)) ** 2))


def _bessel_j1(x: float) -> float:
    """``J1(x)`` by its power series; plenty accurate for the ``x < 10`` we need here."""
    total, term = 0.0, 0.5 * x
    for k in range(60):
        if k:
            term *= -(x * x) / (4.0 * k * (k + 1))
        total += term
        if abs(term) < 1e-18 * max(1.0, abs(total)):
            break
    return total


def describing_function_amplitude() -> float:
    """Limit-cycle drive amplitude [V] from ``G * 2 J1(theta_a) / theta_a = 1``."""
    def excess(theta: float) -> float:
        if theta == 0.0:
            return LOOP_GAIN - 1.0
        return LOOP_GAIN * 2.0 * _bessel_j1(theta) / theta - 1.0

    lo, hi = 1e-6, 3.8317  # first zero of J1: the gain is monotonically falling up to here
    if excess(lo) <= 0.0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    return theta * VPI / math.pi


def build_netlist(path: str) -> str:
    electronic = f"""* L.2.a -- optoelectronic oscillator, electronic (band-pass) frequency selection
* Reproducibility options -- see _common.spice_options() for why they matter here.
{spice_options(T_STOP, NUM_T).rstrip()}
* Photocurrent of pd1 lands on npd through the pd_level1 equivalent circuit.
Rpd npd 0 {R_LOAD:g}
* Transimpedance / voltage gain stage.
Etia ntia 0 npd 0 {A_GAIN:.12g}
* Series R-L-C band pass, output taken across the resistor: unity gain at f0.
Lbp ntia nlc {L_BP:.12g}
Cbp nlc nbp {C_BP:.12g}
Rbp nbp 0 {R_BP:g}
* Quadrature offset (+vpi/2) plus the RF drive, onto the modulator port.
Vq nq 0 {VPI / 2:g}
Edrv nrf nq nbp 0 1.0
.tran 0 {T_STOP:g} {NUM_T}
.print tran v(npd) v(nbp) v(nrf)
"""

    photonic = f"""# MZM bar port detected. At a drive of vbias plus half of vpi it sits at quadrature.
mzm0 nin nin_unused nbar ncross nrf level1 vpi={VPI} vbias={VBIAS} er={ER_DB:.0f} il={IL_DB} act_l=1e-4
pd1 nbar npd level1 r0={R0}
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
    parser.add_argument('--anderson-depth', type=int, default=12,
                        help='Anderson history length (0 = plain damped Picard, which is '
                             'expected to diverge here)')
    args = parser.parse_args()

    directory = work_dir(args, 'oeo_electronic')
    netlist = build_netlist(os.path.join(directory, 'oeo_electronic.sp'))
    # The default rtol/atol of 1e-3 is used unchanged.  Tighter than that is not meaningful
    # here: HSPICE chooses its own internal time steps, so feeding it a *different* photocurrent
    # PWL gives back a slightly different interpolated waveform even where the circuit is
    # linear.  That reproducibility floor is ~1e-3 V rms on this record (it is ~3e-4 V on the
    # feedback-free examples in test2/), and asking the iteration to go below it only buys a
    # residual that will not fall.
    set_config(args, max_iter=200, anderson_depth=args.anderson_depth)

    banner('L.2.a  optoelectronic oscillator with electronic frequency selection')
    print(f"  netlist        : {netlist}")
    print(f"  band pass      : f0 = {F0 / 1e9:.6f} GHz, Q = {Q:g}, "
          f"R = {R_BP:g} Ohm, L = {L_BP * 1e9:.6f} nH, C = {C_BP * 1e12:.6f} pF")
    print(f"  optical slope  : dv(npd)/dv(nrf) = {SLOPE:.9f} V/V at quadrature")
    print(f"  amplifier gain : A = {A_GAIN:.9f} V/V  ->  open-loop gain G = {LOOP_GAIN:g}")
    print(f"  analytic pole  : s = sigma +- j*w_osc, "
          f"sigma = {SIGMA:.6e} 1/s (growth time constant {1e9 / SIGMA:.4f} ns)")
    print(f"  analytic freq  : f_osc = {F_ANALYTIC / 1e9:.6f} GHz")
    amplitude = describing_function_amplitude()
    print(f"  analytic ampl. : describing function gives a drive amplitude of "
          f"{amplitude:.6f} V  (theta_a = {amplitude * math.pi / VPI:.6f} rad)")

    circuit = Circuit(netlist, spice_exe=resolve_spice_exe(args),
                      spice_wrk_dir=os.path.join(directory, 'tmp'),
                      spice_file_name='oeo.sp')

    print(f"\n  -- fixed-point solve (anderson_depth = {config['anderson_depth']}) -------------")
    try:
        prob_e, _, param_e, param_p, _ = circuit.simulate(
            x0=torch.full((1,), VPI / 2, dtype=torch.float64))
    except FixedPointError as error:
        report_failure(error)
        return 1
    report_convergence(circuit.fixed_point_info)

    t = circuit.time
    drive = param_p[:, 0]
    bandpass = prob_e['v(nbp)']
    photocurrent = param_e[:, 0]

    f_fft, amp_fft = dominant_frequency(bandpass, t, skip_fraction=0.45)
    f_zero = zero_crossing_frequency(bandpass, t, skip_fraction=0.45)
    tail = bandpass[int(0.45 * len(bandpass)):]
    peak_to_peak = float(tail.max() - tail.min())

    print('\n  -- measured oscillation -----------------------------------------------------')
    print(f"      record                 : 0 .. {float(t[-1]) * 1e9:.3f} ns, {len(t)} samples "
          f"({float(t[1] - t[0]) * 1e12:.3f} ps apart)")
    print(f"      frequency (FFT peak)   : {f_fft / 1e9:.6f} GHz")
    print(f"      frequency (0 crossings): {f_zero / 1e9:.6f} GHz")
    print(f"      analytic estimate      : {F_ANALYTIC / 1e9:.6f} GHz "
          f"(filter centre {F0 / 1e9:.6f} GHz)")
    print(f"      error vs analytic      : {(f_fft - F_ANALYTIC) / F_ANALYTIC * 100:+.4f} % "
          f"(FFT), {(f_zero - F_ANALYTIC) / F_ANALYTIC * 100:+.4f} % (zero crossings)")
    print(f"      FFT bin spacing        : "
          f"{1.0 / (float(t[-1]) * (1 - 0.45)) / 1e9:.6f} GHz -- the resolution floor")
    print(f"      drive swing            : {float(drive.min()):.6f} .. "
          f"{float(drive.max()):.6f} V")
    print(f"      band-pass output       : peak-to-peak {peak_to_peak:.6f} V, "
          f"fundamental amplitude {amp_fft:.6f} V")
    print(f"      analytic amplitude     : {amplitude:.6f} V "
          f"(describing function, ratio measured/analytic = {amp_fft / amplitude:.4f})")
    print(f"      photocurrent           : {float(photocurrent.min()) * 1e6:.4f} .. "
          f"{float(photocurrent.max()) * 1e6:.4f} uA")
    oscillating = peak_to_peak > 0.05 * VPI
    print(f"      oscillating            : {oscillating}")

    if args.plot:
        _plot(directory, t, drive, bandpass, photocurrent)
    return 0 if oscillating else 2


def _plot(directory: str, t, drive, bandpass, photocurrent) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t * 1e9, drive, linewidth=1.8)
    axes[0].set_ylabel('MZM drive (V)')
    axes[1].plot(t * 1e9, bandpass, linewidth=1.8, color='tab:orange')
    axes[1].set_ylabel('band-pass out (V)')
    axes[2].plot(t * 1e9, photocurrent * 1e6, linewidth=1.8, color='tab:green')
    axes[2].set_ylabel('photocurrent (uA)')
    axes[2].set_xlabel('time (ns)')
    axes[0].set_title(f'L.2.a electronic-delay OEO, f0 = {F0 / 1e9:g} GHz, G = {LOOP_GAIN:g}')
    fig.tight_layout()
    path = os.path.join(directory, 'oeo_electronic.png')
    fig.savefig(path, dpi=120)
    print(f"      plot written to {path}")


if __name__ == '__main__':
    raise SystemExit(main())
