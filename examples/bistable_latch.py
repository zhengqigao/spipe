"""L.2.b -- optical bistability: an electro-optic latch with two stable fixed points.

What this demonstrates
----------------------
This is the case that exposes **initialisation sensitivity**.  The circuit is a Mach-Zehnder
modulator whose *bar* output is detected and fed back, with positive gain, to its own RF port:

    laser -> MZM (bar port) -> photodetector -> transimpedance load -> +gain -> MZM RF port

The MZM bar transmission ``sin^2(pi*v/(2*vpi))`` is an S-shaped function of the drive, so the
round-trip map

    g(v) = v_offset + C * sin^2(pi * v / (2 * vpi)),     C = A * R_load * r0 * P_in * 10^(-il/10)

crosses the identity **three** times.  Two of those crossings have ``|g'| < 1`` and are stable;
the middle one has ``g' > 1`` and is not.  The circuit therefore has *two* answers, and which
one a fixed-point solver reports is a property of the initial guess, not of the circuit.

That is the whole point: a solver that silently returned "the" answer here would be lying.
:func:`spipe.core.core.solve_fixed_point` converges to one of the two states and this example
reports *which*, and then establishes multiplicity directly, by multi-starting the solve from a
sweep of initial guesses and showing that the set of answers has more than one element.

Feedback, in the sense of Definition 1 of the paper (a photocurrent that reaches a modulator
drive), is present: ``I_pd`` -> ``v(npd)`` -> ``v(nrf)`` -> the drive of ``mzm0``.

Run it::

    python examples/bistable_latch.py                 # hspice
    python examples/bistable_latch.py --sim xyce
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch

from _common import (add_common_arguments, banner, report_convergence, report_failure,
                     resolve_spice_exe, set_config, spice_options, work_dir, write_netlist)

from spipe import Circuit, config
from spipe.core.core import FixedPointError

# ---------------------------------------------------------------------------- design point
#
# Everything below is chosen so that the round-trip map has three fixed points that are all
# comfortably conditioned: the two stable ones have |g'| = 0.32 and 0.61, far from the fold, so
# the demonstration does not depend on hitting a tangency.

VPI = 2.0            # MZM half-wave voltage [V]
VBIAS = 0.0          # MZM bias [V]; the drive is measured from here
IL_DB = 3.0          # MZM insertion loss [dB]
ER_DB = 1e9          # extinction ratio [dB]; huge, i.e. ideal, so the closed form below is exact
P_IN = 1.0e-3        # laser power into the MZM [W]
R0 = 0.9             # detector responsivity [A/W]
R_LOAD = 1.0e3       # transimpedance load [Ohm]
V_OFFSET = 0.1       # amplifier output offset [V] -- lifts the low state off zero
LOOP_C = 2.2         # the round-trip gain constant C of the map above [V]

#: Amplifier voltage gain that realises ``LOOP_C``.
A_GAIN = LOOP_C / (R_LOAD * R0 * P_IN * 10 ** (-IL_DB / 10))

T_STOP = 20e-9
NUM_T = 51

#: Constant initial guesses (in volts, on the MZM RF port) used to multi-start the solve.
#: 0.65 and 0.75 straddle the unstable root at 0.6989, i.e. the basin boundary.
INITIAL_GUESSES = (0.0, 0.4, 0.65, 0.75, 1.5, 3.0)

#: Two states are called the same state when they agree to this many volts.  Far looser than
#: the solver tolerance (1e-5 V below) and far tighter than the 2.1 V that separates the two
#: latch states, so the grouping is not a judgement call.
STATE_TOLERANCE = 1e-2


def transfer(v: float) -> float:
    """The exact round-trip map ``g(v)`` of this circuit, in volts."""
    return V_OFFSET + LOOP_C * math.sin(math.pi * (v - VBIAS) / (2.0 * VPI)) ** 2


def analytic_fixed_points(lo: float = -0.5, hi: float = 4.0,
                          samples: int = 200001) -> List[Tuple[float, float]]:
    """Every root of ``g(v) - v`` on ``[lo, hi]``, as ``(v*, g'(v*))`` pairs.

    Bisection on a dense sign scan: the map is smooth and one dimensional, so this finds all of
    them, and ``g'`` at each root is what decides stability (``|g'| < 1`` is stable).
    """
    def residual(v: float) -> float:
        return transfer(v) - v

    roots: List[Tuple[float, float]] = []
    previous_v, previous_r = lo, residual(lo)
    for index in range(1, samples):
        v = lo + (hi - lo) * index / (samples - 1)
        r = residual(v)
        if previous_r == 0.0 or previous_r * r < 0.0:
            a, b = previous_v, v
            for _ in range(200):
                mid = 0.5 * (a + b)
                if residual(a) * residual(mid) <= 0.0:
                    b = mid
                else:
                    a = mid
            root = 0.5 * (a + b)
            step = 1e-7
            slope = (transfer(root + step) - transfer(root - step)) / (2 * step)
            roots.append((root, slope))
        previous_v, previous_r = v, r
    return roots


def build_netlist(path: str) -> str:
    electronic = f"""* L.2.b -- optical bistability / electro-optic latch
{spice_options(T_STOP, NUM_T).rstrip()}
* The photocurrent of pd1 is injected at node npd by the pd_level1 equivalent circuit.
Rload npd 0 {R_LOAD}
* Amplifier: v(nrf) = V_OFFSET + A_GAIN * v(npd).  Positive gain == positive feedback.
Vofs nofs 0 {V_OFFSET}
Eamp nrf nofs npd 0 {A_GAIN:.12g}
.tran 0 {T_STOP} {NUM_T}
.print tran v(npd) v(nrf)
"""

    photonic = f"""# MZM bar output detected; the bar port is an S curve in the drive.
mzm0 nin nin_unused nbar ncross nrf level1 vpi={VPI} vbias={VBIAS} er={ER_DB:.0f} il={IL_DB} act_l=1e-3
pd1 nbar npd level1 r0={R0}
    .mode neff=2.35
    .freq 193.5e12 193.5e12 1
    .source {math.sqrt(P_IN):.12g}@nin power=0.05w eff=0.2
    .prob nbar ncross
    .end
"""
    return write_netlist(path, electronic, photonic)


def state_of(drive: torch.Tensor) -> float:
    """The latch state of a converged run: the median RF drive over the settled part of the
    record.

    The median, and not the mean, because sample 0 is special: HSPICE is started with ``uic``
    so every node begins at zero volts, and the very first sample therefore reports the
    amplifier offset alone rather than a latch state.  Everything from ~1 ns onwards is the
    steady state.
    """
    settled = drive[max(1, len(drive) // 10):, 0]
    return float(settled.median())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser)
    args = parser.parse_args()

    directory = work_dir(args, 'bistable_latch')
    netlist = build_netlist(os.path.join(directory, 'bistable_latch.sp'))

    # Tight tolerances so the two states come out crisp enough to compare with the closed form;
    # the *semantics* of the criterion (relative + absolute, rms per entry) are untouched.
    set_config(args, max_iter=120, rtol=1e-5, atol=1e-5)

    banner('L.2.b  optical bistability / electro-optic latch')
    print(f"  netlist   : {netlist}")
    print(f"  loop map  : g(v) = {V_OFFSET} + {LOOP_C} * sin^2(pi*v/(2*{VPI}))   [volts]")
    print(f"  amplifier : A = {A_GAIN:.6g} V/V into R_load = {R_LOAD:g} Ohm, "
          f"r0 = {R0} A/W, P_in = {P_IN * 1e3:g} mW, IL = {IL_DB} dB")

    print('\n  -- analytic fixed points of the round-trip map ------------------------------')
    roots = analytic_fixed_points()
    stable = []
    for root, slope in roots:
        kind = 'STABLE  ' if abs(slope) < 1.0 else 'unstable'
        print(f"      v* = {root:12.9f} V    g'(v*) = {slope:+8.4f}    {kind}")
        if abs(slope) < 1.0:
            stable.append(root)
    print(f"      -> the map has {len(roots)} fixed points, {len(stable)} of them stable.")

    spice_exe = resolve_spice_exe(args)

    def solve(guess: float, depth: int):
        """One full SPIPE run started from the constant drive ``guess``."""
        config['anderson_depth'] = depth
        circuit = Circuit(netlist, spice_exe=spice_exe,
                          spice_wrk_dir=os.path.join(directory, 'tmp'),
                          spice_file_name='bistable.sp')
        x0 = torch.full((1,), float(guess), dtype=torch.float64)
        _, _, param_e, param_p, _ = circuit.simulate(x0=x0)
        return state_of(param_p), float(param_e[len(param_e) // 2, 0]), circuit.fixed_point_info

    print('\n  -- basin scan: undamped Picard (anderson_depth = 0) --------------------------')
    print('     Plain relaxation follows the *physical* dynamics of the loop, so it can only '
          'land\n     on a stable state, and the basin boundary is the unstable root.')
    results: Dict[float, Tuple[float, Dict]] = {}
    for guess in INITIAL_GUESSES:
        try:
            state, photocurrent, info = solve(guess, depth=0)
        except FixedPointError as error:
            print(f"\n  x0 = {guess:5.2f} V")
            report_failure(error)
            continue
        results[guess] = (state, info)
        print(f"\n  x0 = {guess:5.2f} V  ->  v_rf = {state:.9f} V   "
              f"I_pd = {photocurrent * 1e6:.4f} uA")
        report_convergence(info, label=f'   x0={guess:.2f}')

    print('\n  -- multiplicity -------------------------------------------------------------')
    distinct: List[float] = []
    for state, _ in results.values():
        if not any(abs(state - existing) < STATE_TOLERANCE for existing in distinct):
            distinct.append(state)
    distinct.sort()
    print(f"      {len(results)} initial guesses -> {len(distinct)} distinct converged states:")
    for state in distinct:
        which = [f'{g:g}' for g, (s, _) in results.items()
                 if abs(s - state) < STATE_TOLERANCE]
        transmission = math.sin(math.pi * (state - VBIAS) / (2 * VPI)) ** 2
        nearest = min((abs(state - root), root, slope) for root, slope in roots)
        print(f"      v_rf* = {state:.9f} V   (bar transmission = {transmission:.6f}, "
              f"P_bar = {P_IN * transmission * 10 ** (-IL_DB / 10) * 1e6:.4f} uW)")
        print(f"                closed form   {nearest[1]:.9f} V   "
              f"(error {state - nearest[1]:+.3e} V, g' = {nearest[2]:+.4f}, "
              f"{'stable' if abs(nearest[2]) < 1 else 'UNSTABLE'})"
              f"   reached from x0 in {{{', '.join(which)}}}")
    if len(distinct) > 1:
        print("      => the fixed point of this circuit is NOT unique.  The solver reports the "
              "state it reached; it does not present it as the only one.")
    else:
        print("      => only one state was reached from the guesses tried.")

    print('\n  -- the same circuit with Anderson acceleration on -----------------------------')
    print('     Anderson is a *root finder*: it does not follow the physical relaxation, so on '
          'a\n     multi-valued problem it can also converge to the unstable root.  That is '
          'exactly why\n     the state reached has to be reported rather than assumed.')
    for guess in (0.0, 3.0):
        try:
            state, _, info = solve(guess, depth=5)
        except FixedPointError as error:
            print(f"\n  x0 = {guess:5.2f} V (Anderson)")
            report_failure(error)
            continue
        nearest = min((abs(state - root), root, slope) for root, slope in roots)
        print(f"\n  x0 = {guess:5.2f} V (Anderson) -> v_rf = {state:.9f} V after "
              f"{info['iters']} iterations "
              f"(nearest closed-form root {nearest[1]:.9f} V, g' = {nearest[2]:+.4f}, "
              f"{'stable' if abs(nearest[2]) < 1 else 'UNSTABLE'})")
        print(f"      residuals: "
              f"{', '.join('%.3e' % r for r in info['residuals'])}")

    if args.plot:
        _plot(directory, roots, distinct)
    return 0


def _plot(directory: str, roots, distinct) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    v = torch.linspace(-0.2, 3.5, 800, dtype=torch.float64)
    g = torch.tensor([transfer(float(x)) for x in v], dtype=torch.float64)
    plt.figure(figsize=(7, 6))
    plt.plot(v, g, linewidth=2.5, label=r'round trip $g(v)$')
    plt.plot(v, v, '--', color='grey', linewidth=1.5, label=r'$v$')
    for root, slope in roots:
        plt.plot([root], [root], 'o', markersize=10,
                 color='tab:green' if abs(slope) < 1 else 'tab:red')
    for state in distinct:
        plt.axvline(state, color='tab:blue', alpha=0.3)
    plt.xlabel('MZM drive $v_{rf}$ (V)')
    plt.ylabel('$g(v_{rf})$ (V)')
    plt.title('L.2.b bistability: green = stable fixed point, red = unstable')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(directory, 'bistable_latch.png')
    plt.savefig(path, dpi=120)
    print(f"      plot written to {path}")


if __name__ == '__main__':
    raise SystemExit(main())
