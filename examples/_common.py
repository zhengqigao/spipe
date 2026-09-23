"""Shared plumbing for the SPIPE example circuits.

Nothing in here is physics: it is the argument parsing, the SPICE executable lookup and the
little bit of reporting that every example in this directory would otherwise repeat.

The examples are deliberately *scripts*, not a library: each one writes the netlist it
simulates into its own working directory, so the netlist SPIPE actually consumed is left on
disk next to the results and can be read, diffed or fed to SPICE by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Make `import spipe` work when the examples are run straight out of a checkout.
# The package lives in src/ (PEP 517 src-layout); a pip-installed spipe wins.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, 'src')
for _cand in (_SRC, _REPO_ROOT):
    if os.path.isdir(os.path.join(_cand, 'spipe')) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

import torch  # noqa: E402  (after the sys.path fix-up on purpose)

from spipe import config  # noqa: E402
from spipe.core.core import FixedPointError  # noqa: E402

#: The SPICE executables the paper decks in ``examples/paper/`` use.  Override with
#: ``--spice-exe`` (or the ``SPIPE_SPICE_EXE`` environment variable) on any machine where they
#: live somewhere else.
DEFAULT_SPICE_EXE = {
    'hspice': os.environ.get('SPIPE_HSPICE', 'hspice') + ' ',
    'xyce': os.environ.get('SPIPE_XYCE', 'Xyce') + ' -quiet -hspice-ext all',
}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``--sim`` / ``--spice-exe`` / ``--work-dir`` / ``--max-iter`` block."""
    parser.add_argument('--sim', choices=sorted(DEFAULT_SPICE_EXE), default='hspice',
                        help='which SPICE engine to drive (default: hspice)')
    parser.add_argument('--spice-exe', default=None,
                        help='full command line of the SPICE executable; overrides --sim')
    parser.add_argument('--work-dir', default=None,
                        help='directory for the generated netlist and SPICE scratch files '
                             '(default: a per-example directory next to this script)')
    parser.add_argument('--max-iter', type=int, default=None,
                        help="override spipe.config['max_iter'] for this run")
    parser.add_argument('--plot', action='store_true',
                        help='also write a PNG of the result into the working directory')
    return parser


def resolve_spice_exe(args: argparse.Namespace) -> str:
    """The SPICE command line to hand :class:`spipe.Circuit`."""
    if args.spice_exe:
        return args.spice_exe
    from_env = os.environ.get('SPIPE_SPICE_EXE')
    if from_env:
        return from_env
    return DEFAULT_SPICE_EXE[args.sim]


def work_dir(args: argparse.Namespace, default_name: str) -> str:
    """Create and return the directory the example writes its netlist and results into."""
    path = args.work_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '_run', default_name)
    os.makedirs(path, exist_ok=True)
    return path


def write_netlist(path: str, electronic: str, photonic: str) -> str:
    """Assemble and write a SPIPE ``.sp`` file; returns ``path``."""
    with open(path, 'w') as handle:
        handle.write('.electronic\n')
        handle.write(electronic.rstrip() + '\n\n')
        handle.write('.photonic\n')
        handle.write(photonic.rstrip() + '\n')
    return path


def spice_options(t_stop: float, num_t: int) -> str:
    """SPICE options that make a transient run *repeatable* enough to iterate on.

    A fixed-point iteration over waveforms compares two SPICE runs that differ by less than a
    millivolt, so the simulator itself has to be that repeatable, and by default it is not:

    * ``numdgt`` -- HSPICE prints its tabular output with five significant digits, i.e. a
      1e-4 V quantisation on a 1 V node.  That quantisation is a floor under the residual.
    * ``delmax`` -- the transient time step is chosen adaptively from the waveforms, so handing
      SPICE a slightly different photocurrent also hands it a *different integration grid*, and
      the interpolation back onto SPIPE's grid then differs by far more than the change that
      caused it.  Pinning the step removes that feedback path entirely.
    * ``relv``/``reli``/``absv``/``absi`` -- tighter Newton tolerances, for the same reason.

    On the electronic-delay OEO these options take the residual floor from ~3e-3 V to below
    1e-3 V and the iteration count from 125 to 45.
    """
    return (f".option numdgt=7 delmax={t_stop / num_t / 4:.6g} "
            f"relv=1e-6 reli=1e-8 absv=1e-9 absi=1e-15\n")


def run_spice(deck_path: str, spice_exe: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """Run one *plain* SPICE deck (no SPIPE coupling) and return ``(time, {column: values})``.

    The derived circuits of L.3 are characterised the way a circuit designer would characterise
    them -- a DC transfer curve, a step response -- which does not involve the photonic side at
    all, so they are run directly rather than through :class:`spipe.Circuit`.  Both back ends
    are supported: HSPICE writes its ``.print`` tables into the ``.lis`` listing between ``x``
    and ``y`` markers, Xyce writes a ``.prn`` file with a one-line header.
    """
    import subprocess

    if 'xyce' in spice_exe.lower():
        subprocess.run(spice_exe.split() + [deck_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return _parse_prn(deck_path + '.prn')

    subprocess.run(spice_exe.split() + [deck_path, '-o', deck_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return _parse_lis(deck_path + '.lis')


def _parse_prn(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    with open(path) as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    header = lines[0].split()
    names = header[2:]
    time: List[float] = []
    columns: Dict[str, List[float]] = {name.lower(): [] for name in names}
    for line in lines[1:]:
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        time.append(float(fields[1]))
        for name, value in zip(names, fields[2:]):
            columns[name.lower()].append(float(value))
    return time, columns


def _parse_lis(path: str) -> Tuple[List[float], Dict[str, List[float]]]:
    """Parse the ``x`` / ``y`` delimited print tables HSPICE writes into its ``.lis``.

    Same format :class:`spipe.electronic.electronic.SimulateHspice` reads, and the same unit
    suffix handling (``1.0004`` but also ``20.9861m`` and ``449.146u``), which is why it borrows
    ``spipe.utils.convert``.
    """
    from spipe.utils import convert

    unit_map = {'time': 's', 'current': 'A', 'voltage': 'v', 'param': 'w'}
    time: List[float] = []
    columns: Dict[str, List[float]] = {}
    order: List[str] = []

    with open(path) as handle:
        lines = handle.read().splitlines()

    index, parsing, units, names = 0, False, None, None
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == 'x':
            parsing, units, names = True, None, None
            index += 1
            continue
        if stripped == 'y':
            parsing = False
            index += 1
            continue
        if parsing and units is None and stripped.startswith('time'):
            units = [unit_map[key] for key in stripped.split()]
            names = lines[index + 1].split()
            for name in names:
                if name not in columns:
                    columns[name] = []
                    order.append(name)
            index += 2
            continue
        if parsing and units is not None and stripped and stripped[0].isdigit():
            values = [convert(value + units[i]) for i, value in enumerate(stripped.split())]
            if not time or len(time) <= len(columns[names[0]]):
                time.append(values[0])
            for name, value in zip(names, values[1:]):
                columns[name].append(value)
        index += 1
    return time, {name.lower(): values for name, values in columns.items()}


def banner(title: str) -> None:
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def report_convergence(info: Dict, label: str = 'fixed point') -> None:
    """Print the iteration count and the full residual history of a solve."""
    print(f"  {label}: converged={info['converged']}  iterations={info['iters']}")
    print(f"  {label}: residual history (rms per entry, volts)")
    for k, residual in enumerate(info['residuals']):
        print(f"      iter {k:3d}   residual = {residual:.6e}")
    print(f"  {label}: final threshold = rtol*|x| + atol = {info['tolerance']:.6e}")


def report_failure(error: FixedPointError) -> None:
    """Print a non-convergent solve the way the example is supposed to report it."""
    print(f"  !! {type(error).__name__}: {error}")
    print(f"  !! residual history ({error.iters} iterations):")
    for k, residual in enumerate(error.residuals):
        print(f"      iter {k:3d}   residual = {residual:.6e}")


def set_config(args: argparse.Namespace, **overrides) -> None:
    """Apply the per-example solver settings and the ``--max-iter`` override."""
    config.update(overrides)
    if args.max_iter is not None:
        config['max_iter'] = args.max_iter


def dominant_frequency(signal: torch.Tensor, time: torch.Tensor,
                       skip_fraction: float = 0.4) -> Tuple[float, float]:
    """Dominant non-DC frequency of ``signal`` and its (single-sided) amplitude.

    The first ``skip_fraction`` of the record is discarded so that the start-up transient of an
    oscillator does not contribute; the remainder is de-meaned, Hann windowed and FFT'd on the
    uniform grid ``time``.  Returns ``(frequency in Hz, amplitude in the units of signal)``,
    with the frequency refined by a three-point parabolic interpolation of the peak so that the
    answer is not quantised to the FFT bin spacing.
    """
    start = int(skip_fraction * len(signal))
    y = signal[start:].to(dtype=torch.float64)
    t = time[start:].to(dtype=torch.float64)
    if len(y) < 8:
        return float('nan'), float('nan')
    dt = float(t[1] - t[0])
    y = y - y.mean()
    window = torch.hann_window(len(y), periodic=False, dtype=torch.float64)
    spectrum = torch.fft.rfft(y * window)
    magnitude = spectrum.abs()
    magnitude[0] = 0.0
    peak = int(torch.argmax(magnitude))
    freq = torch.fft.rfftfreq(len(y), d=dt)
    if 0 < peak < len(magnitude) - 1:
        a, b, c = (float(magnitude[peak - 1]), float(magnitude[peak]), float(magnitude[peak + 1]))
        denominator = a - 2 * b + c
        delta = 0.5 * (a - c) / denominator if denominator != 0 else 0.0
    else:
        delta = 0.0
    df = float(freq[1] - freq[0])
    # Coherent-gain corrected single-sided amplitude of a Hann-windowed sinusoid.
    amplitude = 4.0 * float(magnitude[peak]) / len(y)
    return float(freq[peak]) + delta * df, amplitude


def zero_crossing_frequency(signal: torch.Tensor, time: torch.Tensor,
                            skip_fraction: float = 0.4) -> float:
    """Oscillation frequency from the mean spacing of upward zero crossings of ``signal``.

    A completely independent estimate of :func:`dominant_frequency`, immune to spectral leakage
    but sensitive to harmonic distortion, so the two agreeing is worth something.
    """
    start = int(skip_fraction * len(signal))
    y = (signal[start:] - signal[start:].mean()).to(dtype=torch.float64)
    t = time[start:].to(dtype=torch.float64)
    crossings: List[float] = []
    for k in range(len(y) - 1):
        if float(y[k]) <= 0.0 < float(y[k + 1]):
            span = float(y[k + 1]) - float(y[k])
            frac = -float(y[k]) / span if span != 0 else 0.0
            crossings.append(float(t[k]) + frac * (float(t[k + 1]) - float(t[k])))
    if len(crossings) < 2:
        return float('nan')
    return (len(crossings) - 1) / (crossings[-1] - crossings[0])


def settling_time(signal: torch.Tensor, time: torch.Tensor, tolerance: float = 0.01,
                  final: Optional[float] = None) -> float:
    """Time after which ``signal`` stays within ``tolerance`` (relative to the total step) of
    its final value.  ``nan`` if it never does."""
    y = signal.to(dtype=torch.float64)
    end = float(y[-1]) if final is None else float(final)
    band = tolerance * abs(end - float(y[0]))
    if band == 0.0:
        return float('nan')
    last_outside = -1
    for k in range(len(y)):
        if abs(float(y[k]) - end) > band:
            last_outside = k
    if last_outside < 0:
        return 0.0
    if last_outside + 1 >= len(y):
        return float('nan')
    return float(time[last_outside + 1])


def summarise_sequence(values: Sequence[float]) -> str:
    return '[' + ', '.join(f'{v:.6g}' for v in values) + ']'
