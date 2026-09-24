import contextlib
import warnings
from typing import Tuple, Any, Callable, Dict, Optional, List, Union
import os
import sys
import re
import shlex
import shutil
import subprocess
import tempfile
import weakref
from spipe.utils import extract, convert
import numpy as np
import torch
from spipe import config
from scipy.interpolate import interp1d
import numpy as np

from spipe.electronic.sensitivity import (DeviceParameter, ZeroSensitivityError, format_key,
                                          guard_all_zero, guard_analysis_ran, parse_sensparam,
                                          read_device_parameter, rewrite_device_parameters,
                                          warn_adjoint_override)

__all__ = ['Electronic', 'reset', 'register', 'ZeroSensitivityError']


class BaseModel(object):
    model = dict()
    user_model = dict()
    name = 'Base'

    @classmethod
    def __call__(cls, level: str) -> str:
        if level in cls.model.keys():
            return cls.model[level]
        elif level in cls.user_model.keys():
            return cls.user_model[level]
        else:
            raise RuntimeError(f"The model'{level}' is not supported for {cls.name}."
                               f" Please define it via the register() function.")

    @classmethod
    def register(cls, level: str, model_str: str) -> None:
        if level in cls.model.keys():
            raise RuntimeError(f"'{level}' is already defined for {cls.name}."
                               f" Please register it under another name, e.g. 'level{len(cls.model) + len(cls.user_model) + 1}'.")
        cls.user_model[level] = model_str

    @classmethod
    def reset(cls) -> None:
        cls.user_model = dict()


class ModModel(BaseModel):
    """Electrical load a modulator presents to its driver.

    ``level1`` and ``debug`` are kept **bit for bit** -- existing netlists reference them -- even
    though ``level1`` is not a load at all: ``1e9`` Ohm is an open circuit and ``1e-18`` F is 1 aF,
    roughly five orders of magnitude below any real depletion-mode junction.  A driver terminated
    that way sees no load, so the dominant electro-optic bandwidth limit of the link is simply
    missing from the simulation.

    ``level3`` is the realistic one: a two-section RC ladder for a depletion-mode silicon MZM
    segment.  Orders of magnitude (1 mm segment, reverse-biased lateral pn junction in a rib
    waveguide):

    ==========  =========  =========================================================
    parameter   default    order of magnitude / where it comes from
    ==========  =========  =========================================================
    ``cj``      200 fF     junction capacitance, 0.1 - 1 pF/mm for a lateral pn
                           depletion phase shifter (0.2 - 0.5 pF/mm is typical)
    ``rs``      10 Ohm     series access resistance of the doped slab, 5 - 30 Ohm/mm
    ``rsh``     1 MOhm     shunt leakage of the reverse-biased junction (sub-uA at a
                           few volts), i.e. nearly, but not exactly, an open circuit
    ``cpad``    20 fF      pad / routing capacitance seen at the driver node
    ==========  =========  =========================================================

    ``rs * cj`` then gives a ~2 ps RC time constant, i.e. an intrinsic ~80 GHz electrical pole --
    the right ballpark for a lumped segment of a travelling-wave modulator, and in any case a
    *finite* load where ``level1`` had none.  Splitting it into two sections (each ``rs/2`` and
    ``cj/2``) makes the phase response of the ladder closer to the distributed line than a single
    lumped RC would be.

    The element carrying the DC current is deliberately called ``rmod``: the power report emits
    ``.PRINT TRAN power3=par(p(x_mod_<n>.rmod))``, so every ``mod`` level must contain one.
    """

    model = {'level1': ".subckt mod_level1 n ground\n" # a little strange, but understandable? 
                       "rmod n ground 1e9\n"
                       "c1 n ground 1e-18\n"
                       ".ends mod_level1\n",
             # Depletion-mode Si MZM segment: series access resistance + junction capacitance,
             # as a two-section RC ladder, plus junction leakage and pad capacitance.
             'level3': ".subckt mod_level3 n ground cj=200f rs=10 rsh=1meg cpad=20f\n"
                       "cpad1 n ground 'cpad'\n"
                       "rs1 n a 'rs/2'\n"
                       "cj1 a ground 'cj/2'\n"
                       "rmod a ground 'rsh*2'\n"
                       "rs2 a b 'rs/2'\n"
                       "cj2 b ground 'cj/2'\n"
                       "rsh2 b ground 'rsh*2'\n"
                       ".ends mod_level3\n",
             'debug': ".subckt mod_debug n ground\n"
                      "rmod n ground 1e9\n"
                      ".ends mod_debug\n", }
    name = 'Mod'


class PdModel(BaseModel):
    model = {
        'level2': ".subckt pd_level2 n ground Is=1e-12 N=1.0 Cj0=1e-12 Vj=0.7 M=0.5\n"  # a more advanced source model of a PD.
                  "Ipd cathode anode\n"  # at run-time, it will be replaced by a PWL current source
                  "d1 anode cathode diode\n"
                  "r1 anode cathode 10k\n"
                  "c1 anode cathode 'Cj0*(1+V(anode,cathode)/Vj)**(-M)'\n" # 
                  ".model diode D(IS=Is N=N)\n"
                  "vneg anode ground -1.0\n"
                  "rf cathode n 10k\n"
                  "cf cathode n 2fF\n"
                  "E n ground ground cathode 3162\n" # 70db gain ideal Opamp
                  ".ends pd_level2\n",
        'level1': ".subckt pd_level1 n ground\n"  # the most simplified source model of a PD.
                  "Ipd ground n1\n"  # at run-time, it will be replaced by a PWL current source
                  "c1 ground n1 2f\n"
                  "r1 n1 n 1k\n"
                  ".ends pd_level1\n",
        'debug': ".subckt pd_debug n ground\n"
                 "Ipd ground n\n"
                 ".ends pd_debug\n"
        }
    name = 'Pd'


def _element_name(p_line: str) -> str:
    """The element name of a photonic netlist line, or ``''`` if the line declares no element."""
    line = p_line.split('#')[0].strip()
    return line.split()[0] if line else ''


def _is_active_photonic(p_line: str) -> bool:
    """Whether a photonic netlist line declares an electrically driven device.

    Asks the photonic model table whether the model behind this element declares an
    electrical drive port (``active_port > 0``).  This used to be ``p_line.startswith('mod')``,
    which emitted the modulator-equivalent load only for elements whose *name* began with
    ``mod``: an ``mzm`` line got no load and no drive node, so the photonic side saw a device
    frozen at its bias point.  ``modm`` and ``modp`` declare ``active_port = 1``, so the two
    tests agree on them exactly.

    The import is deferred because :mod:`spipe.electronic` is imported while
    :mod:`spipe.photonic` is still being set up.
    """
    from spipe.photonic.photonic import is_active_device

    name = _element_name(p_line)
    return bool(name) and is_active_device(name)


def _is_detector(p_line: str) -> bool:
    """Whether a photonic netlist line declares a photo detector.

    Detectors are not entries of the photonic model table (they are handled by
    :class:`~spipe.photonic.model.pd_array.PDArray`, not by a :class:`Device`), so they are
    still recognised by their ``pd`` prefix -- the same rule the photonic parser uses.
    """
    return _element_name(p_line).lower().startswith('pd')


#: terminals of each built-in device between which a DC current can flow
_DC_TERMINALS = {'r': (0, 1), 'l': (0, 1), 'v': (0, 1), 'e': (0, 1), 'h': (0, 1),
                 'd': (0, 1), 's': (0, 1), 'w': (0, 1), 'm': (0, 2), 'q': (0, 1, 2)}


def _check_photocurrent_dc_path(elements, blocks, p_content) -> None:
    """Refuse a photodetector whose current has no DC path to ground.

    A detector injects a current. If the node it flows into reaches ground only through
    capacitors, that current charges them forever: the built-in engine either fails to converge
    or -- worse -- runs and returns a detector voltage of hundreds of kilovolts. SPICE itself
    rejects a node with no DC path to ground; this names the fix (a load resistor).
    """
    ground = {'0', 'gnd', 'gnd!', 'ground'}
    adjacency: Dict[str, set] = {}
    for el in elements:
        idx = _DC_TERMINALS.get(el.letter)
        if idx is None:
            continue
        nodes = [el.nodes[i] for i in idx if i < len(el.nodes)]
        for u in nodes:
            adjacency.setdefault(u, set()).update(n for n in nodes if n != u)
    for instance, plus, minus in blocks:
        for node in (plus, minus):
            if node.lower() in ground:
                continue
            seen, stack = {node}, [node]
            while stack:
                for nxt in adjacency.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt); stack.append(nxt)
            if not seen & ground:
                index = int(instance.rsplit('_', 1)[-1])
                p_line = p_content[index].strip() if index < len(p_content) else instance
                out = extract(p_line)[1][_model_dict['pd'][1]] if index < len(p_content) else '?'
                raise ValueError(
                    f"photodetector '{p_line}': its output node {out!r} has no DC path to "
                    f"ground, so the photocurrent would charge a capacitor without limit. Add a "
                    f"load, e.g. 'Rload {out} 0 1k', or a transimpedance amplifier.")


def _process_spice_wrk_dir(spice_wrk_dir: str) -> None:
    """Make the SPICE scratch directory, keeping the reason it could not be made.

    ``exist_ok`` closes the check-then-create race (two SPIPE runs sharing a work
    directory), and chaining preserves the OSError -- "Permission denied" and "No space
    left on device" need different fixes, and the old bare ``except:`` discarded both.
    """
    try:
        os.makedirs(spice_wrk_dir, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f'Could not create the SPICE working directory {spice_wrk_dir!r}: {error}'
        ) from error


_model_dict = {'mod': [ModModel(), -2], 'pd': [PdModel(), -2]}


def _detect_backend(spice_exe: str) -> str:
    """Which electronic engine a ``spice_exe`` string selects: xyce, hspice or native.

    ``'native'`` (SPIPE's own differentiable engine, :mod:`spipe.electronic.native`) is
    selected by passing the literal string ``'native'`` as ``spice_exe``.  The subprocess
    engines keep being recognised by the executable path, exactly as before, and are tested
    first so that a path that happens to contain the word ``native`` still resolves to the
    tool it names.
    """
    low = str(spice_exe).lower()
    if 'xyce' in low:
        return 'xyce'
    if 'hspice' in low:
        return 'hspice'
    if 'native' in low or 'spipe' in low:
        return 'native'
    raise NotImplementedError(
        f"Construction is not implemented for {spice_exe!r}. The electronic back ends are "
        f"'xyce' (a path containing 'Xyce'), 'hspice' (a path containing 'hspice') and "
        f"'native' (the string 'native', SPIPE's own differentiable engine -- the only one "
        f"that gives exact gradients with respect to device parameters).")


def _xyce_device_separator(spice_exe: str) -> str:
    """``':'`` or ``'.'`` -- how Xyce spells ``DEVICE<sep>PARAM`` in a ``.SENS`` card.

    Xyce's default hierarchy separator is ``':'``, and a device-instance sensitivity
    parameter is written ``M1:W``.  ``-hspice-ext separator`` (which ``-hspice-ext all``
    implies) switches the separator to ``'.'``, and ``M1:W`` then fails to parse::

        $ Xyce -hspice-ext all deck.sp
        Netlist error: Function or variable M1:W is not defined

    Measured on Xyce 7.10; ``-hspice-ext units`` and ``-hspice-ext math`` are both harmless.
    The shipped examples drive Xyce with ``-hspice-ext all``, so this has to be detected
    rather than documented.
    """
    low = str(spice_exe).lower()
    match = re.search(r'-hspice-ext\s+([a-z,]+)', low)
    if match and ('all' in match.group(1).split(',') or 'separator' in match.group(1).split(',')):
        return '.'
    return ':'


#: Placeholder in the Xyce sub-circuit list where the sensitivity cards go when they are
#: wanted. Kept as a marker so the cards keep their position in the deck either way.
_XYCE_SENS_SLOT = '\x00SPIPE_XYCE_SENS\x00'

#: Environment variable that names the executable for each subprocess back end.
_SPICE_ENV = {'xyce': 'SPIPE_XYCE', 'hspice': 'SPIPE_HSPICE'}
#: The executable's usual name on PATH. Xyce's is capitalised.
_SPICE_BINARY = {'xyce': 'Xyce', 'hspice': 'hspice'}


def _resolve_spice_command(spice_exe: str, backend: str) -> List[str]:
    """Turn ``spice_exe`` into an argument list for :func:`subprocess.run`.

    ``spice_exe`` may be a bare back-end name (``'xyce'``, ``'Xyce'``, ``'hspice'``), a path
    to the executable, or either followed by extra flags (``'Xyce -hspice-ext all'``). A bare
    name is looked up as ``$SPIPE_XYCE`` / ``$SPIPE_HSPICE`` first and then on ``PATH`` under
    the tool's real spelling, so the documented lower-case ``'xyce'`` finds ``Xyce``.
    """
    tokens = shlex.split(str(spice_exe))
    if not tokens:
        raise RuntimeError(f"spice_exe is empty; expected e.g. 'native', 'xyce' or 'hspice'.")
    exe, extra = tokens[0], tokens[1:]
    if os.sep not in exe and exe.lower() == backend:
        env_name = _SPICE_ENV[backend]
        from_env = os.environ.get(env_name)
        if from_env and shutil.which(from_env) is None:
            raise RuntimeError(
                f"${env_name}={from_env!r} does not name an executable file. Point it at the "
                f"{_SPICE_BINARY[backend]} executable (check with `which {_SPICE_BINARY[backend]}`), "
                f"or unset it to search PATH.")
        chosen = from_env or shutil.which(_SPICE_BINARY[backend]) or shutil.which(exe)
        if not chosen:
            raise RuntimeError(
                f"spice_exe={spice_exe!r} selects the {backend} back end, but no executable "
                f"was found: ${env_name} is not set and neither {_SPICE_BINARY[backend]!r} nor "
                f"{exe!r} is on PATH. Check with `which {_SPICE_BINARY[backend]}`, set "
                f"${env_name} to the executable, or use spice_exe='native'.")
        exe = chosen
    elif shutil.which(exe) is None:
        raise RuntimeError(
            f"The {backend} executable {exe!r} (from spice_exe={spice_exe!r}) does not exist or "
            f"is not executable.")
    return [exe] + extra


def _run_spice(command: List[str], outputs: List[str]) -> None:
    """Run one SPICE job; raise on failure, never fall back to an earlier run's results.

    Every file in *outputs* is deleted first. SPIPE reads results back from files, so without
    this a run that failed -- a missing executable, a netlist error, a licence refusal --
    left the previous run's output in place, and it was read back as though it were new.
    """
    for path in outputs:
        if os.path.exists(path):
            os.remove(path)
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        text = (proc.stderr or '') + (proc.stdout or '')
        tail = '\n'.join(text.strip().splitlines()[-15:])
        raise RuntimeError(
            f"{os.path.basename(command[0])} exited with status {proc.returncode}.\n"
            f"  command: {' '.join(shlex.quote(c) for c in command)}\n"
            f"  last output:\n{tail}")


# -------------------------------------------------------------------------------------------
# native back end helpers
# -------------------------------------------------------------------------------------------

_GROUND_NAMES = ('0', 'gnd', 'gnd!', 'ground')


def _strip_photocurrent_source(model_content: str, model_name: str
                               ) -> Tuple[str, List[str], Tuple[str, str]]:
    """Split a ``pd_*`` sub-circuit into (body without ``Ipd``, port list, Ipd's two nodes).

    The Xyce and HSPICE back ends turn the ``Ipd`` placeholder into a PWL / ``table()``
    current source whose breakpoint values are ``.PARAM``s.  The native back end does not: it
    deletes the card and attaches an external block on the same two nodes instead, which is
    both O(1) per evaluation and differentiable without any parameter bookkeeping.
    """
    lines = model_content.rstrip().split('\n')
    header = lines[0].split()
    ports = [token for token in header[2:] if '=' not in token]
    for index, line in enumerate(lines):
        if line.strip().lower().startswith('ipd'):
            tokens = line.split()
            if len(tokens) < 3:
                raise RuntimeError(
                    f"The photocurrent connection 'Ipd' of 'pd_{model_name}' must name two "
                    f"nodes ('Ipd <node+> <node->'), but the card is {line.strip()!r}.")
            body = '\n'.join(lines[:index] + lines[index + 1:])
            return body, ports, (tokens[1], tokens[2])
    raise RuntimeError(
        f"Please provide the photocurrent connection in the PD model definition "
        f"'pd_{model_name}'")


def _flatten_node(node: str, port_map: Dict[str, str], instance: str) -> str:
    """The flattened name of a node inside sub-circuit instance *instance*.

    A port maps to whatever the instance connected it to, ground stays ground, anything else
    becomes the hierarchical ``instance.node`` the native parser produces.
    """
    low = str(node).lower()
    for port, actual in port_map.items():
        if str(port).lower() == low:
            return str(actual).lower()
    if low in _GROUND_NAMES:
        return '0'
    return f"{instance}.{low}".lower()


class _PhotocurrentBlock(object):
    """The detector photocurrent, as a native-engine external block.

    Holds the whole photocurrent waveform as one ``(num_time,)`` float64 tensor and injects
    it as an ideal current source between ``node+`` and ``node-``, linearly interpolated onto
    whatever internal time step the integrator picks.  The block is stateless and its
    residual does not depend on the circuit unknowns, so both Jacobians are zero; the current
    does depend on the tensor, which is what makes ``d(anything)/d(photocurrent)`` fall out of
    the engine's adjoint sweep with no extra machinery.

    Sign convention is SPICE's: ``I n+ n-`` drives positive current out of ``n+``, through
    the source, into ``n-``, i.e. the residual pair is ``[+i, -i]`` -- the same pair a
    two-terminal device stamp contributes.
    """

    n_states = 0

    def __init__(self, node_p: str, node_n: str, num_time: int, name: str = 'ipd') -> None:
        self.name = name
        self.nodes = [str(node_p), str(node_n)]
        self.I = torch.zeros(int(num_time), dtype=torch.float64)
        self._grid = np.linspace(0.0, 1.0, max(2, int(num_time)))
        self._zero = torch.zeros(2, 2, dtype=torch.float64)

    def set_grid(self, grid) -> None:
        """Set the time abscissa the photocurrent samples sit on."""
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
        self._grid = np.asarray(grid, dtype=float).reshape(-1)

    def parameters(self) -> Dict[str, torch.Tensor]:
        """The photocurrent waveform, exposed as a differentiable tensor."""
        return {'I': self.I}

    def init_state(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.float64)

    def value(self, t: float) -> torch.Tensor:
        grid, n = self._grid, len(self._grid)
        if n < 2 or self.I.numel() < 2:
            return self.I.reshape(-1)[0]
        k = int(np.searchsorted(grid, t, side='right')) - 1
        k = 0 if k < 0 else (n - 2 if k > n - 2 else k)
        span = grid[k + 1] - grid[k]
        w = 0.0 if span <= 0.0 else (float(t) - grid[k]) / span
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
        if w == 0.0:
            return self.I[k]
        if w == 1.0:
            return self.I[k + 1]
        return self.I[k] * (1.0 - w) + self.I[k + 1] * w

    def residual_and_jacobian(self, x, xdot, t):
        current = self.value(t)
        return torch.stack([current, -current]), self._zero, self._zero


_SIGNAL_RE = re.compile(r'^\s*([vi])\s*\(\s*([^),\s]+)\s*(?:,\s*([^),\s]+)\s*)?\)\s*$', re.I)


def _native_signal(result, name: str) -> torch.Tensor:
    """One ``.print tran`` column out of a native :class:`TranResult`.

    Understands ``v(node)``, ``v(a,b)``, ``i(device)`` and a bare node name.
    """
    match = _SIGNAL_RE.match(name)
    if match is None:
        if re.match(r'^\s*[\w.#!]+\s*$', name):
            return result.v(name.strip())
        raise RuntimeError(
            f"The native electronic back end cannot evaluate the '.print tran' column "
            f"{name!r}. It understands 'v(node)', 'v(a,b)', 'i(device)' and a bare node "
            f"name; expressions such as par(...) are Xyce/HSPICE only.")
    kind, first, second = match.group(1).lower(), match.group(2), match.group(3)
    if kind == 'i':
        if second is not None:
            raise RuntimeError(f"'i(a,b)' is not a native back end signal: {name!r}")
        return result.i(first)
    if second is None:
        return result.v(first)
    return result.v(first) - result.v(second)


def _resample_weights(times: torch.Tensor, grid: torch.Tensor):
    """``None`` when *times* already **is** *grid*, else linear interpolation weights.

    The native engine puts an output point at every requested print time, so with a uniform
    ``.tran`` grid the two agree exactly and nothing is resampled -- which matters, because
    an interpolation between the solver's own output points would put a smoothing error on
    top of every gradient.
    """
    scale = float(times.abs().max()) if times.numel() else 1.0
    if times.numel() == grid.numel():
        if float((times - grid).abs().max()) <= 1e-9 * max(scale, 1e-30):
            return None
    known = times.detach().cpu().numpy()
    want = grid.detach().cpu().numpy()
    upper = np.clip(np.searchsorted(known, want, side='left'), 1, max(1, len(known) - 1))
    lower = upper - 1
    span = known[upper] - known[lower]
    weight = np.where(span > 0, (want - known[lower]) / np.where(span > 0, span, 1.0), 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    return (torch.as_tensor(lower, dtype=torch.long),
            torch.as_tensor(upper, dtype=torch.long),
            torch.as_tensor(weight, dtype=torch.float64))


def _resample(values: torch.Tensor, weights) -> torch.Tensor:
    """Apply :func:`_resample_weights`, differentiably, to a ``(time, column)`` tensor."""
    if weights is None:
        return values
    lower, upper, weight = weights
    factor = weight.reshape(-1, *([1] * (values.dim() - 1)))
    return values.index_select(0, lower) * (1.0 - factor) + values.index_select(0, upper) * factor


# extract 3 from the following strings using -2 as index:
# '5 7 15 17 3 level1' given 'modp1 5 7 15 17 3 level1 c=10'
# '5 7 3 level1' given 'modp1 5 7 3 level1 c=10'

def register(device: str, model_name: str, model_content: str) -> None:
    if device.lower() == 'mod':
        _model_dict['mod'][0].register(model_name, model_content)
    elif device.lower() == 'pd':
        _model_dict['pd'][0].register(model_name, model_content)
    else:
        raise NotImplementedError(f"register() function receives an unrecognized device: {device}")


def reset() -> None:
    for model in _model_dict.values():
        model[0].reset()


class Electronic(object):
    def __init__(self,
                 e_content: List[str],
                 p_content: List[str],
                 spice_exe: str,
                 num_time_ein: int,
                 power_node: Optional[List] = None,
                 use_adjoint: Optional[bool] = False,
                 spice_wrk_dir: Optional[str] = None,
                 spice_file_name: Optional[str] = 'tmp_sim.sp',
                 param_file_name: Optional[str] = 'param.txt',
                 keep_spice: Optional[bool] = False,
                 include_dir: Optional[str] = None,
                 ) -> None:
        # E2.1: '.sensparam DEV:PARAM ...' declares the electronic device parameters this
        # netlist is differentiable with respect to.  It is stripped here, before any back end
        # sees the deck, so a netlist that does not use the card produces exactly the list it
        # always did and every back end's output is unchanged.
        self.e_content, self.sens_declared = parse_sensparam(e_content)
        self.p_content = p_content
        self.num_time_ein = num_time_ein
        self.power_node = [] if power_node is None else power_node
        self.pd_identifier = []
        self.mod_identifier = []
        self.pd_cnt = 0

        # With no directory given, every instance gets its own scratch directory, removed when
        # the instance is garbage collected, so two runs never share (and clobber) one deck and
        # nothing is left in the caller's working directory. Pass a directory to keep the files.
        if spice_wrk_dir is None:
            spice_wrk_dir = tempfile.mkdtemp(prefix='spipe_')
            weakref.finalize(self, shutil.rmtree, spice_wrk_dir, True)
        _process_spice_wrk_dir(spice_wrk_dir)
        self.spice_wrk_dir = spice_wrk_dir
        #: relative ``.include`` / ``.lib`` paths in the deck resolve against this directory
        #: (the netlist's own directory when called through Circuit) on the built-in engine
        self.include_dir = os.path.abspath(include_dir) if include_dir else os.getcwd()
        self.spice_file_path = os.path.join(spice_wrk_dir, spice_file_name)
        self.param_file_name, self.param_file_path = param_file_name, os.path.join(spice_wrk_dir, param_file_name)

        self.keep_spice = keep_spice
        self.spice_exe = spice_exe
        self.backend = _detect_backend(spice_exe)

        # Xyce's transient *adjoint* returns an identically zero sensitivity block for device
        # parameters (measured on 7.10, see spipe.electronic.sensitivity).  When device
        # parameters are declared the direct method is therefore forced, and the caller is told.
        if self.sens_declared and use_adjoint:
            warn_adjoint_override(self.sens_declared)
            use_adjoint = False
        self.use_adjoint = use_adjoint

        self.sub_circuit = []
        self.print_order = []
        self.prob_res = []

        #: ``{(device_lower, PARAM): float64 leaf tensor}`` for every ``.sensparam`` entry.
        self.sens_values: Dict[DeviceParameter, torch.Tensor] = {}
        #: Gradients are only tracked while :meth:`enable_gradients` is active; the plain
        #: fixed-point loop runs under ``no_grad`` so that a hundred iterations of the native
        #: engine do not retain a hundred transient records.
        self.grad_enabled = False
        self._native = None

        if self.backend == 'xyce':
            self._init_device_parameters_from_netlist()
            self.construct_for_xyce()
        elif self.backend == 'hspice':
            self._init_device_parameters_from_netlist()
            self.construct_for_hspice()
        else:
            self.construct_for_native()

    # ------------------------------------------------------------------ device parameters
    def _init_device_parameters_from_netlist(self) -> None:
        """Read each declared device parameter's value off its card into a float64 leaf."""
        for device, name in self.sens_declared:
            value = read_device_parameter(self.e_content, device, name)
            tensor = torch.tensor(float(value), dtype=torch.float64,
                                  device=config['device'])
            tensor.requires_grad_(True)
            self.sens_values[(device.lower(), name)] = tensor

    def param(self, device: str, name: str) -> torch.Tensor:
        """The float64 **leaf tensor** holding one declared device parameter.

        Mirrors :meth:`spipe.electronic.native.Netlist.param`, so the same code works
        whichever back end is driving the circuit::

            w = circuit.e_circuit.param('mn1', 'W')   # already requires grad: .sensparam did that
            ...                             # simulate, build an objective, backward()
            w.grad                          # d(objective) / dW, in metres^-1

        Writing a new value (``w.data.fill_(20e-6)``) takes effect on the next
        :meth:`simulate` call on every back end.

        :raises RuntimeError: if the parameter was not declared with ``.sensparam``, listing
            what was.
        """
        key = (str(device).lower(), str(name).upper())
        if key in self.sens_values:
            return self.sens_values[key]
        listed = ', '.join(format_key(d, p) for d, p in self.sens_declared) or '(none)'
        raise RuntimeError(
            f"{format_key(device, name)} is not a differentiable electronic parameter of "
            f"this circuit. Declare it in the .electronic section with "
            f"'.sensparam {format_key(device, name)}'. Currently declared: {listed}.")

    def parameters(self) -> Dict[DeviceParameter, torch.Tensor]:
        """Every declared device parameter, keyed by ``(device_lower, PARAM)``."""
        return dict(self.sens_values)

    @property
    def sens_param_names(self) -> List[str]:
        """The declared device parameters as ``DEVICE:PARAM`` strings, in declaration order."""
        return [format_key(d, p) for d, p in self.sens_declared]

    @contextlib.contextmanager
    def enable_gradients(self):
        """Context manager: build an autograd graph across :meth:`simulate` calls.

        Outside it the native back end runs under ``torch.no_grad()``, which is what keeps a
        long fixed-point iteration from retaining one transient record per iteration.  The
        subprocess back ends are unaffected (their gradients come from a file, not a graph).
        """
        previous = self.grad_enabled
        self.grad_enabled = True
        try:
            yield self
        finally:
            self.grad_enabled = previous

    @property
    def xyce_photocurrent_sensitivity(self) -> bool:
        """Whether the Xyce deck asks for the *photocurrent* sensitivity block.

        This is the cost decision E2.1 is about.  The two classes of sensitivity parameter
        are wildly different sizes -- ``pd_cnt * num_time_ein`` photocurrent PWL values
        (hundreds to thousands) against a handful of ``.sensparam`` device parameters -- and
        Xyce's ``direct`` method, which is the only one that works at all for device
        parameters, costs **one extra linear solve per parameter per time step**.  Measured
        here: 401 time samples with one detector takes a 4 ms transient to over five minutes.

        The photocurrent block is what makes ``d(drive)/d(photocurrent)`` -- the coupling
        Jacobian -- available, and it is needed only by a circuit whose photocurrent actually
        reaches a modulator.  Setting ``spipe.config['xyce_sens_photocurrent'] = False`` drops
        it; SPIPE then establishes the coupling by measurement
        (:meth:`probe_decoupling`) exactly as it does for HSPICE, and refuses to proceed on a
        circuit that turns out to have feedback rather than guessing.  The default is
        ``True``, which is what SPIPE has always emitted.
        """
        return bool(config.get('xyce_sens_photocurrent', True))

    @property
    def supports_photocurrent_gradient(self) -> bool:
        """Whether this back end can differentiate the drive voltages w.r.t. the photocurrent.

        ``True`` for the native engine (autograd) and for Xyce (``.SENS`` on the PWL
        parameters); ``False`` for HSPICE, which has no transient parameter sensitivity
        analysis at all, and for Xyce when the photocurrent block has been switched off for
        cost (see :attr:`xyce_photocurrent_sensitivity`).  A ``False`` here does not mean no
        gradients: it means the coupling Jacobian has to be established by the measurement in
        :meth:`probe_decoupling` rather than computed, and only a circuit that turns out to be
        feedback free can proceed.
        """
        if self.backend == 'native':
            return True
        return self.backend == 'xyce' and self.xyce_photocurrent_sensitivity

    def probe_decoupling(self, t_value: torch.Tensor, param_value: torch.Tensor,
                         drive: torch.Tensor, rtol: Optional[float] = None) -> Tuple[bool, float]:
        """Measure whether the photocurrent influences the modulator drive at all.

        Runs the electronic solve once more with the photocurrent perturbed by a large,
        deterministic amount and compares the drive voltages.  If nothing moved -- **bit for
        bit**, since the same deck with the same numbers is a deterministic function -- then
        no detector reaches a modulator, the coupling Jacobian is identically zero, and
        ``(I - g'f')^{-1}`` is the identity *exactly*.  That is the only case in which a back
        end with no photocurrent sensitivity can still deliver an exact coupled gradient, and
        this is how SPIPE establishes it rather than assuming it.

        The "bit for bit" is a real requirement and a real trap on a subprocess back end: a
        SPICE run whose *time step* is chosen adaptively answers a slightly different
        discretised problem when the photocurrent changes, so a decoupled circuit can still
        report a ~1e-3 relative change simply because the integration grid moved.  Pin the
        step and the print precision (``.option numdgt=7 delmax=... relv=... absv=...``, as
        :func:`examples._common.spice_options` does) or raise
        ``config['coupling_probe_rtol']`` deliberately.

        :param rtol: threshold, default ``config['coupling_probe_rtol']`` (1e-9).
        :returns: ``(decoupled, relative change)``.
        """
        rtol = float(config.get('coupling_probe_rtol', 1e-9)) if rtol is None else float(rtol)
        base = param_value.detach()
        scale = float(base.abs().max())
        scale = scale if scale > 0 else 1.0
        generator = torch.Generator(device=base.device)
        generator.manual_seed(int(config['seed']))
        direction = torch.randn(base.shape, generator=generator, dtype=base.dtype,
                                device=base.device)
        reference = max(float(drive.detach().abs().max()), 1.0)

        def response(amplitude: float) -> float:
            with torch.no_grad():
                perturbed, _, _ = self.simulate(t_value, base + amplitude * scale * direction)
            return float((perturbed - drive.detach()).abs().max()) / reference

        large = response(0.1)
        if large <= rtol:
            return True, large

        # The drive moved -- but a subprocess SPICE with adaptive time stepping answers a
        # slightly different discretised problem whenever its input waveform changes, so
        # "moved" is not yet "responded".  A *response* scales with the stimulus; numerical
        # irreproducibility does not.  Shrink the perturbation a hundredfold and look.
        small = response(0.001)
        ratio = small / max(large, 1e-300)
        decoupled = ratio > float(config.get('coupling_probe_linearity', 0.2))
        return decoupled, (small if decoupled else large)

    def capture_state(self) -> Optional[List[torch.Tensor]]:
        """An opaque snapshot of whatever per-run state the back end keeps between solves.

        The implicit-function-theorem gradient in :mod:`spipe.core.core` builds one round trip
        to expose the coupling Jacobian, then runs a *second* round trip to give the other
        outputs a total derivative -- and that second run overwrites the drive the first one
        was solved with.  A vector-Jacobian product through the first graph must see the first
        run's state, so it is captured here and put back by :meth:`restored_state`.

        Only the native back end keeps such state (the photocurrent tensors of its external
        blocks).  The subprocess back ends re-read or re-run, and return ``None``.
        """
        if self.backend == 'native':
            return [block.I for block in self._native_blocks]
        return None

    @contextlib.contextmanager
    def restored_state(self, state: Optional[List[torch.Tensor]]):
        """Temporarily put back a snapshot taken by :meth:`capture_state`."""
        if state is None:
            yield self
            return
        previous = [block.I for block in self._native_blocks]
        for block, value in zip(self._native_blocks, state):
            block.I = value
        try:
            yield self
        finally:
            for block, value in zip(self._native_blocks, previous):
                block.I = value

    def xyce_device_param_names(self) -> List[str]:
        """The declared device parameters spelled the way this Xyce command line wants them.

        ``M1:W`` normally, ``M1.W`` when ``-hspice-ext`` switches the separator; see
        :func:`_xyce_device_separator`.
        """
        separator = _xyce_device_separator(self.spice_exe)
        return [f"{device}{separator}{name}" for device, name in self.sens_declared]

    def _write_spice_deck(self) -> None:
        """(Re)write the generated SPICE deck with the current device-parameter values.

        Called once at construction and, when ``.sensparam`` declared anything, again before
        every :meth:`simulate`, so that writing a new value into the leaf returned by
        :meth:`param` really does change the next run -- which is what an optimiser needs.
        """
        content = self.e_content
        if self.sens_values:
            content = rewrite_device_parameters(
                content, {(d, n): self.sens_values[(d.lower(), n)]
                          for d, n in self.sens_declared})
        cards = (''.join(getattr(self, '_xyce_sens_cards', []))
                 if getattr(self, '_deck_with_sens', False) else '')
        with open(self.spice_file_path, 'w') as f:
            f.write('* electronic circuit cut off the photonic part\n')
            f.write(f'.INC {self.param_file_name}\n')
            f.write(''.join(self.sub_circuit).replace(_XYCE_SENS_SLOT, cards) + '\n')
            f.write(''.join(content) + '\n')


    def construct_for_xyce(self) -> None:
        occur = set()
        for line_index, p_line in enumerate(self.p_content):
            if _is_active_photonic(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['mod'][0](model_name)
                _, _, required_kv_pair = extract(model_content.split('\n')[0])
                node_interface = strings[_model_dict['mod'][1]]

                if f"mod_{model_name}" not in occur:
                    self.sub_circuit.append('\n' + model_content + '\n')
                    occur.add(f"mod_{model_name}")
                self.sub_circuit.append(f"x_mod_{line_index} {node_interface} 0 mod_{model_name} " +
                                        ''.join([f"{k}={v} " for k, v in got_kv_pair.items() if
                                                 k in required_kv_pair.keys()]) + "\n")
                self.mod_identifier.append(f"x_mod_{line_index}")
                if node_interface in self.print_order:
                    warnings.warn(f"node {node_interface} connects two modulators")
                else:
                    self.print_order.append(node_interface)

            elif _is_detector(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['pd'][0](model_name)
                model_content_first_line = model_content.split('\n')[0]
                _, _, required_kv_pair = extract(model_content_first_line)
                node_interface = strings[_model_dict['pd'][1]]

                if f"pd_{model_name}" not in occur:
                    split_model_content = model_content.rstrip().split('\n')
                    split_model_content[0] += ''.join([f" t{i}=-1 v{i}=-1" for i in range(self.num_time_ein)])

                    for i in range(1, len(split_model_content)):
                        if split_model_content[i].startswith('Ipd'):
                            split_model_content[i] = split_model_content[i].replace('Ipd ', 'bpwlin ')
                            split_model_content[i] += f' i = {{table(time,' + ','.join(
                                [f"{{t{i}}},{{v{i}}}" for i in range(self.num_time_ein)]) + ')}'
                            break
                    else:
                        raise RuntimeError(
                            f"Please provide the photocurrent connection in the PD model definition 'pd_{model_name}'")

                    model_content = '\n'.join(split_model_content)
                    self.sub_circuit.append('\n' + model_content + 2 * '\n')
                    occur.add(f"pd_{model_name}")

                param_content = []
                for i in range(self.num_time_ein):
                    param_content.append(f"t{i}={{param_t{i}}} v{i}={{param_{i}_{self.pd_cnt}}} ")
                self.pd_cnt += 1

                self.sub_circuit.append(f"x_pd_{line_index} {node_interface} 0 pd_{model_name} " +
                                        ''.join([f"{k}={v} " for k, v in got_kv_pair.items() if
                                                 k in required_kv_pair.keys()]) +
                                        ''.join(param_content) + '\n')
                self.pd_identifier.append(f"x_pd_{line_index}")

        # The sensitivity cards are kept aside rather than written straight into the deck:
        # they cost one extra linear solve per parameter per time step (a 400-sample run went
        # from ~4 s to ~12 min), and only a backward pass reads their output. The deck carries
        # them only while a gradient is being built -- see _write_spice_deck and simulate.
        self._xyce_sens_cards = [f"\n.PRINT SENS\n",
                                 f".options sensitivity direct={int(not self.use_adjoint)} "
                                 f"adjoint={int(self.use_adjoint)}\n"]
        self.sub_circuit.append(_XYCE_SENS_SLOT)
        # Two parameter classes in one list (E2.1): the photocurrent PWL values
        # `param_{i}_{j}` (pd_cnt * num_time_ein of them) and the '.sensparam' device
        # parameters (a handful).  Xyce's `.options sensitivity` is global, so both classes
        # share one method; see spipe.electronic.sensitivity for why that method is `direct`.
        photocurrent_names = ([f"param_{i}_{j}" for j in range(self.pd_cnt)
                               for i in range(self.num_time_ein)]
                              if self.xyce_photocurrent_sensitivity else [])
        sens_names = photocurrent_names + self.xyce_device_param_names()
        if sens_names:
            for node in self.print_order:
                self._xyce_sens_cards.append(f".sens objfunc={{v({node})}} param=" +
                                             ','.join(sens_names) + "\n")

        for i, content in enumerate(self.e_content):
            if content.lower().startswith('.print tran'):
                self.prob_res.extend(content.lower().strip().strip('#').split('.print tran ')[-1].split(' '))
                self.e_content[i] = ""

        # The trailing newline is not cosmetic: without it this card and the power cards below
        # are written onto one physical line ('... v(npd).PRINT TRAN power2=par(...)'), which
        # Xyce rejects with "Unrecognized parenthetical specification in .print".  Every deck
        # with a photo detector or a power monitor emits at least one power card, so on Xyce
        # that was every deck.  (The HSPICE branch below always had the newline.)
        self.sub_circuit.append(f".PRINT TRAN {' '.join([f'v({node})' for node in self.print_order])} {' '.join(self.prob_res)}\n")

        if self.power_node:
            self.sub_circuit.append(f".PRINT TRAN power1=par(`-{'-'.join([f'p({node})' for node in self.power_node])}`)\n")

        if self.pd_identifier:
            self.sub_circuit.append(f".PRINT TRAN power2=par(`{'+'.join([f'p({id}.Ipd)' for id in self.pd_identifier])}`)\n")

        if self.mod_identifier:
            self.sub_circuit.append(f".PRINT TRAN power3=par(`{'+'.join([f'p({id}.rmod)' for id in self.mod_identifier])}`)\n")


        self._write_spice_deck()
        return

    def construct_for_hspice(self) -> None:
        occur = set()
        for line_index, p_line in enumerate(self.p_content):
            if _is_active_photonic(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['mod'][0](model_name)
                _, _, required_kv_pair = extract(model_content.split('\n')[0])
                node_interface = strings[_model_dict['mod'][1]]

                if f"mod_{model_name}" not in occur:
                    self.sub_circuit.append('\n' + model_content + '\n')
                    occur.add(f"mod_{model_name}")
                self.sub_circuit.append(f"x_mod_{line_index} {node_interface} 0 mod_{model_name} " +
                                        ''.join([f"{k}={v} " for k, v in got_kv_pair.items() if
                                                 k in required_kv_pair.keys()]) + "\n")

                if node_interface in self.print_order:
                    warnings.warn(f"node {node_interface} connects two modulators")
                else:
                    self.print_order.append(node_interface)

                self.mod_identifier.append(f"x_mod_{line_index}")

            elif _is_detector(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['pd'][0](model_name)
                model_content_first_line = model_content.split('\n')[0]
                _, _, required_kv_pair = extract(model_content_first_line)
                node_interface = strings[_model_dict['pd'][1]]

                if f"pd_{model_name}" not in occur:
                    split_model_content = model_content.rstrip().split('\n')
                    split_model_content[0] += ''.join([f" t{i}=-1 v{i}=-1" for i in range(self.num_time_ein)])

                    for i in range(1, len(split_model_content)):
                        if split_model_content[i].startswith('Ipd'):
                            split_model_content[i] += f' PWL (' + ','.join(
                                [f"t{i},v{i}" for i in range(self.num_time_ein)]) + ')'
                            break
                    else:
                        raise RuntimeError(
                            f"Please provide the photocurrent connection in the PD model definition 'pd_{model_name}'")

                    model_content = '\n'.join(split_model_content)
                    self.sub_circuit.append('\n' + model_content + 2 * '\n')
                    occur.add(f"pd_{model_name}")

                param_content = []
                for i in range(self.num_time_ein):
                    param_content.append(f"t{i}=param_t{i} v{i}=param_{i}_{self.pd_cnt} ")
                self.pd_cnt += 1

                self.sub_circuit.append(f"x_pd_{line_index} {node_interface} 0 pd_{model_name} " +
                                        ''.join([f"{k}={v} " for k, v in got_kv_pair.items() if
                                                 k in required_kv_pair.keys()]) +
                                        ''.join(param_content) + '\n')
                self.pd_identifier.append(f"x_pd_{line_index}")
        # for node in self.print_order:
        #     self.sub_circuit.append(f".PRINT TRAN v({node})\n")
        # self.sub_circuit.append(
        #     f".options sensitivity direct={int(not self.use_adjoint)} adjoint={int(self.use_adjoint)}\n")
        # for node in self.print_order:
        #     self.sub_circuit.append(f".sens objfunc={{v({node})}} param=" +
        #                             ','.join([f"param_{i}_{j}" for j in
        #                                       range(self.pd_cnt) for i in range(self.num_time_ein)]) + "\n")

        for i, content in enumerate(self.e_content):
            if content.lower().startswith('.print tran'):
                self.prob_res.extend(content.lower().strip().strip('#').split('.print tran ')[-1].split(' '))
                self.e_content[i] = ""

        self.sub_circuit.append(f".PRINT TRAN {' '.join([f'v({node})' for node in self.print_order])} {' '.join(self.prob_res)}\n")

        if self.power_node:
            self.sub_circuit.append(f".PRINT TRAN power1=par(`-{'-'.join([f'p({node})' for node in self.power_node])}`)\n")

        if self.pd_identifier:
            self.sub_circuit.append(f".PRINT TRAN power2=par(`{'+'.join([f'p({id}.Ipd)' for id in self.pd_identifier])}`)\n")

        if self.mod_identifier:
            self.sub_circuit.append(f".PRINT TRAN power3=par(`{'+'.join([f'p({id}.rmod)' for id in self.mod_identifier])}`)\n")


        self._write_spice_deck()

        return

    def simulate(self, t_value: torch.Tensor, param_value: torch.Tensor) -> [torch.Tensor, Dict, Dict]:
        if param_value.shape[0] != self.num_time_ein or param_value.shape[1] != self.pd_cnt:
            raise RuntimeError(f"Expected parameters of shape ({self.num_time_ein}, {self.pd_cnt}), "
                               f"but got ({param_value.shape[0]}, {param_value.shape[1]})")
        if len(t_value) != self.num_time_ein:
            raise RuntimeError(f"Expect {self.num_time_ein} time grids, but got {len(t_value)}")

        if self.backend == 'native':
            return self._simulate_native(t_value, param_value)

        # A '.sensparam' value lives in a torch leaf, so the deck has to be re-rendered with
        # whatever the leaf currently holds before the subprocess reads it.  Without declared
        # device parameters this is the same deck construction already wrote, byte for byte.
        # Xyce's sensitivity cards go into the deck only while a gradient is being built:
        # a plain run with them paid for every photocurrent sensitivity and then discarded it.
        want_sens = self.backend == 'xyce' and self.grad_enabled
        if self.sens_values or want_sens != getattr(self, '_deck_with_sens', False):
            self._deck_with_sens = want_sens
            self._write_spice_deck()

        sens_tensors = tuple(self.sens_values[(d.lower(), n)] for d, n in self.sens_declared)
        sens_names = tuple(self.sens_param_names)

        if self.backend == 'xyce':
            return SimulateXyce.apply(t_value, param_value,
                                      (len(self.power_node) >= 1, len(self.pd_identifier) >= 1, len(self.mod_identifier) >= 1),
                                      len(self.print_order), self.prob_res, self.spice_file_path,
                                      self.param_file_path, self.spice_exe,
                                      self.keep_spice,
                                      sens_names, tuple(self.xyce_device_param_names()),
                                      'adjoint' if self.use_adjoint else 'direct',
                                      self.grad_enabled,
                                      *sens_tensors)
        elif self.backend == 'hspice':
            return SimulateHspice.apply(t_value, param_value,
                                        (len(self.power_node) >= 1, len(self.pd_identifier) >= 1, len(self.mod_identifier) >= 1), len(self.print_order), self.prob_res, self.spice_file_path,
                                      self.param_file_path, self.spice_exe,
                                      self.keep_spice,
                                      sens_names, self, *sens_tensors)
        else:
            raise NotImplementedError(f"Simulation interface is only implemented for xyce, "
                                      f"hspice and the native engine.")

    # ------------------------------------------------------------------ native back end
    def construct_for_native(self) -> None:
        """Build the deck for :mod:`spipe.electronic.native`, SPIPE's own engine.

        Structurally the same deck the subprocess back ends get -- one ``mod_<level>``
        sub-circuit per electrically driven photonic device, one ``pd_<level>`` sub-circuit
        per detector -- with two differences, both forced by the fact that this engine is
        differentiable rather than a file interface:

        * the detector's photocurrent is **not** a PWL source.  A PWL card with one segment
          per time sample costs O(num_time) torch operations at every internal Newton
          evaluation, which for a few hundred samples dominates the whole solve.  It is
          injected instead as an :class:`~spipe.electronic.native.ExternalBlock` holding the
          photocurrent as a single ``(num_time,)`` tensor and interpolating it in O(1).  The
          engine's time-domain adjoint differentiates a block's residual with respect to its
          parameters automatically, so ``d(anything)/d(photocurrent)`` comes out exact and in
          one backward sweep.
        * no ``.sens`` / ``.options sensitivity`` cards: the gradient is autograd, not a file.

        Device parameters declared with ``.sensparam`` are looked up in the engine, so
        :meth:`param` hands back *the engine's own leaf tensor*: writing to it changes the
        next solve, and its ``.grad`` is exact rather than a finite difference.
        """
        from spipe.electronic.native import Netlist, UnknownParameterError

        occur = set()
        deck: List[str] = []
        blocks: List[Tuple[str, str, str]] = []      # (instance, node+, node-)

        for line_index, p_line in enumerate(self.p_content):
            if _is_active_photonic(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['mod'][0](model_name)
                _, _, required_kv_pair = extract(model_content.split('\n')[0])
                node_interface = strings[_model_dict['mod'][1]]

                if f"mod_{model_name}" not in occur:
                    deck.append('\n' + model_content + '\n')
                    occur.add(f"mod_{model_name}")
                deck.append(f"x_mod_{line_index} {node_interface} 0 mod_{model_name} " +
                            ''.join([f"{k}={v} " for k, v in got_kv_pair.items()
                                     if k in required_kv_pair.keys()]) + "\n")
                self.mod_identifier.append(f"x_mod_{line_index}")
                if node_interface in self.print_order:
                    warnings.warn(f"node {node_interface} connects two modulators")
                else:
                    self.print_order.append(node_interface)

            elif _is_detector(p_line):
                _, strings, got_kv_pair = extract(p_line)
                model_name = strings[-1]
                model_content = _model_dict['pd'][0](model_name)
                _, _, required_kv_pair = extract(model_content.split('\n')[0])
                node_interface = strings[_model_dict['pd'][1]]
                instance = f"x_pd_{line_index}"

                body, ports, source_nodes = _strip_photocurrent_source(model_content, model_name)
                if f"pd_{model_name}" not in occur:
                    deck.append('\n' + body + 2 * '\n')
                    occur.add(f"pd_{model_name}")

                actual = {ports[0]: node_interface, ports[1]: '0'} if len(ports) >= 2 else {}
                blocks.append((instance,
                               _flatten_node(source_nodes[0], actual, instance),
                               _flatten_node(source_nodes[1], actual, instance)))

                deck.append(f"{instance} {node_interface} 0 pd_{model_name} " +
                            ''.join([f"{k}={v} " for k, v in got_kv_pair.items()
                                     if k in required_kv_pair.keys()]) + '\n')
                self.pd_identifier.append(instance)
                self.pd_cnt += 1

        for i, content in enumerate(self.e_content):
            if content.lower().startswith('.print tran'):
                self.prob_res.extend(content.lower().strip().strip('#').split('.print tran ')[-1].split(' '))
                self.e_content[i] = ""

        self.sub_circuit = deck
        text = ('* electronic circuit cut off the photonic part (native back end)\n' +
                ''.join(deck) + '\n' + ''.join(self.e_content) + '\n')
        self.native_deck = text
        with open(self.spice_file_path, 'w') as f:
            f.write(text)

        self._native = Netlist(text, self.include_dir)
        _check_photocurrent_dc_path(self._native.parsed.elements, blocks, self.p_content)

        # X4 of the native engine: a co-simulation fixed point re-solves the electronic side
        # every iteration and does not need 1e-8 LTE, so relax the step controller unless the
        # deck asked for something specific. The gradient is the gradient of whatever was
        # integrated; how close that is to the continuous circuit depends on the step count
        # (see docs/backends.md, 'Time-step accuracy of the built-in engine').
        self._native.options['lte_reltol'] = float(config.get('native_lte_reltol', 1e-6))

        # Step control.  The engine chooses the number of internal sub-steps per print
        # interval from a truncation-error estimate, *globally*, so the step sequence is a
        # deterministic function of the netlist -- but it is still a function of the netlist,
        # and a device parameter is part of the netlist.  On a stiff or near-null circuit a
        # 0.1 % change in W can move the chosen `nsub`, and then a central finite difference
        # over W compares two different discretisations and scatters by percent while the
        # analytic gradient -- which is the exact gradient of whatever was integrated -- does
        # not.  `spipe.config['native_nsub'] = k` pins it and removes that entirely; it is the
        # setting to use when checking a gradient against finite differences.
        nsub = config.get('native_nsub')
        if nsub is not None:
            self._native.nsub = int(nsub)
            self._native.adaptive = False
            self._native.options['nsub'] = int(nsub)
            self._native.options['adaptive'] = False
        elif config.get('native_adaptive') is not None:
            self._native.adaptive = bool(config['native_adaptive'])
            self._native.options['adaptive'] = self._native.adaptive
        #: ``uic`` semantics, to match the other two back ends.  SPIPE's Xyce and HSPICE
        #: drivers have always written ``.Tran ... uic``, i.e. skip the operating point and
        #: start from the ``.ic`` values, and a back end that silently started from the DC
        #: solution instead would answer a *different* initial value problem: on a driver
        #: whose output RC time constant is comparable with the record length the two are not
        #: small perturbations of each other.  Set ``config['native_uic'] = False`` for an
        #: operating-point start, which is better conditioned when the netlist has no ``.ic``
        #: and the startup transient is not what is being studied.
        self.native_uic = bool(config.get('native_uic', True))

        self._native_blocks = []
        for instance, node_p, node_n in blocks:
            block = _PhotocurrentBlock(node_p, node_n, self.num_time_ein,
                                       name=f"{instance}.ipd")
            self._native.add_external_block(block)
            self._native_blocks.append(block)

        for device, name in self.sens_declared:
            try:
                tensor = self._native.param(device, name)
            except UnknownParameterError as exc:
                raise RuntimeError(
                    f"'.sensparam {format_key(device, name)}': {exc}") from None
            tensor.requires_grad_(True)
            self.sens_values[(device.lower(), name)] = tensor
        return

    def _simulate_native(self, t_value: torch.Tensor, param_value: torch.Tensor):
        """One transient of the native engine, driven by *param_value* photocurrents.

        Returns the same ``(modulator drive voltages, probe dict, power dict)`` triple the
        subprocess back ends return.  Inside :meth:`enable_gradients` every returned tensor
        carries an autograd graph reaching both *param_value* and the ``.sensparam`` device
        leaves; outside it the whole solve runs under ``no_grad``.

        The power report is ``None`` for the three electrical entries, exactly as the Xyce
        back end reports it -- the native engine has no ``par(p(...))`` expression evaluator.
        """
        grid = t_value.detach().to(torch.float64).reshape(-1).cpu()
        guard = contextlib.nullcontext() if self.grad_enabled else torch.no_grad()
        with guard:
            values = param_value.to(torch.float64)
            for j, block in enumerate(self._native_blocks):
                block.set_grid(grid)
                block.I = values[:, j]

            tstart, tstop = float(grid[0]), float(grid[-1])
            tstep = (tstop - tstart) / max(1, len(grid) - 1)
            result = self._native.tran(tstep, tstop, tstart=tstart, uic=self.native_uic)

            times = torch.as_tensor(result.t, dtype=torch.float64).reshape(-1)
            weights = _resample_weights(times, grid)

            columns = [_native_signal(result, f'v({node})') for node in self.print_order]
            out = (_resample(torch.stack(columns, dim=1), weights) if columns
                   else torch.zeros(len(grid), 0, dtype=torch.float64))

            probes = {}
            if self.prob_res:
                probe_stack = torch.stack([_native_signal(result, name)
                                           for name in self.prob_res], dim=1)
                probe_stack = _resample(probe_stack, weights)
                probes = {name: probe_stack[:, i] for i, name in enumerate(self.prob_res)}

        return out, probes, {'electronic': None, 'pd_equiv': None, 'mod_equiv': None}


def interp1d_warp(time: np.ndarray, result: np.ndarray, new_time: torch.Tensor):

    if time[-1] < new_time[-1]:
        warnings.warn(f"The last time point in given data is {time[-1]}, "
                      f"while the desired interpolated last time point is {new_time[-1]}."
                      f" They will be hard aligned.")
        time[-1] = new_time[-1]

    unique_time, indices = np.unique(time, return_index=True)
    unique_result = result[indices]
    f = interp1d(unique_time, unique_result, kind='cubic', axis=0)

    result_interp = torch.Tensor(f(new_time.detach().cpu().numpy())).to(new_time.device)

    return result_interp


#: Serial number for the per-run copies of Xyce's sensitivity output; see
#: :meth:`SimulateXyce.forward`.
_SENS_RUN_COUNTER = 0

_SENS_COLUMN = re.compile(r'^d_(?P<obj>.+)/d_(?P<param>.+)_(?:dir|adj)$', re.I)
_PHOTO_PARAM = re.compile(r'^param_(?P<time>\d+)_(?P<pd>\d+)$', re.I)


def _parse_xyce_sens(path: str, num_out: int, dim_ein: int, time_ein: int,
                     device_spice_names: Tuple[str, ...], t_value: torch.Tensor):
    """Read a Xyce ``.SENS.prn`` file, by header rather than by column position.

    :returns: ``(photocurrent, device, objective)`` where ``photocurrent`` has shape
        ``(len(t_value), num_out, dim_ein, time_ein)``, ``device`` has shape
        ``(len(t_value), num_out, len(device_spice_names))`` and ``objective`` has shape
        ``(len(t_value), num_out)`` -- the objective *as the sensitivity analysis itself saw
        it*, which is what lets the caller tell "this parameter has no effect" apart from
        "the whole sensitivity analysis silently produced nothing".

    The file's header names every column::

        Index  TIME  {V(D)}  d_{V(D)}/d_PARAM_0_0_dir  ...  d_{V(D)}/d_M1:W_dir

    so the two parameter classes are told apart by name and neither the number of
    ``.print tran`` columns nor the order Xyce chose can silently shift the mapping.  The
    original implementation indexed by position and assumed the transient print list
    contained nothing but the modulator nodes.
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"Xyce produced no sensitivity output ({path}). A '.SENS' analysis needs "
            f"'.PRINT SENS' and at least one parameter; check the Xyce log for a netlist "
            f"error, and note that '-hspice-ext all' switches the device parameter separator "
            f"from ':' to '.' (Xyce 7.10).")

    device_lookup = {name.lower(): index for index, name in enumerate(device_spice_names)}
    header, rows, times = None, [], []
    with open(path, 'r') as handle:
        for line in handle:
            tokens = line.split()
            if not tokens:
                continue
            if header is None:
                if tokens[0].lower() == 'index':
                    header = tokens
                continue
            if tokens[0].isdigit():
                times.append(float(tokens[1]))
                rows.append([float(value) for value in tokens[2:]])

    if header is None or not rows:
        raise RuntimeError(f"Xyce's sensitivity output {path} has no data rows; the "
                           f"sensitivity analysis did not run.")

    # Classify every data column: an objective value, or d(objective)/d(parameter).
    objectives: List[str] = []
    objective_columns: List[int] = []
    entries: List[Tuple[int, str, int]] = []                # (objective index, param, column)
    for column, name in enumerate(header[2:]):
        match = _SENS_COLUMN.match(name)
        if match is None:
            objectives.append(name)
            objective_columns.append(column)
            continue
        if not objectives:
            raise RuntimeError(f"Xyce's sensitivity output {path} starts with a derivative "
                               f"column ({name}) before any objective column.")
        entries.append((len(objectives) - 1, match.group('param'), column))

    if len(objectives) < num_out:
        raise RuntimeError(
            f"Xyce's sensitivity output {path} carries {len(objectives)} objective(s) but "
            f"SPIPE asked for {num_out} (one per modulator drive node). The '.sens' cards "
            f"and the '.PRINT TRAN' list have gone out of step.")

    values = np.asarray(rows, dtype=float)
    photo = np.zeros((values.shape[0], num_out, max(dim_ein, 0), time_ein))
    device = np.zeros((values.shape[0], num_out, len(device_spice_names)))
    photo_found = 0

    for objective_index, name, column in entries:
        if objective_index >= num_out:
            continue
        photo_match = _PHOTO_PARAM.match(name)
        if photo_match is not None:
            i, j = int(photo_match.group('time')), int(photo_match.group('pd'))
            if i < time_ein and j < dim_ein:
                photo[:, objective_index, j, i] = values[:, column]
                photo_found += 1
            continue
        index = device_lookup.get(name.lower())
        if index is not None:
            device[:, objective_index, index] = values[:, column]

    seen = values[:, objective_columns[:num_out]]

    time = np.asarray(times, dtype=float)
    photo_interp = interp1d_warp(time, photo, t_value)
    device_interp = (interp1d_warp(time, device, t_value) if device.shape[2]
                     else torch.zeros(len(t_value), num_out, 0, dtype=photo_interp.dtype,
                                      device=photo_interp.device))
    seen_interp = interp1d_warp(time, seen, t_value)
    return photo_interp, device_interp, seen_interp, photo_found


class SimulateXyce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                t_value: torch.Tensor,
                param_value: torch.Tensor,
                monitor_power: Tuple[bool],
                num_out: int,
                prob_res: List,
                spice_file_path: str,
                param_file_path: str,
                spice_exe: str,
                keep_spice: Optional[bool] = False,
                sens_names: Tuple[str, ...] = (),
                sens_spice_names: Tuple[str, ...] = (),
                sens_analysis: str = 'direct',
                differentiating: bool = False,
                *sens_tensors: torch.Tensor,
                ):
        # t_value shape (time_ein)
        # param_value shape (time_ein, dim_ein)

        with open(param_file_path, 'w') as f:
            # Print step, not 0.  With `0` Xyce reports its own *internal* time points, and
            # those move whenever any device parameter moves -- so the cubic interpolation
            # back onto SPIPE's grid is taken from a different set of abscissae for every
            # value of W, and a finite difference over W measures the resampling as much as
            # the circuit.  Printing on SPIPE's own grid makes the interpolation a no-op and
            # the output a smooth function of the parameters, which is what a gradient needs.
            # (The HSPICE branch below has always written an explicit print step.)
            #
            # Applied only when the deck actually has a modulator, i.e. exactly the decks that
            # `construct_for_xyce` also emits a `.PRINT TRAN power3=par(...)` card for -- and
            # those could not be run at all before the missing newline there was fixed, so
            # nothing that used to work changes.  A purely electrical probe deck (no active
            # photonic device) keeps Xyce's own output grid, byte for byte as before.
            if num_out > 0:
                step = (max(t_value) - min(t_value)) / max(1, len(t_value) - 1)
                f.write(f".Tran {step} {max(t_value)} {min(t_value)} uic\n")
                # Xyce treats the .TRAN step as an initial-step hint, not a print interval, and
                # by default writes a row per internal time step.  INITIAL_INTERVAL is what
                # actually pins the output grid.
                f.write(f".OPTIONS OUTPUT INITIAL_INTERVAL={step}\n")
            else:
                f.write(f".Tran 0 {max(t_value)} {min(t_value)} uic\n")
            for i in range(len(t_value)):
                f.write(f'.PARAM param_t{i}={t_value[i]}\n')

            for i in range(len(t_value)):
                for j in range(param_value.shape[1]):
                    f.write(f".PARAM param_{i}_{j}={param_value[i][j]}\n")

        _run_spice(_resolve_spice_command(spice_exe, 'xyce') + [spice_file_path],
                   [spice_file_path + '.prn', spice_file_path + '.SENS.prn'])

        if not os.path.exists(spice_file_path + '.prn'):
            raise RuntimeError(
                f"Xyce finished without error but wrote no transient output "
                f"({spice_file_path}.prn). Check the deck's .PRINT TRAN card.")

        time, result = [], []

        with open(spice_file_path + '.prn', 'r') as f:
            for line in f.readlines():
                index, t, *values = line.split()
                if index.isdigit():
                    result.append([eval(value) for value in values])
                    time.append(eval(t))

        time, result = np.array(time), np.array(result)
        result_interp = interp1d_warp(time, result, t_value)  # (time_eout, dim_eout)

        if not keep_spice:
            os.remove(spice_file_path + '.prn')

        # The implicit-function-theorem path runs the electronic solve more than once per
        # gradient, and each run overwrites '<deck>.SENS.prn'.  When a gradient is actually
        # being built, the sensitivity block is therefore moved to a per-run name so that the
        # backward of run k reads run k's numbers.  A deck without '.sensparam' and without a
        # differentiable photocurrent never takes this branch, so its files are unchanged.
        sens_path = spice_file_path + '.SENS.prn'
        if differentiating and os.path.exists(sens_path):
            global _SENS_RUN_COUNTER
            _SENS_RUN_COUNTER += 1
            unique = f"{spice_file_path}.SENS.{_SENS_RUN_COUNTER}.prn"
            os.replace(sens_path, unique)
            sens_path = unique

        ctx.spice_file_path = spice_file_path
        ctx.sens_path = sens_path
        ctx.time_eout, ctx.dim_eout = result.shape[0], result.shape[1]
        ctx.time_ein, ctx.dim_ein = param_value.shape[0], param_value.shape[1]
        ctx.num_out = num_out
        ctx.keep_spice = keep_spice
        ctx.t_value = t_value
        ctx.sens_names = tuple(sens_names)
        ctx.sens_spice_names = tuple(sens_spice_names)
        ctx.sens_analysis = sens_analysis
        ctx.objective = result_interp[:, :num_out].detach()
        ctx.param_dtype = param_value.dtype
        ctx.sens_tensors = sens_tensors
        ctx.sens_cache = None
        return (result_interp[:, :num_out],
                {prob_res[i]: result_interp[:, num_out + i] for i in range(len(prob_res))},
                {'electronic': None,
                 'pd_equiv': None,
                 'mod_equiv': None}
                )

    @staticmethod
    def backward(ctx, *grad_output):
        """Gradients from Xyce's ``.SENS`` block: photocurrent parameters **and** devices.

        The ``.SENS.prn`` file is parsed by its *header*, not by column position, and cached
        on ``ctx``, so that (a) a deck with extra ``.print tran`` columns is handled correctly
        and (b) the implicit-function-theorem solve in :mod:`spipe.core.core`, which calls
        this backward once per probe direction, re-reads nothing.

        An identically zero device block raises :class:`~.sensitivity.ZeroSensitivityError`
        rather than being returned -- see that module for the measured Xyce behaviour that
        makes the guard necessary.
        """
        if ctx.sens_cache is None:
            ctx.sens_cache = _parse_xyce_sens(ctx.sens_path,
                                              ctx.num_out, ctx.dim_ein, ctx.time_ein,
                                              ctx.sens_spice_names, ctx.t_value)
            if (not ctx.keep_spice or ctx.sens_path != ctx.spice_file_path + '.SENS.prn') \
                    and os.path.exists(ctx.sens_path):
                os.remove(ctx.sens_path)
        grad_interp, device_interp, seen_objective, photo_found = ctx.sens_cache

        # Asking for d(drive)/d(photocurrent) from a deck that did not request that block is
        # the one thing that must never come back as an array of zeros.
        if ctx.needs_input_grad[1] and photo_found == 0 and ctx.dim_ein * ctx.time_ein > 0:
            raise NotImplementedError(
                f"This deck's Xyce '.SENS' card carries no photocurrent parameters, so "
                f"d(drive)/d(photocurrent) is not available -- and SPIPE will not return "
                f"zeros for it. spipe.config['xyce_sens_photocurrent'] is "
                f"{bool(config.get('xyce_sens_photocurrent', True))}; set it to True to have "
                f"the {ctx.dim_ein * ctx.time_ein} photocurrent sensitivities computed (the "
                f"direct method costs one extra linear solve per parameter per time step, so "
                f"this is slow), or use spice_exe='native', whose adjoint gives all of them "
                f"in one sweep.")

        # The whole-analysis guard.  Xyce's transient adjoint prints the objective itself as
        # zero along with every sensitivity, so a sensitivity block that is zero *and* an
        # objective column that is zero where the transient analysis says the node moved means
        # the sensitivity analysis produced nothing at all -- not that the derivative is zero.
        guard_analysis_ran(seen_objective, ctx.objective, ctx.sens_analysis, 'xyce',
                           list(ctx.sens_names) or ['the photocurrent parameters'])

        # Xyce's sensitivity block arrives as float32 (interp1d_warp builds a torch.Tensor
        # from a numpy array), while the cotangent from the photonic side and the device
        # parameter leaves are float64.  The contraction is done in the wider dtype and each
        # gradient is returned in the dtype of the input it belongs to, which is what autograd
        # asks of a Function's backward.
        work = torch.promote_types(grad_interp.dtype, torch.float64)
        cotangent = grad_output[0]
        if cotangent is None:
            cotangent = torch.zeros_like(ctx.objective)
        cotangent = cotangent.to(dtype=work, device=grad_interp.device)

        # grad_interp: (time_eout, num_out, dim_ein, time_ein)
        grad_result = torch.einsum('ij,ijkl->lk', cotangent, grad_interp.to(work))
        grad_result = grad_result.to(ctx.param_dtype)

        device_grads = []
        for index, (name, tensor) in enumerate(zip(ctx.sens_names, ctx.sens_tensors)):
            block = device_interp[:, :, index]                    # (time_eout, num_out)
            guard_all_zero(block, [name], ctx.objective, ctx.sens_analysis, 'xyce',
                           objective_name='the modulator drive voltage')
            device_grads.append((cotangent * block.to(work)).sum().to(tensor.dtype))

        return ((None, grad_result) + (None,) * 11 + tuple(device_grads))


def _run_hspice(t_value: torch.Tensor,
                param_value: torch.Tensor,
                spice_file_path: str,
                param_file_path: str,
                spice_exe: str,
                keep_spice: bool) -> Tuple[torch.Tensor, np.ndarray]:
    """One HSPICE transient; returns ``(result interpolated onto t_value, raw result array)``.

    Split out of :meth:`SimulateHspice.forward` unchanged so that the finite-difference
    device-parameter backward can re-run the same deck without duplicating the parser.
    """
    with open(param_file_path, 'w') as f:
        f.write(f".Tran {(max(t_value)-min(t_value)) / 5 / len(t_value)} {max(t_value)} start={min(t_value)} uic\n")
        for i in range(len(t_value)):
            f.write(f'.PARAM param_t{i}={t_value[i]}\n')

        for i in range(len(t_value)):
            for j in range(param_value.shape[1]):
                f.write(f".PARAM param_{i}_{j}={param_value[i][j]}\n")

    _run_spice(_resolve_spice_command(spice_exe, 'hspice') +
               [spice_file_path, '-o', spice_file_path],
               [spice_file_path + '.lis'])
    if not os.path.exists(spice_file_path + '.lis'):
        raise RuntimeError(
            f"HSPICE finished without error but wrote no listing ({spice_file_path}.lis).")

    time, result, block_id = None, None, 0
    unit_map = {'time': 's', 'current': 'A', 'voltage': 'v', 'param': 'w'} # we will use parameter in Hspice to calcualte power

    with open(spice_file_path + '.lis', 'r') as f:
        parsing_data = False
        unit = None
        cur_time, cur_res = [], []
        for line in f.readlines():
            if line.strip() == 'x':
                parsing_data = True
                block_id += 1
                continue
            if line.strip() == 'y':
                parsing_data = False
                if block_id == 1:
                    time, result = np.array(cur_time), np.array(cur_res)
                else:
                    if np.sum(np.abs(time - np.array(cur_time))) != 0: raise RuntimeError(f"Spice time grids for different blocks are different")
                    result = np.concatenate((result, np.array(cur_res)), axis=1)
                cur_time, cur_res, unit = [], [], None
                continue
            if parsing_data and line.strip().startswith('time'):
                unit = [unit_map[key] for key in line.split()]
            if parsing_data:
                line = line.strip()
                if line and line[0].isnumeric():
                    vals = [convert(val + unit[i]) for i, val in enumerate(line.split())]
                    cur_time.append(vals[0])
                    cur_res.append(vals[1:])

    result_interp = interp1d_warp(time, result, t_value)  # (time_eout, dim_eout)

    if not keep_spice:
        os.remove(spice_file_path + '.lis')

    return result_interp, result


class SimulateHspice(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                t_value: torch.Tensor,
                param_value: torch.Tensor,
                monitor_power: Tuple[bool],
                num_out: int,
                prob_res: List,
                spice_file_path: str,
                param_file_path: str,
                spice_exe: str,
                keep_spice: Optional[bool] = False,
                sens_names: Tuple[str, ...] = (),
                owner: Optional['Electronic'] = None,
                *sens_tensors: torch.Tensor,
                ):
        # t_value shape (time_ein)
        # param_value shape (time_ein, dim_ein)

        result_interp, result = _run_hspice(t_value, param_value, spice_file_path,
                                            param_file_path, spice_exe, bool(keep_spice))

        ctx.spice_file_path = spice_file_path
        ctx.time_eout, ctx.dim_eout = result.shape[0], result.shape[1]
        ctx.time_ein, ctx.dim_ein = param_value.shape[0], param_value.shape[1]
        ctx.keep_spice = keep_spice
        ctx.t_value = t_value
        ctx.num_out = num_out
        ctx.sens_names = tuple(sens_names)
        ctx.owner = owner
        ctx.param_value = param_value.detach().clone()
        ctx.objective = result_interp[:, :num_out].detach()
        ctx.spice_exe = spice_exe
        ctx.param_file_path = param_file_path

        return (result_interp[:, :num_out],
                {prob_res[i]: result_interp[:, num_out + i] for i in range(len(prob_res))},
                {'electronic': result_interp[:, -3] if monitor_power[0] else None,
                 'pd_equiv': result_interp[:, -2] if monitor_power[1] else None,
                 'mod_equiv': result_interp[:, -1] if monitor_power[2] else None})

    @staticmethod
    def backward(ctx, *grad_output):
        """Device-parameter gradients by **central finite differences over re-runs**.

        HSPICE has no transient sensitivity analysis SPIPE can drive, so there is nothing to
        read out of the listing.  What is implemented here is the honest alternative: for
        each parameter declared with ``.sensparam`` the deck is re-rendered with
        ``W +- h`` and HSPICE is run again, twice.

        **This is expensive.**  A gradient with respect to *p* device parameters costs
        ``2 * p`` extra HSPICE transients *per backward pass*, and a fixed-point co-simulation
        calls backward once per iteration.  On the Xyce back end the same gradient costs one
        extra linear solve per parameter inside a single run, and on the native back end
        (:mod:`spipe.electronic.native`) it costs one adjoint sweep **regardless** of the
        number of parameters and is exact rather than differenced.  Use HSPICE here only
        when the model cards leave no choice.

        Gradients with respect to the **photocurrent** parameters are still not available on
        this back end -- HSPICE cannot supply ``d(node voltage)/d(PWL breakpoint)`` and SPIPE
        will not manufacture ``time_ein * pd_cnt`` finite differences to fake it.  Asking for
        one raises :class:`NotImplementedError` naming the back ends that can.
        """
        if ctx.needs_input_grad[1]:
            raise NotImplementedError(
                "The HSPICE back end cannot supply gradients with respect to the "
                "photocurrent parameters: HSPICE has no transient parameter sensitivity "
                "analysis, and there are time_ein * pd_cnt of them "
                f"({ctx.time_ein} * {ctx.dim_ein} = {ctx.time_ein * ctx.dim_ein}), so "
                "differencing them is not a serious option either. Use spice_exe='native' "
                "(SPIPE's own engine -- exact gradients, one adjoint sweep) or a Xyce "
                "command line (.SENS, direct method). Device parameters declared with "
                "'.sensparam' ARE differentiable on this back end, by finite differences.")

        if not ctx.sens_names:
            return (None,) * 11

        owner = ctx.owner
        if owner is None:                                     # pragma: no cover - defensive
            raise RuntimeError("the HSPICE back end lost its Electronic owner")

        cotangent = grad_output[0]
        if cotangent is None:
            cotangent = torch.zeros_like(ctx.objective)
        cotangent = cotangent.to(dtype=torch.float64)

        relative = float(config.get('fd_step', 1e-3))
        warnings.warn(
            f"Computing d(objective)/d({', '.join(ctx.sens_names)}) on the HSPICE back end "
            f"by central finite differences: {2 * len(ctx.sens_names)} extra HSPICE "
            f"transient runs for this one backward pass. The Xyce and native back ends do "
            f"this analytically; see SimulateHspice.backward.", RuntimeWarning, stacklevel=2)

        device_grads = []
        for name, (device, parameter) in zip(ctx.sens_names, owner.sens_declared):
            tensor = owner.sens_values[(device.lower(), parameter)]
            nominal = float(tensor.detach())
            step = relative * abs(nominal) if nominal != 0.0 else relative
            samples = []
            for shifted in (nominal + step, nominal - step):
                with torch.no_grad():
                    tensor.fill_(shifted)
                owner._write_spice_deck()
                result, _ = _run_hspice(ctx.t_value, ctx.param_value, ctx.spice_file_path,
                                        ctx.param_file_path, ctx.spice_exe, ctx.keep_spice)
                samples.append(result[:, :ctx.num_out].detach().to(torch.float64))
            with torch.no_grad():
                tensor.fill_(nominal)
            owner._write_spice_deck()

            block = (samples[0] - samples[1]) / (2.0 * step)
            guard_all_zero(block, [name], ctx.objective, 'finite-difference', 'hspice',
                           objective_name='the modulator drive voltage')
            device_grads.append((cotangent * block).sum())

        return (None,) * 11 + tuple(device_grads)