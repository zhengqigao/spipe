import torch
from typing import List, Union, Dict, Tuple, Optional
from spipe.utils import extract, convert
import warnings
from collections import defaultdict
from .model import _json_path_, _extra_model_
import json
import importlib
from spipe import config
import itertools
from .model import PDArray
from .func import FreeLightSpeed
from math import pi
import time

try:  # SuperLU is only needed by the sparse back end; SPIPE still runs without SciPy.
    import numpy as _np
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    _HAS_SPARSE = True
except ImportError:  # pragma: no cover - depends on the installation
    _HAS_SPARSE = False

#: ``2 * num_node`` at or above which the sparse (SuperLU) back end is preferred over the
#: dense batched LAPACK path.  Measured on a passive pbum mesh (CPU, complex128, one
#: frequency, ``T = 1``), wall clock of the whole ``Photonic.simulate`` call::
#:
#:     mesh      2N    dense    sparse
#:     6x6      672    0.28 s   0.23 s
#:     8x8     1152    0.43 s   0.39 s
#:     11x11   2112    0.92 s   0.73 s
#:     14x14   3360    1.88 s   1.09 s
#:     16x16   4352    2.29 s   1.68 s
#:     30x30  14880   23.06 s   4.70 s      (dense peak 7.3 GB, sparse 0.44 GB)
#:
#: Sparse is never slower here, but the margin only becomes worth a behaviour change well
#: above 1000, and below the threshold the dense path factorises all ``(t, omega)`` systems
#: in one batched LAPACK call while the sparse path pays one Python-level SuperLU call per
#: pair.  Override the choice with
#: ``spipe.config['photonic_solver'] = 'dense'`` / ``'sparse'`` / ``'auto'``.
SPARSE_DIM_THRESHOLD = 2048

#: A moderately sized system that is swept over many frequencies (or many time points)
#: still blows up as a dense ``(T, F, 2N, 2N)`` batch -- an 18x18 mesh over the 100
#: frequency points of the paper's mesh deck is 120 GB dense but a few MB sparse.  Above this many
#: bytes the sparse back end is used even when ``2N`` is below
#: :data:`SPARSE_DIM_THRESHOLD`, provided the system is at least :data:`SPARSE_MIN_DIM`
#: wide.  Measured on the same mesh at the 100 frequency points that deck uses::
#:
#:     mesh      2N   dense batch       dense              sparse
#:     6x6      672      0.72 GB     0.88 s /  1853 MB   0.40 s /  404 MB
#:     9x9     1440      3.32 GB     4.51 s /  6922 MB   0.90 s /  424 MB
#:     12x12   2496      9.97 GB    17.46 s / 19780 MB   1.50 s /  434 MB
#:
#: Below ``SPARSE_MIN_DIM`` the dense batch is so much faster per system (one LAPACK call
#: for all of them, against one Python-level SuperLU call each) that it is worth the
#: memory -- on a 2N = 128 modulated circuit with T = 400, F = 4 the sparse path is 4.6x
#: slower.  Override with ``spipe.config['photonic_dense_budget']``.
DENSE_MEMORY_BUDGET = 256 * 1024 ** 2
SPARSE_MIN_DIM = 512

#: Number of time points whose modulator Jacobian is evaluated in one call to
#: :meth:`Device.transfer`.  ``transfer`` returns the full ``(T, T, F, p, p)`` Jacobian
#: ``dS(t_j)/dact(t_i)``, which is quadratic in ``T`` in both work and memory even though
#: it is block-diagonal in the two time axes (paper Assumption 1).  Evaluating the drive in
#: chunks of ``c`` makes the cost ``O(T * c)`` and bounds the memory independently of
#: ``T``; the values are bit-for-bit identical (measured: ``reldiff = 0`` over
#: ``c = 8 .. T``).  Small ``c`` minimises the flops but pays a per-call overhead, so the
#: best value is a compromise -- backward wall clock on a 16-modulator circuit at
#: ``T = 400``::
#:
#:     c            8     16     32     64    128    256    400
#:     F = 1     7.16s  2.72s  1.76s  1.55s  1.52s  1.64s  1.83s
#:     F = 4     5.32s  3.04s  2.53s  2.29s  2.60s  3.16s  4.92s
#:
#: :data:`JAC_CHUNK_BYTES` caps ``c`` further when ``F`` or the port count is large.
#: Override with ``spipe.config['photonic_jac_chunk']``.
JAC_TIME_CHUNK = 64
JAC_CHUNK_BYTES = 64 * 1024 ** 2


def _jacobian_chunk(num_time: int, num_freq: int, num_port: int) -> int:
    """How many time points of a modulator Jacobian to evaluate at once."""
    override = config.get('photonic_jac_chunk')
    if override:
        return max(1, min(int(override), num_time))
    itemsize = torch.empty(0, dtype=config['complex_dtype']).element_size()
    per_pair = max(1, num_freq * num_port * num_port * itemsize)
    budget = config.get('photonic_jac_bytes', JAC_CHUNK_BYTES)
    chunk = min(JAC_TIME_CHUNK, int((budget / per_pair) ** 0.5))
    return max(1, min(chunk, num_time))


def _solver_backend(dim: int, num_time: int, num_freq: int) -> str:
    """Pick the linear-solver back end for ``num_time * num_freq`` systems of size ``dim``.

    ``dim`` is ``2 * num_node``.  See :data:`SPARSE_DIM_THRESHOLD` and
    :data:`DENSE_MEMORY_BUDGET` for the two criteria and the measurements behind them.
    """
    mode = config.get('photonic_solver', 'auto')
    if mode == 'dense':
        return 'dense'
    if mode == 'sparse':
        if not _HAS_SPARSE:
            raise RuntimeError("config['photonic_solver'] == 'sparse' but SciPy is not installed.")
        return 'sparse'
    if mode != 'auto':
        raise ValueError(f"config['photonic_solver'] must be 'auto', 'dense' or 'sparse', got {mode!r}.")
    if not _HAS_SPARSE:
        return 'dense'
    if dim >= SPARSE_DIM_THRESHOLD:
        return 'sparse'
    budget = config.get('photonic_dense_budget', DENSE_MEMORY_BUDGET)
    itemsize = torch.empty(0, dtype=config['complex_dtype']).element_size()
    if dim >= SPARSE_MIN_DIM and num_time * num_freq * dim * dim * itemsize > budget:
        return 'sparse'
    return 'dense'


class _DenseLU:
    """A batched dense LU factorisation of the ``(T, F, n, n)`` system matrix.

    Replaces the explicit matrix inverse the adjoint used to build (SPEC-P2.2).  A
    factorisation is both cheaper and better conditioned than an inverse, and the same
    factors serve the forward solve and every adjoint right-hand side.

    ``A`` is overwritten by its own factors (LAPACK works in place anyway), so keeping the
    system factorised costs no more memory than holding ``A`` did.  The caller must not
    use ``A`` afterwards.
    """

    def __init__(self, A: torch.Tensor):
        self.batch = tuple(A.shape[:-2])
        self.dim = A.shape[-1]
        piv = torch.empty((*self.batch, self.dim), dtype=torch.int32, device=A.device)
        info = torch.empty(self.batch, dtype=torch.int32, device=A.device)
        self.LU, self.piv, info = torch.linalg.lu_factor_ex(A, out=(A, piv, info))
        if int(info.abs().max()) != 0:
            raise RuntimeError(
                "The photonic system matrix is singular: the circuit is under-determined "
                "(check for dangling nodes, duplicated sources or a device with no path to a "
                "source or a detector).")

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """``A x = b``."""
        return torch.linalg.lu_solve(self.LU, self.piv, b.expand(*self.batch, self.dim, b.shape[-1]))

    def solve_adjoint(self, b: torch.Tensor) -> torch.Tensor:
        """``A^H x = b`` -- i.e. apply ``A^{-H}``, which is what the adjoint needs."""
        return torch.linalg.lu_solve(self.LU, self.piv,
                                     b.expand(*self.batch, self.dim, b.shape[-1]), adjoint=True)


class _SparseLU:
    """SuperLU factorisations of the system matrix, one per ``(time, omega)`` pair.

    The matrix is never materialised densely: each row encodes one device-port constraint
    or one boundary condition, so ``A`` carries only about three non-zeros per row whatever
    the circuit size (SPEC-P2.3).  Factorisation happens straight from the COO triplets.

    SuperLU works in double precision, so the factorisation is always carried out in
    ``complex128`` and the solution cast back to ``config['complex_dtype']`` afterwards.

    ``keep`` decides whether the factors are retained.  The adjoint needs them a second
    time, so a gradient run keeps all ``T * F`` of them; a plain forward run factorises and
    discards one system at a time, which keeps its memory at a single factorisation.
    """

    def __init__(self, lin_index: torch.Tensor, values: torch.Tensor, dim: int,
                 num_time: int, num_freq: int, keep: bool = True):
        if not _HAS_SPARSE:  # pragma: no cover - guarded by _solver_backend
            raise RuntimeError("The sparse photonic back end needs SciPy.")
        self.dim, self.batch = dim, (num_time, num_freq)
        self._rows = (lin_index // dim).cpu().numpy()
        self._cols = (lin_index % dim).cpu().numpy()
        vals = values.expand(num_time, num_freq, values.shape[-1])
        self._vals = vals.reshape(-1, values.shape[-1]).detach().cpu().numpy().astype(_np.complex128)
        if keep:
            self._lu = [self._factor(i) for i in range(self._vals.shape[0])]
            self._vals = None            # the factors supersede the entries
        else:
            self._lu = None

    def _factor(self, i: int):
        return _spla.splu(_sp.csc_matrix((self._vals[i], (self._rows, self._cols)),
                                         shape=(self.dim, self.dim)))

    def _apply(self, b: torch.Tensor, trans: str) -> torch.Tensor:
        cols = b.shape[-1]
        rhs = b.expand(*self.batch, self.dim, cols).reshape(-1, self.dim, cols)
        rhs = rhs.detach().cpu().numpy().astype(_np.complex128)
        out = _np.empty_like(rhs)
        for i in range(rhs.shape[0]):
            lu = self._lu[i] if self._lu is not None else self._factor(i)
            out[i] = lu.solve(rhs[i], trans=trans)
        res = torch.from_numpy(out).to(device=config['device'], dtype=config['complex_dtype'])
        return res.reshape(*self.batch, self.dim, cols)

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        """``A x = b``."""
        return self._apply(b, 'N')

    def solve_adjoint(self, b: torch.Tensor) -> torch.Tensor:
        """``A^H x = b``."""
        return self._apply(b, 'H')


def _modulator_jacobian(class_, base_kwargs: Dict, t_value: torch.Tensor, act: torch.Tensor,
                        num_freq: int, num_port: int) -> torch.Tensor:
    """``dS(t_i) / d act(t_i)``, the *time-diagonal* of the modulator Jacobian.

    :meth:`Device.transfer` hands back the full ``(T, T, F, p, p)`` Jacobian
    ``d S(t_j) / d act(t_i)``.  Under the paper's Assumption 1 an active device's scatter
    matrix at ``t_j`` depends only on its drive at ``t_j``, so every ``i != j`` block is
    identically zero and all the information sits on the ``i == j`` diagonal.  Building the
    whole thing is quadratic in ``T`` in both time and memory, so the drive is split into
    chunks (see :func:`_jacobian_chunk`) and only the diagonal of each chunk is kept: cost
    ``O(T * chunk)`` instead of ``O(T^2)``, with the values bit-for-bit the same.

    :return: a tensor of shape ``(T, F, p, p)``.
    """
    num_time = int(t_value.shape[0])
    chunk = _jacobian_chunk(num_time, num_freq, num_port)

    pieces = []
    for start in range(0, num_time, chunk):
        stop = min(start + chunk, num_time)
        span = stop - start
        instance = class_(**{**base_kwargs, 'time': t_value[start:stop], 'act': act[start:stop]})
        transfer_matrix, jac = instance.transfer(['act'])
        jac_act = jac['act']

        if jac_act.ndim == transfer_matrix.ndim + 1 and jac_act.shape[:2] == (span, span):
            # the (d act(t_i), d S(t_j)) layout produced by Device._autodiff
            diag = torch.arange(span, device=jac_act.device)
            jac_act = jac_act[diag, diag]
        elif jac_act.shape != transfer_matrix.shape:
            raise RuntimeError(
                f"Model '{class_.__name__}' returned a Jacobian of shape {tuple(jac_act.shape)} for a "
                f"scatter matrix of shape {tuple(transfer_matrix.shape)}; expected either "
                f"{(span, *transfer_matrix.shape)} (one row per drive sample) or "
                f"{tuple(transfer_matrix.shape)} (already reduced to the time diagonal).")
        pieces.append(jac_act)

    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _preprocess_mode(neff, ng=None, wl=1550e-9):
    # NOTE: the positional form of `.mode` (e.g. ".mode 2.35 4.0 wl=1550e-9") delivers *strings*,
    # because `extract(..., convert_numeric=True)` only converts the key=value pairs.  All three
    # values must therefore be cast; forgetting `ng` used to leave a str in the parameter dict,
    # which blew up much later with "unsupported operand type(s) for -: 'Parameter' and 'str'".
    neff = float(neff)
    wl = float(wl)

    ng = neff if ng is None else float(ng)
    return neff, ng, wl


def _preprocess_freq(start, stop, steps):
    return 2 * pi * torch.linspace(float(start), float(stop), steps=int(steps), dtype = config['real_dtype'])

def _match(given_str, target_dict):
    for key, value in target_dict.items():
        if given_str.startswith(key):
            return value
    return None


def _entry_is_active(entry: Optional[Dict]) -> bool:
    """Whether a ``model.json`` entry describes an electrically driven (active) device.

    The test is the model's own declared ``active_port`` count, i.e. how many electrical
    drive nodes it exposes -- **not** its name.  SPIPE used to decide this with
    ``name.startswith('mod')``, which silently demoted any active model whose name did not
    happen to begin with ``mod`` (``mzm``, say) to a passive element frozen at its bias
    point: it parsed, it solved, and it simply never saw its drive.  ``modm`` and ``modp``
    declare ``active_port = 1`` so the two predicates agree on them exactly.
    """
    if not entry:
        return False
    try:
        return int(entry.get('active_port') or 0) > 0
    except (TypeError, ValueError):
        return False


def is_active_device(name: str) -> bool:
    """Whether the netlist element ``name`` refers to an active photonic device.

    Resolves ``name`` against the model table in force (built-ins plus anything registered
    through :func:`spipe.photonic_register`) and reports whether that model declares an
    electrical drive port.  This is the single predicate the photonic parser, the solver and
    the electronic netlist writer all share, so a new active model becomes drivable purely by
    declaring ``_active_port``.
    """
    return _entry_is_active(_match(name, _model_info()))


inward_map = lambda index: 2 * index
outward_map = lambda index: 2 * index + 1

module = importlib.import_module("spipe.photonic.model")

with open(_json_path_, 'r') as f:
    model_info = json.load(f)


def _model_info() -> Dict:
    """The model table actually in force: the built-in models plus everything registered via
    :func:`spipe.photonic_register`.

    ``model_info`` is read from ``model.json`` once, at import time, so a model registered later
    in the same session would otherwise be invisible to the parser and to the solver.  Models in
    ``_extra_model_`` always win over a stale JSON entry of the same name.
    """
    if not _extra_model_:
        return model_info
    info = dict(model_info)
    for name, class_ in _extra_model_.items():
        info[name] = class_._collect_info()
    return info


def _model_class(key: str, value: Dict):
    """Resolve the python class implementing the model named ``key``."""
    if key in _extra_model_:
        return _extra_model_[key]
    return getattr(module, value['class_name'])


def _device_length(attr: Dict, class_) -> float:
    """Physical propagation length (in meters) of a single device instance.

    Every device attribute called ``l`` or ending in ``_l`` is a length; they are summed because
    such sections are cascaded.  The exception is an interferometric device, recognised by its
    declaring *both* ``wgu_l`` and ``wgl_l``: those are the two *parallel* arms of the
    interferometer, so the slower one is taken rather than their sum.  (``modm`` and ``mzm``
    are the models that do; the rule used to be spelled ``_name.startswith('modm')``.)
    """
    params = dict(class_._optional_attr)
    params.update({k: v for k, v in attr.items() if k not in ('ln', 'rn', 'an')})

    def _f(key):
        value = params.get(key, 0.0)
        try:
            return abs(float(value))
        except (TypeError, ValueError):
            return 0.0

    if 'wgu_l' in class_._optional_attr and 'wgl_l' in class_._optional_attr:
        return _f('act_l') + max(_f('wgu_l'), _f('wgl_l'))

    return sum(_f(k) for k in params.keys() if k == 'l' or k.endswith('_l'))


def _longest_simple_path(adjacency: Dict, budget: int = 50000, max_depth: int = 512) -> float:
    """Longest simple (loop-free) path of a weighted undirected graph, as a *lower bound*.

    The exact problem is NP-hard, so the depth-first search is stopped after ``budget`` node
    expansions (or ``max_depth`` hops).  Whatever it found by then is a genuine simple path, hence
    a safe lower bound on the true maximum.  For small, loop-free circuits the search is
    exhaustive and the answer is exact.
    """
    best = 0.0
    expansions = [0]
    visited = set()

    def dfs(node, acc, depth):
        nonlocal best
        expansions[0] += 1
        if acc > best:
            best = acc
        if expansions[0] >= budget or depth >= max_depth:
            return
        for nxt, w in adjacency[node]:
            if nxt in visited:
                continue
            visited.add(nxt)
            dfs(nxt, acc + w, depth + 1)
            visited.discard(nxt)

    for start in list(adjacency.keys()):
        if expansions[0] >= budget:
            break
        visited = {start}
        dfs(start, 0.0, 0)

    return best


class Photonic(object):
    def __init__(self,
                 p_content: List[str],
                 need_grads: Optional[bool] = False):

        self.p_content = p_content
        self.need_grads = need_grads

        self.circuit_element = dict()
        self.mod_element = dict()

        self.mode_info = dict()
        self.omega = None
        self.srce_node = {}
        self.dout_node = []
        self.middle_node = []
        self.pd_args = []
        self.occur_order = {}

        self.node2ind = {}
        self.ind2node = {}
        self.node_has_ele = {}
        self.inward_node = {}
        self.outward_node = {}
        self.laser_info = {}
        self._max_group_delay = None
        self._quasistatic_checked = False

        self.preprocess()

        self.pd_array = PDArray(self.pd_args, self.omega)

    def preprocess(self) -> None:

        cnt = 0
        model_table = _model_info()
        for line in self.p_content:
            line = line.split('#')[0].strip()
            if not line: continue

            initial, strings, kv_pair = extract(line, convert_numeric=True)


            for k, v in model_table.items():
                if initial.startswith(k):
                    ln, rn = v['num_port']
                    an = v['active_port']

                    # active iff the model declares an electrical drive port -- see
                    # _entry_is_active(); the old test was initial.startswith('mod')
                    if _entry_is_active(v):
                        if initial in self.mod_element.keys():
                            raise RuntimeError(f"Device '{initial}' occurs more than once.")

                        self.mod_element[initial] = {'ln': strings[:ln],
                                                     'rn': strings[ln:ln + rn],
                                                     'an': strings[ln + rn:ln + rn + an],
                                                     **kv_pair}
                    else:
                        if initial in self.circuit_element.keys():
                            raise RuntimeError(f"Device '{initial}' occurs more than once.")

                        self.circuit_element[initial] = {'ln': strings[:ln],
                                                         'rn': strings[ln:ln + rn],
                                                         **kv_pair}

            if initial.lower() == '.mode':
                self.mode_info['neff'], self.mode_info['ng'], self.mode_info['wl'] = _preprocess_mode(*strings,
                                                                                                      **kv_pair)

            if initial.lower() == '.freq':
                self.omega = _preprocess_freq(*[convert(string) for string in strings], **kv_pair)

            if initial.lower() == '.source':

                for s in strings:
                    value, node = s.split('@')
                    if node not in self.srce_node.keys():
                        self.srce_node[node] = complex(value)
                    else:
                        warnings.warn(f"Node {node} has been assigned source twice.")
                self.laser_info['power'] = kv_pair.get('power', None)
                self.laser_info['eff'] = kv_pair.get('eff', None)

            if initial.lower().startswith('pd'):
                self.dout_node.append(strings[0])
                self.pd_args.append(kv_pair)

            if initial.lower().startswith('.prob'):
                self.middle_node.extend(strings)

            # The drive column each active device reads from `param_value`.  Assigned in
            # netlist order, and gated by exactly the same predicate that fills
            # `self.mod_element`, so `occur_order` and `mod_element` always hold the same set.
            if _entry_is_active(_match(initial, model_table)):
                self.occur_order[initial] = cnt
                cnt += 1

        # final preprocessing
        for k, v in self.srce_node.items():
            self.srce_node[k] = torch.ones(self.omega.size(), dtype=config['complex_dtype'],
                                           device=config['device']) * v

        self.node2ind, self.ind2node, self.node_has_ele, self.inward_node, self.outward_node = self._node_preprocess()

        return

    def max_group_delay(self) -> float:
        """Estimate the largest optical group delay ``tau = sum(ng * L / c)`` in the network.

        What is computed: the network is turned into an undirected graph whose vertices are the
        optical nodes and whose edges are the devices (every left port is connected to every right
        port of the same device).  Each edge is weighted with the device group delay
        ``ng * L / c``, where ``L`` is the propagation length of the device
        (see :func:`_device_length`) and ``ng`` its group index (per-device ``ng=`` attribute if
        present, otherwise the ``.mode`` value).  The returned number is the weight of the longest
        *simple* path found by a budget-limited depth-first search.

        Because the longest-simple-path problem is NP-hard, and because a circuit with loops has
        no well-defined "longest path" at all, the search is truncated
        (see :func:`_longest_simple_path`).  The result is therefore a **lower bound** on the true
        maximum group delay: the quasi-static check built on it can under-warn but never
        over-warn.

        :return: the estimated maximum group delay, in seconds.
        """
        if self._max_group_delay is not None:
            return self._max_group_delay

        model_table = _model_info()
        default_ng = self.mode_info.get('ng', self.mode_info.get('neff', 1.0))

        adjacency = defaultdict(list)
        for ele, attr in itertools.chain(self.circuit_element.items(), self.mod_element.items()):
            entry = _match(ele, model_table)
            if entry is None:
                continue
            class_ = _model_class(entry['model_name'], entry)

            length = _device_length(attr, class_)
            try:
                ng = float(attr.get('ng', default_ng))
            except (TypeError, ValueError):
                ng = 1.0
            delay = ng * length / FreeLightSpeed

            for left in attr['ln']:
                for right in attr['rn']:
                    adjacency[left].append((right, delay))
                    adjacency[right].append((left, delay))

        self._max_group_delay = _longest_simple_path(adjacency) if adjacency else 0.0
        return self._max_group_delay

    def check_quasistatic(self, dt: float) -> Optional[float]:
        """Warn if the steady-state (quasi-static) photonic solve is no longer justified.

        SPIPE solves the photonic network in steady state at every time sample, i.e. it assumes
        the optical network settles instantaneously compared with the modulation timescale.  That
        holds only while the optical transit / round-trip time is far smaller than the transient
        time step ``dt``.  This helper compares the estimated maximum group delay with ``dt`` and
        emits a :func:`warnings.warn` (never an exception) when
        ``max_group_delay > 0.1 * dt``.

        Disable it with ``spipe.config['quasistatic_check'] = False``.

        :param dt: the transient time step, in seconds.
        :return: the estimated maximum group delay, or ``None`` if the check was skipped.
        """
        self._quasistatic_checked = True

        if not config.get('quasistatic_check', True):
            return None

        try:
            dt = float(dt)
        except (TypeError, ValueError):
            return None

        if not (dt > 0) or dt != dt:  # non-positive or NaN
            return None

        tau = self.max_group_delay()

        if tau > 0.1 * dt:
            warnings.warn(
                f"Quasi-static assumption is being stretched: the estimated maximum optical group "
                f"delay of the photonic network is {tau:.6g} s, which is more than 10% of the "
                f"transient time step dt = {dt:.6g} s (max_group_delay / dt = {tau / dt:.6g}). "
                f"SPIPE solves the photonic network in steady state at every time sample, which "
                f"assumes the optical network settles instantaneously, i.e. "
                f"max_group_delay << dt. Run with simulate(..., mode='envelope') to keep the "
                f"optical memory (the passive network's impulse response is convolved with the "
                f"modulated field instead of being collapsed to its DC value), or shorten the "
                f"optical paths / increase the time step. "
                f"Set spipe.config['quasistatic_check'] = False to silence this check.",
                stacklevel=2)

        return tau

    def simulate(self, t_value: Optional[torch.tensor] = None,
                 param_value: Optional[torch.tensor] = None,
                 mode: str = 'quasistatic') -> Tuple[torch.Tensor, Dict, Dict]:
        # t_value: (time_pin)
        # param_value: (time_pin, dim_pin)
        #
        # ``mode`` selects how the photonic network is propagated in time (SPEC-P3):
        #   'quasistatic' (default, unchanged) -- the network is solved in steady state at every
        #        time sample, i.e. all optical transit and round-trip times are taken to be zero.
        #   'envelope' -- the passive sub-network keeps its optical memory: its transfer over the
        #        `.freq` band is inverse-Fourier-transformed into an impulse response and convolved
        #        with the modulated field, so a detector at time t sees the modulator state at the
        #        retarded time t - tau.  See spipe.photonic.envelope.
        if mode not in ('quasistatic', 'envelope'):
            raise ValueError(f"Photonic.simulate(mode=...) must be 'quasistatic' or 'envelope', "
                             f"got {mode!r}.")

        if param_value is not None and len(self.mod_element.keys()) == 0:
            warnings.warn(f"param_value will be ignored because Modulators are not defined in the netlist.")
            t_value, param_value = None, None
        if (param_value is None) and len(self.mod_element.keys()):
            raise RuntimeError(f"param_value is not provided but Modulators are defined in the netlist")

        if (t_value is None) != (param_value is None):
            raise RuntimeError("Both t_value and param_value must be None or provided at the same time.")

        if t_value is not None:
            if t_value.ndim != 1:
                raise RuntimeError(f"t_value must be a 1D tensor, but got shape {t_value.shape}")

            if param_value.ndim != 2:
                raise RuntimeError(f"param_value must be a 2D tensor, but got shape {param_value.shape}")

            expected_shape = (len(t_value), len(self.occur_order))
            if param_value.shape != expected_shape:
                raise RuntimeError(f"Expected param_value of shape {expected_shape}, but got {param_value.shape}")

            if not self._quasistatic_checked and len(t_value) > 1 and mode == 'quasistatic':
                self.check_quasistatic(float(t_value[1] - t_value[0]))

        if mode == 'envelope':
            from .envelope import simulate_envelope          # local: envelope.py imports this module
            res, middle = simulate_envelope(self, t_value, param_value)
            power = self._power_report(res)
            # Hand PDArray the transient grid: both the coherent sum and the bw= low-pass
            # need it, and without it each silently degrades (to the incoherent sum and to
            # no low-pass respectively) behind a warning.
            return self.pd_array(res, t_value), middle, power

        res, middle = Simulate.apply(t_value, param_value, self.omega, self.node_has_ele,
                                        self.srce_node,
                                        self.node2ind,
                                        self.circuit_element,
                                        self.mod_element,
                                        self.mode_info,
                                        self.occur_order,
                                        (self.dout_node, self.middle_node),
                                        self.inward_node,
                                        self.outward_node,
                                        self.need_grads
                                        )  # res shape (time_pout, len(omega), dim_pout), but here time_pout = time_pin
        power = self._power_report(res)

        res = self.pd_array(res, t_value)  # (tim_pout, dim_pout)
        return res, middle, power

    def _power_report(self, res: torch.Tensor) -> Dict:
        """Optical power budget of the laser source, in watts.

        ``.source`` carries two laser attributes: ``power`` -- the *electrical* power the laser
        draws, expressed per unit of ``sum |A|^2`` of the declared source amplitudes -- and
        ``eff``, the laser wall-plug efficiency (optical out / electrical in).

        Returned keys (all values are tensors of shape ``(len(time),)``, all in **watts**):

        ``'photonic'``
            The power actually consumed by the photonic front-end, i.e. the electrical wall-plug
            power drawn by the laser.  Identical to ``'laser_drive_power'``; kept under this name
            because it is the key the rest of SPIPE aggregates.
        ``'laser_drive_power'``
            Electrical power drawn by the laser, ``power * sum|A|^2``.
        ``'optical_input_power'``
            Optical power launched into the circuit, ``laser_drive_power * eff``.
        ``'optical_output_power'``
            Optical power reaching the photo detectors, ``optical_input_power * T`` where
            ``T = sum|out|^2 / sum|A|^2`` is the (dimensionless) power transmission of the
            network at that time sample.
        ``'optical_loss'``
            ``optical_input_power - optical_output_power``, the optical power dissipated inside
            the circuit.

        Because the transmission enters as a *ratio* of the two unit sums, the report is invariant
        to how the source amplitudes are normalised: scaling every source amplitude so that
        ``sum|A|^2`` grows by ``s`` and dividing ``power`` by ``s`` leaves every reported number
        unchanged.  And since ``T <= 1`` for a passive network, the optical output power can never
        exceed the optical input power.
        """
        if self.laser_info.get('power') is None or self.laser_info.get('eff') is None:
            return {'photonic': None}

        # sum |A|^2 over all sources and all frequency channels: the arbitrary "source unit" in
        # which `power` is expressed.
        input_unit = 0.0
        for source in self.srce_node.values():
            input_unit = input_unit + (source.abs() ** 2).reshape(len(self.omega)).sum()

        # sum |out|^2 over all frequency channels and all detectors, per time sample.
        output_unit = (res.abs() ** 2).sum(dim=[1, 2])

        ones = torch.ones_like(output_unit)

        laser_drive_power = self.laser_info['power'] * input_unit * ones
        optical_input_power = laser_drive_power * self.laser_info['eff']

        transmission = output_unit / input_unit if input_unit != 0 else torch.zeros_like(output_unit)
        optical_output_power = optical_input_power * transmission
        optical_loss = optical_input_power - optical_output_power

        return {'photonic': laser_drive_power,           # W, consumed (electrical wall-plug)
                'laser_drive_power': laser_drive_power,  # W, electrical
                'optical_input_power': optical_input_power,    # W, optical
                'optical_output_power': optical_output_power,  # W, optical
                'optical_loss': optical_loss}                  # W, optical

    def _node_preprocess(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:

        total_index, node2ind, ind2node, node_has_ele = 0, {}, {}, {}
        inward_node, outward_node = {}, {}
        model_table = _model_info()

        for ele, attr in itertools.chain(self.circuit_element.items(), self.mod_element.items()):
            assert _match(ele, model_table), f"The model for `{ele}` is not defined."
            assert [len(attr['ln']), len(attr['rn'])] == _match(ele, model_table)['num_port'], \
                f"The numbers of left and right ports of `{ele}` are not correct."

            # update node2ind, ind2node, node_has_ele
            for node in (attr['ln'] + attr['rn']):
                if node not in node2ind:
                    node2ind[node] = total_index
                    ind2node[total_index] = node
                    total_index += 1
                if node not in node_has_ele.keys():
                    node_has_ele[node] = [ele]
                elif node in self.srce_node.keys():
                    raise RuntimeError(f"The node {node} connects to {node_has_ele[node][0]}, {ele}, and a source")
                else:
                    node_has_ele[node] += [ele]
                # node_has_ele[node] = node_has_ele.get(node, []) + [ele]

            # update inward_node, outward_node
            inward_node[ele], outward_node[ele] = {'ln': [], 'rn': []}, {'ln': [], 'rn': []}
            for string, ele_nodes in zip(['ln', 'rn'], [attr['ln'], attr['rn']]):
                for node in ele_nodes:
                    if node_has_ele[node][0] == ele:
                        inward_node[ele][string].append(inward_map(node2ind[node]))
                        outward_node[ele][string].append(outward_map(node2ind[node]))
                    elif node_has_ele[node][1] == ele:
                        inward_node[ele][string].append(outward_map(node2ind[node]))
                        outward_node[ele][string].append(inward_map(node2ind[node]))
                    else:
                        raise RuntimeError(f"The node {node} has more than two circuit elements connected.")
        return node2ind, ind2node, node_has_ele, inward_node, outward_node


class Simulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t_value: Union[torch.Tensor, None],
                param_value: Union[torch.Tensor, None],
                omega: torch.Tensor,
                node_has_ele: Dict,
                srce_node: Dict,
                node2ind: Dict,
                circuit_element: Dict,
                mod_element: Dict,
                mode_info: Dict,
                occur_order: List,
                node_tuple: Tuple,
                inward_node: Dict,
                outward_node: Dict,
                need_grads: bool):
        try:
            dout_node, prob_node = node_tuple
            # On the configured device: backward index_add_s CUDA tensors with this, and a CPU
            # index there raised 'Expected all tensors to be on the same device' -- so every
            # gradient on a GPU failed, while the forward pass ran fine.
            detect_ind = torch.tensor([node2ind[node] for node in dout_node], dtype=torch.long,
                                      device=config['device'])
            prob_ind = torch.tensor([node2ind[node] for node in prob_node], dtype=torch.long)
        except KeyError:
            raise KeyError("At least one node required by photo detector and .prob syntax is not in the circuit.")

        if t_value is None and param_value is None:
            t_value = torch.tensor([0.0], dtype=config['real_dtype'], device=config['device'])

        line_counter, num_node = 0, len(node2ind.keys())
        dim = 2 * num_node
        num_time, num_freq = len(t_value), len(omega)
        cdtype, device = config['complex_dtype'], config['device']

        # The system matrix is assembled as COO triplets rather than written straight into a
        # dense buffer (SPEC-P2.3).  Each row is one device-port constraint or one boundary
        # condition, so `A` carries about three non-zeros per row whatever the circuit size,
        # and the triplets can feed either the dense or the sparse back end.  Entries are
        # split into a *static* group -- boundary rows plus every passive device row, which
        # are identical at every time point -- and a *dynamic* group holding the modulator
        # rows, the only ones that actually vary with t (paper Assumption 1).
        # The entries whose value is the constant +1 (boundary rows) or -1 (the outward-wave
        # column of a device row) are gathered as plain python indices and turned into one
        # tensor at the end, so assembly costs no per-device tensor allocation.
        unit_index, minus_unit_index = [], []
        static_index, static_value = [], []
        dynamic_index, dynamic_value = [], []

        # sources do not depend on time either, so `b` keeps a length-1 time axis
        b = torch.zeros([1, num_freq, dim, 1], dtype=cdtype, device=device)

        model_table = _model_info()

        for node, ele in node_has_ele.items():
            if len(ele) == 1:
                src_value = srce_node.get(node, torch.zeros(num_freq, dtype=cdtype, device=device))

                # source must be injected at floating node, and the inward direction takes the first place
                unit_index.append(line_counter * dim + inward_map(node2ind[node]))
                b[..., line_counter, 0] = src_value
                line_counter += 1

        if torch.all(b == 0): raise RuntimeError(
            f"Source is not correctly connected to circuit; simulation will trivially be all zeros.")

        for ele, attr in itertools.chain(sorted(circuit_element.items()), sorted(mod_element.items())):

            for key, value in model_table.items():
                if ele.startswith(key):
                    class_ = _model_class(key, value)

                    if _entry_is_active(value):
                        # active devices are the ones that read a drive column; they are all
                        # in `mod_element`, hence sorted after every passive element, so
                        # `ctx.mod_line` is the first row the adjoint has to account for
                        ele_instance = class_(**{**attr, **mode_info, 'omega': omega, 'time': t_value,
                                                 'act': param_value[..., occur_order[ele]]})
                        if not hasattr(ctx, 'mod_line'): ctx.mod_line = line_counter
                    else:
                        ele_instance = class_(**{**attr, **mode_info, 'omega': omega, 'time': t_value})

                    inward_ln, inward_rn = inward_node[ele]['ln'], inward_node[ele]['rn']
                    outward_ln, outward_rn = outward_node[ele]['ln'], outward_node[ele]['rn']
                    in_cols = inward_ln + inward_rn
                    out_cols = outward_ln + outward_rn
                    num_constraint = len(in_cols)
                    rows = range(line_counter, line_counter + num_constraint)

                    # shape: (time_pin, len(omega), len(ln) + len(rn), len(ln) + len(rn))
                    transfer_matrix = ele_instance.transfer(None)
                    if transfer_matrix.ndim == 3:
                        # a model that leaves the (broadcast) time axis out entirely
                        transfer_matrix = transfer_matrix.unsqueeze(0)

                    # set matrix A for the output node and the input node; note the difference between these two cases
                    minus_unit_index.extend(r * dim + c for r, c in zip(rows, out_cols))

                    block_index = [r * dim + c for r in rows for c in in_cols]
                    block_value = transfer_matrix.reshape(transfer_matrix.shape[0], num_freq,
                                                          num_constraint * num_constraint)

                    if block_value.shape[0] == 1:
                        time_varying = False
                    elif block_value.shape[0] == num_time:
                        # a passive model is free to broadcast its scatter matrix over the time
                        # axis instead of returning a length-1 one; detect that and store it once
                        time_varying = not torch.equal(block_value, block_value[:1].expand_as(block_value))
                        if not time_varying:
                            block_value = block_value[:1]
                    else:
                        raise RuntimeError(
                            f"Model '{class_.__name__}' returned a scatter matrix whose leading (time) axis "
                            f"has length {block_value.shape[0]}, but the simulation has {num_time} time "
                            f"point(s).  It must be either 1 (time invariant) or {num_time}.")

                    if time_varying:
                        dynamic_index.extend(block_index)
                        dynamic_value.append(block_value)
                    else:
                        static_index.extend(block_index)
                        static_value.append(block_value)

                    line_counter += num_constraint

                    break
            else:
                raise RuntimeError(f"The model {ele} is not defined in the simulator. Check if there is a typo, "
                                   "or define the model by yourself.")

        static_index = torch.tensor(unit_index + minus_unit_index + static_index, dtype=torch.long)
        static_value = torch.cat(                               # (1, len(omega), nnz_static)
            [torch.ones(1, num_freq, len(unit_index), dtype=cdtype, device=device),
             -torch.ones(1, num_freq, len(minus_unit_index), dtype=cdtype, device=device)]
            + static_value, dim=-1)
        if dynamic_index:
            dynamic_index = torch.tensor(dynamic_index, dtype=torch.long)
            dynamic_value = torch.cat(dynamic_value, dim=-1)    # (time_pin, len(omega), nnz_dynamic)
        else:
            dynamic_index, dynamic_value = None, None

        all_index = static_index if dynamic_index is None else torch.cat([static_index, dynamic_index])
        if torch.unique(all_index).numel() != all_index.numel():
            # Two entries landing on the same (row, column) would be summed by the sparse
            # assembler but overwritten by the dense one; each row belongs to exactly one
            # boundary condition or one device port, so this must never happen.
            raise RuntimeError("Internal error: the photonic system matrix was assembled with "
                               "duplicate (row, column) entries.")

        backend = _solver_backend(dim, num_time, num_freq)

        if backend == 'sparse':
            all_value = static_value.expand(num_time, num_freq, static_value.shape[-1])
            if dynamic_value is not None:
                all_value = torch.cat([all_value, dynamic_value], dim=-1)
            factor = _SparseLU(all_index, all_value, dim, num_time, num_freq, keep=need_grads)
            del all_index, all_value
            x = factor.solve(b)
        else:
            A = torch.zeros([num_time, num_freq, dim, dim], dtype=cdtype, device=device)
            flat = A.view(num_time, num_freq, dim * dim)
            flat[..., static_index] = static_value
            if dynamic_index is not None:
                flat[..., dynamic_index] = dynamic_value
            del flat, static_index, static_value, dynamic_index, dynamic_value

            if need_grads:
                # SPEC-P2.2: one LU factorisation, reused by the forward solve and by every
                # adjoint right-hand side.  An explicit matrix inverse used to be formed here, which is
                # both slower and less accurate.
                factor = _DenseLU(A)
                del A
                x = factor.solve(b)
            else:
                factor = None
                x = torch.linalg.solve(A, b.expand(num_time, num_freq, dim, 1))

        ctx.need_grads = need_grads
        if need_grads:
            ctx.factor = factor
            ctx.save_for_backward(x, omega, t_value, param_value)
            ctx.circuit_element = circuit_element
            ctx.mod_element = mod_element
            ctx.model_info = model_table
            ctx.node_has_ele = node_has_ele
            ctx.srce_node = srce_node
            ctx.node2ind = node2ind
            ctx.mode_info = mode_info
            ctx.occur_order = occur_order
            ctx.inward_node = inward_node
            ctx.outward_node = outward_node
            ctx.node_indices = detect_ind

        returned_res = x[..., outward_map(detect_ind), 0]  # shape (time_pout, len(omega), dim_pout)
        middle_res = {prob_node[i]: x[..., [inward_map(node), outward_map(node)], 0] for i, node in enumerate(prob_ind)}

        return returned_res, middle_res

    @staticmethod
    def backward(ctx, *grad_output):
        # grad_output[0] corresponds to dL/dt_value, we know it must be zero.
        # grad_output[1] corresponds to dL/dreturned_res
        # grad_output[2] corresponds to dL/dmiddle_res, we know it must be zero.
        # Because our Loss L=L(returned_res) only.
        if not ctx.need_grads:
            raise RuntimeError("Please set photonic.need_grads to True if backward is needed.")

        x, omega, t_value, param_value = ctx.saved_tensors

        grad_return = [None] * 15
        if param_value is None or not ctx.mod_element:
            return tuple(grad_return)

        cdtype, device = config['complex_dtype'], config['device']
        num_time, num_freq, dim = x.shape[0], x.shape[1], x.shape[2]

        # ------------------------------------------------------------------ adjoint state
        # The chain rule through the linear solve is
        #     dL/dtheta = -v^H A^{-1} (dA/dtheta) x ,   v = dL/dres scattered onto the detector rows.
        # Solving `A^H q = v` *once* turns every remaining per-modulator contribution into a
        # cheap contraction, so the adjoint costs a single extra right-hand side no matter how
        # many modulators the circuit has, and no inverse is ever formed (SPEC-P2.2).
        #
        # Wirtinger derivative: https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc
        # For our case z is real (act-signal), s is complex, Eq (4) on the page reduced to a*conj(b) + conj(a)*b
        # Moreover, tricky: d/dz_conj = 0.5 * d / dx, our grads is d / dx. So to use the equation (4), the 2 cancels.
        detect = outward_map(ctx.node_indices).to(torch.long)
        rhs = torch.zeros((num_time, num_freq, dim, 1), dtype=cdtype, device=device)
        rhs.index_add_(-2, detect, grad_output[0].unsqueeze(-1).to(cdtype))
        adjoint_state = ctx.factor.solve_adjoint(rhs).conj().squeeze(-1)   # (time, len(omega), 2 * num_node)
        del rhs

        # same dtype the old `einsum(...).real` produced: the real counterpart of complex_dtype
        grad_param = torch.zeros(param_value.shape, dtype=torch.empty(0, dtype=cdtype).real.dtype,
                                 device=device)

        line_counter = ctx.mod_line
        for ele, attr in sorted(ctx.mod_element.items()):
            for key, value in ctx.model_info.items():
                if ele.startswith(key):
                    class_ = _model_class(key, value)

                    inward_ln, inward_rn = ctx.inward_node[ele]['ln'], ctx.inward_node[ele]['rn']

                    num_constraint = len(inward_ln) + len(inward_rn)
                    index = torch.arange(line_counter, line_counter + num_constraint, dtype=torch.long)
                    in_cols = torch.tensor(inward_ln + inward_rn, dtype=torch.long)

                    # dS(t)/dact(t), shape (time, len(omega), num_constraint, num_constraint).
                    # The full Jacobian is block-diagonal in its two time axes, so only the
                    # diagonal is ever formed -- the old code allocated a
                    # (time, time, len(omega), 2 * num_node, 2 * num_node) work array to hold it
                    # (SPEC-P2.1).
                    jac_diag = _modulator_jacobian(class_, {**attr, **ctx.mode_info, 'omega': omega},
                                                   t_value, param_value[..., ctx.occur_order[ele]],
                                                   num_freq, num_constraint)

                    # (dA/dtheta) x restricted to the rows this modulator owns
                    dax = torch.matmul(jac_diag, x[..., in_cols, :]).squeeze(-1)  # (time, len(omega), nc)

                    contribution = -(adjoint_state[..., index] * dax).sum(dim=(-1, -2))  # (time,)
                    grad_param[:, ctx.occur_order[ele]] = contribution.real

                    line_counter += num_constraint
                    break

        # grad_output[0] shape (time_pout, len(omega), dim_pout) complex tensor
        # grad_return[1] should be (time_pin, dim_pin)
        grad_return[1] = grad_param

        return tuple(grad_return)
