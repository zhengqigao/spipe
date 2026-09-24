"""Envelope propagation: the photonic network keeps its optical memory (SPEC-P3).

Why this module exists
----------------------
SPIPE's transient solve is *quasi-static*: at every time sample it solves the frequency-domain
network in **steady state**, which silently assumes that every optical transit and round-trip time
in the circuit is negligible compared with the modulation timescale.  The paper states and defends
the *modulator* response assumption (Assumption 1) but never this *network* one.

The failure is structural rather than gradual.  With zero optical memory there is no delay at all,
so resonator ring-up, delay-set oscillation and pattern-dependent ISI are not approximated badly --
they are simply **absent**.  A 250 um waveguide at ``ng = 4`` is 3.3 ps one way and a 3x3 mesh round
trip is tens of ps: at the published 0.1 Gsps there is a factor of ~1000 of margin, but at 10 Gb/s
that is 10-50% of a bit, and a high-Q ring has a photon lifetime measured in nanoseconds.

What this module does
---------------------
The circuit is split at the modulator boundary.

* The **passive** sub-network is linear *and time invariant*.  Its transfer functions are therefore
  computed **once**, over the ``.freq`` grid, and inverse-Fourier-transformed into impulse
  responses.
* The **modulators** carry all the time variation and are evaluated per time sample exactly as they
  are today.
* The detected field is the convolution ``E(t) = integral h(tau) E_mod(t - tau) d tau``: light
  arriving at a detector at time ``t`` left the modulator at ``t - tau`` and therefore carries the
  modulator state at the **retarded** time.  That is precisely the physics the quasi-static
  formulation drops.

The assumption is thereby upgraded from *"optical memory is zero"* to *"optical memory is short
compared with how fast the modulator changes"* -- strictly weaker, and enough to cover 10-100 Gb/s
links and ring ring-up.

The algebra
-----------
The assembled system is ``A(t) x(t) = b`` with ``A(t) = A_s + A_m(t)``, where ``A_s`` is the
*static* group already isolated by :mod:`spipe.photonic.photonic` (boundary rows plus every passive
device row) and ``A_m(t)`` holds only the rows of the time-varying devices.  Writing ``y`` for the
outgoing waves those rows produce (``y = S(t) u``, ``u`` the waves entering the modulators),

    A_s x = b - R y              ==>   x = x_0 - X_R y
    u     = u_0 + G y                  (G  = -[X_R] restricted to the modulator input columns)
    z     = z_0 + D y                  (D  = -[X_R] restricted to the observed columns)

with ``x_0 = A_s^{-1} b`` and ``X_R = A_s^{-1} R``.  ``u_0``, ``G``, ``z_0`` and ``D`` are pure
passive-network transfers: one factorisation of ``A_s`` per optical frequency, ``1 + M`` right-hand
sides, and **no time axis at all**.  Solving the three relations simultaneously at each ``t``
reproduces the quasi-static answer exactly; that is the check :func:`_selftest_quasistatic` makes.

Envelope mode replaces the three *products* by *convolutions*::

    u(t) = u_0 + sum_p g[p] y(t - p dt)
    y(t) = S(t) u(t)
    z(t) = z_0 + sum_p d[p] y(t - p dt)

``u_0`` and ``z_0`` stay products because the laser is CW: they are already the exact response to a
constant source.  ``g[p]`` and ``d[p]`` are the impulse responses of ``G`` and ``D``, sampled on the
transient time grid.  Every tap set satisfies ``sum_p g[p] = G`` exactly (see
:func:`_build_filters`), so the steady state of the envelope recursion is *identical* to the
quasi-static solution -- only the way it is reached differs.

From the ``.freq`` grid to an impulse response
----------------------------------------------
``.freq`` is a uniform grid ``omega_j = omega_0 + j * dOmega`` (``_preprocess_freq`` uses
``linspace``), which is exactly what an inverse FFT needs.  With SPIPE's ``exp(+i beta l)``
propagation convention a delay ``tau`` is ``H(omega) = exp(+i omega tau)``.

A filter sampled at the transient step ``dt`` has a transfer that repeats with period ``1/dt``, so
it is fixed by exactly ``M = 1 / (df * dt)`` samples of ``H`` -- and those samples are already on
the ``.freq`` grid.  The impulse response of carrier ``omega_k`` is therefore the inverse DFT of
the ``M``-point window of ``.freq`` centred on it::

    h_k[p] = (1/M) sum_{q=-M/2}^{M/2-1} H(omega_k + q dOmega) exp(-2 pi i q p / M),
    tau_p  = p / (M df)  ~  p dt

Nothing is interpolated: every value used is a solved sample.  The ``q = 0`` term is the only one
that survives the sum over ``p``, so ``sum_p h_k[p] = H_k`` **exactly** -- every steady state the
envelope reaches is the number the quasi-static solver would have produced.  For a pure delay
``tau_0`` the window gives ``h_k[p] = exp(+i omega_k tau_0) * sinc_M(tau_0/dt - p)``: the envelope
is delayed by ``tau_0`` and picks up the carrier phase, the correct narrowband result, with
sub-sample delays represented by the fractional-delay kernel rather than rounded.

Two things follow, and both are reported rather than hidden:

* **the window has to fit inside the band.**  Carrier ``omega_k`` needs ``H`` over
  ``omega_k +/- pi/dt``; within ``M/2`` grid points of either end of ``.freq`` that range does not
  exist.  Taking the band periodically instead -- one FFT over the whole band, which splices its top
  onto its bottom -- fabricates a discontinuity and is wrong by order one, not by a little (measured
  on an unbalanced MZI: 3e-4 mid-band, 0.5 at the edge).  Those carriers keep the single DC tap,
  i.e. they are reported quasi-statically, and a warning says how many and how much wider the band
  needs to be.
* **``1/df`` has to exceed the circuit's memory.**  The response is unaliased only over
  ``|tau| < 1/(2 df)``; a resonator whose photon lifetime approaches that wraps around onto short
  delays.  ``Photonic.max_group_delay()`` bounds the delay of a single path and a measurement of
  the tap energy near the edge of the window catches the resonant case it cannot see.

Sizing the ``.freq`` grid, and reading a resonator
--------------------------------------------------
Three requirements pull against each other, and a resonator is where they collide:

* ``band >> 1/dt`` -- each carrier needs a full Nyquist window.  The carriers within ``M/2`` grid
  points of either end do not get one; they are handled by ``config['envelope_edge']`` and the
  residual error falls with the *edge fraction* ``M/F = 1/(band dt)``.
* ``1/(2 df) >> tau_max`` -- the impulse response must fit in the unaliased delay window.  For a
  cavity the relevant ``tau_max`` is the **photon lifetime**, longer than the round trip by the
  finesse, which ``Photonic.max_group_delay()`` cannot see.
* for a resonator, reading a *single* ring-up time out of the detector needs the band to stay
  inside one free spectral range, ``1/T_rt``.

The third one is a property of the measurement, not of this module, and it is easy to trip over.
``PDArray`` sums ``|p|^2`` over the whole frequency axis, so a band spanning several FSRs adds
carriers at completely different detunings: the on-resonance ones fill over the photon lifetime and
the far-detuned ones respond almost instantly.  That sum is not a single exponential, and fitting a
time constant to it does not converge as the grid is refined -- it wanders.

The photon lifetime is a property of the **complex field at one carrier** (it depends only on
``|r a|`` and ``T_rt``, so every carrier shares it).  Take it from a ``.prob`` node -- the outward
wave, index 1 -- at one carrier with a full window, not from the photocurrent.  Done that way the
band may span as many FSRs as it likes, which is what lets the first two requirements be met at
once.

Adiabatic limit
---------------
Quasi-static mode uses ``H(omega_k)`` for the whole modulation band; envelope mode uses
``H(omega_k + Omega)``.  The difference between the two over ``|Omega| <= pi/dt`` -- read straight
off the window samples, so nothing is interpolated -- is therefore exactly what envelope mode adds,
and it bounds the error of ignoring it.  If that deviation is below
``config['envelope_adiabatic_ratio']`` (default 1e-2) of the peak transfer, every filter collapses
to its exact DC value and the run is handed straight back to the quasi-static solver, so
``mode='envelope'`` reproduces ``mode='quasistatic'`` **bit for bit**, not merely to some tolerance.
For a pure delay the deviation is ``~ pi tau / dt``, i.e. the default collapses a group delay below
roughly 0.3% of a time step.

The commonest way to land there is ``M < 2``: a ``.freq`` grid whose resolution ``df`` satisfies
``1/df < 2 dt`` cannot represent any delay larger than half a time step, so there is nothing for
the envelope to add.  That is the case for essentially every netlist written for the published
0.1 Gsps flow -- a THz-wide sweep at ns time steps -- which is why ``mode='envelope'`` costs those
runs exactly nothing and changes no number.  Seeing optical memory needs a ``.freq`` grid fine
enough to resolve it (``df <= 1/(10 dt)`` is a reasonable starting point) *and* wide enough to
cover ``1/dt`` around each carrier; the warnings spell both out with the numbers filled in.

Towards a state-space formulation (deliberately out of scope here)
------------------------------------------------------------------
Everything above is a *tapped delay line*: the state is the finite history ``y(t - p dt)``,
``p = 1 .. P``, and the update is the affine map written out in :func:`_envelope_recursion`.  Folding
the photonic delay into the electronic MNA (SPEC-P3 explicitly defers this) therefore only needs the
delay line promoted to explicit unknowns: append ``P * M`` state rows ``s_p(t) = s_{p-1}(t - dt)``
with ``s_0 = y``, replace ``u = u_0 + sum_p g[p] s_p`` by a set of linear MNA rows, and the whole
electronic-photonic system becomes one DAE with no fixed-point iteration.  The tap tensors produced
here (``g``, ``d``, with their integer tap offsets) are exactly the coefficient matrices those rows
would need.
"""

import contextlib
import itertools
import warnings
from math import ceil, log2, pi
from typing import Dict, List, Optional, Tuple

import torch

from spipe import config

from .model.base import _has_fc, as_complex, as_real
from .photonic import (Simulate, _HAS_SPARSE, _entry_is_active, _model_class, _model_info,
                       _solver_backend, inward_map, outward_map)

if _has_fc:  # pragma: no cover - mirrors the import cascade in model/base.py
    try:
        from torch.func import functional_call
    except ImportError:                                        # torch < 2.0
        try:
            from torch.nn.utils.stateless import functional_call
        except ImportError:
            from torch.nn.utils._stateless import functional_call

if _HAS_SPARSE:  # pragma: no cover - mirrors the guard in photonic.py
    import numpy as _np
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

__all__ = ['simulate_envelope', 'EnvelopeModel']


#: Every filter is collapsed to its single DC tap -- making ``mode='envelope'`` identical to
#: ``mode='quasistatic'`` -- when the passive transfer moves by less than this fraction of its peak
#: magnitude across a carrier's whole modulation band ``|Omega| <= pi/dt`` (see
#: :func:`_band_deviation`).  That is what "the optical memory is negligible on this time grid"
#: means quantitatively, and the same number bounds the error the collapse can make.  For a network
#: that is a pure delay ``tau`` the deviation is ``~ pi tau / dt``, so the default collapses a group
#: delay below about 0.3% of a time step.  Override with
#: ``spipe.config['envelope_adiabatic_ratio']``.
ADIABATIC_RATIO = 1e-2

#: What to do with a carrier whose ``M``-point window runs off the end of the ``.freq`` band.
#:
#: ``'zerofill'`` (default)
#:     Build the filter from the part of the window that exists and treat the rest as zero.  The
#:     carrier's own sample always exists, so the DC gain -- and therefore the steady state -- stays
#:     exact; the filter is the true impulse response smeared by the Dirichlet kernel of the known
#:     sub-band.  The delay survives, blurred, instead of being thrown away.
#: ``'quasistatic'``
#:     Keep only the DC tap, i.e. report that carrier with zero optical memory.  This is the
#:     *dangerous* option and is no longer the default: substituting a zero-delay term biases the
#:     detector sum -- which adds the carriers in power -- back towards the quasi-static answer that
#:     envelope mode exists to escape, so the error is invisible and always in the same direction.
#: ``'drop'``
#:     Zero those carriers entirely so they contribute nothing to the detector.  The remaining sum
#:     is a correct average over fewer channels rather than a wrong average over all of them, at the
#:     cost of an obviously reduced photocurrent.
#:
#: Override with ``spipe.config['envelope_edge']``.
EDGE_POLICY = 'zerofill'

#: How much wider than the whole ``.freq`` band a carrier's Nyquist window may be before
#: ``'zerofill'`` gives up and the run collapses to quasi-static.  At the default, at least a
#: quarter of every window is real data.
EDGE_WINDOW_FACTOR = 4

#: Impulse-response taps smaller than this fraction of the largest tap are dropped, which keeps the
#: convolution length proportional to the physical memory of the circuit rather than to the
#: ``.freq`` grid.  Override with ``spipe.config['envelope_tap_tol']``.
TAP_TOL = 1e-12

# --------------------------------------------------------------------------------------- assembly

def _scatter_with_graph(instance, act: Optional[torch.Tensor]) -> torch.Tensor:
    """``instance.transfer(None)``, but with the drive re-bound so that autograd survives.

    :class:`~spipe.photonic.model.base.Device` stores every attribute through ``_param_wrap``,
    which wraps a tensor in ``torch.nn.Parameter``.  That constructor *makes a new leaf*: the
    stored ``params['act']`` has the values of ``param_value`` but none of its history, so the
    modulator scatter matrix comes out with ``requires_grad=True`` (Parameters require grad) and
    yet is **disconnected** from the drive.  Backward then runs without complaint and leaves
    ``param_value.grad`` at ``None``.

    The quasi-static path never notices, because :class:`~spipe.photonic.photonic.Simulate` is a
    custom :class:`torch.autograd.Function` whose backward differentiates the device models
    explicitly (``Device.transfer(vari_set)``) instead of relying on the graph.  Envelope mode is
    plain autograd, so the drive has to be put back.

    ``functional_call`` substitutes the live tensor for the stored Parameter for the duration of
    one forward, without mutating the module -- the same mechanism :meth:`Device._autodiff`
    already uses.
    """
    if act is None or not isinstance(act, torch.Tensor) or not act.requires_grad:
        return instance.transfer(None)
    if not _has_fc:  # pragma: no cover - only on torch too old to have functional_call
        warnings.warn(
            "mode='envelope' cannot carry gradients on this PyTorch build: torch.func."
            "functional_call is unavailable, so the drive cannot be re-bound past the "
            "nn.Parameter that Device.__init__ wraps it in. The result is correct but carries no "
            "autograd graph; use mode='quasistatic' for the adjoint.", stacklevel=3)
        return instance.transfer(None)
    out, _ = functional_call(instance, {'params.act': as_real(act).to(config['device'])}, None)
    return as_complex(out)


def _match_kv(name: str, model_table: Dict) -> Tuple[Optional[str], Optional[Dict]]:
    """``(key, entry)`` of the first model in ``model_table`` that ``name`` starts with.

    Iteration order is the same one :meth:`Simulate.forward` uses, so the two always resolve a
    netlist element to the same model class.
    """
    for key, value in model_table.items():
        if name.startswith(key):
            return key, value
    return None, None


class _StaticSystem:
    """The time-invariant part of the photonic system matrix, ``A_s``, and how to solve with it.

    ``A_s`` is exactly the *static* triplet group of :meth:`Simulate.forward` -- boundary rows plus
    every row whose scatter block does not vary with time -- so the split this class needs is the
    one the sparse assembly already makes.  It is factorised once per optical frequency and applied
    to ``1 + M`` right-hand sides; there is no time axis anywhere in here.
    """

    def __init__(self, rows: List[int], cols: List[int], values: torch.Tensor, dim: int):
        self.dim = dim
        self.num_freq = values.shape[0]
        self._rows, self._cols, self._values = rows, cols, values

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """``A_s X = rhs`` for ``rhs`` of shape ``(F, dim, k)``.

        The dense/sparse choice goes through the same :func:`_solver_backend` the transient solver
        uses, with ``num_time = 1`` -- there is no time axis here, so this system is a factor ``T``
        smaller than the one that solver sizes itself against.
        """
        cdtype, device = config['complex_dtype'], config['device']

        if _solver_backend(self.dim, 1, self.num_freq) == 'dense':
            index = torch.tensor([r * self.dim + c for r, c in zip(self._rows, self._cols)],
                                 dtype=torch.long)
            A = torch.zeros((self.num_freq, self.dim, self.dim), dtype=cdtype, device=device)
            A.view(self.num_freq, self.dim * self.dim)[:, index] = self._values
            try:
                return torch.linalg.solve(A, rhs)
            except RuntimeError as exc:
                raise RuntimeError(
                    "The passive (time-invariant) part of the photonic system matrix is singular, "
                    "so mode='envelope' cannot separate the passive network from the modulators. "
                    "This usually means a device has no path to a source or a detector once the "
                    "modulators are removed.") from exc

        rows = _np.asarray(self._rows)
        cols = _np.asarray(self._cols)
        vals = self._values.detach().cpu().numpy().astype(_np.complex128)
        rhs_np = rhs.detach().cpu().numpy().astype(_np.complex128)
        out = _np.empty_like(rhs_np)
        for f in range(self.num_freq):
            lu = _spla.splu(_sp.csc_matrix((vals[f], (rows, cols)), shape=(self.dim, self.dim)))
            out[f] = lu.solve(rhs_np[f])
        return torch.from_numpy(out).to(device=device, dtype=cdtype)


def _assemble(photonic, t_value: torch.Tensor, param_value: Optional[torch.Tensor]):
    """Split the netlist into ``A_s`` (passive, time invariant) and the time-varying scatter blocks.

    Mirrors the assembly in :meth:`Simulate.forward` row for row -- same row order, same
    ``time_varying`` test -- so the two solvers are looking at the same matrix.

    :return: ``(static_system, b, dynamic, dim)`` where ``dynamic`` is a list of
        ``{'rows', 'in_cols', 'S'}`` dicts, one per time-varying device, ``S`` of shape
        ``(T, F, nc, nc)``.
    """
    omega = photonic.omega
    num_freq, num_time = len(omega), len(t_value)
    node2ind = photonic.node2ind
    dim = 2 * len(node2ind)
    cdtype, device = config['complex_dtype'], config['device']
    model_table = _model_info()

    rows: List[int] = []
    cols: List[int] = []
    values: List[torch.Tensor] = []
    b = torch.zeros((num_freq, dim), dtype=cdtype, device=device)

    line = 0
    for node, ele in photonic.node_has_ele.items():
        if len(ele) == 1:
            rows.append(line)
            cols.append(inward_map(node2ind[node]))
            values.append(torch.ones((num_freq, 1), dtype=cdtype, device=device))
            source = photonic.srce_node.get(node)
            if source is not None:
                b[:, line] = source
            line += 1

    if bool(torch.all(b == 0)):
        from .photonic import _check_sources_connected
        _check_sources_connected(photonic.srce_node, photonic.node_has_ele)

    dynamic: List[Dict] = []
    for ele, attr in itertools.chain(sorted(photonic.circuit_element.items()),
                                     sorted(photonic.mod_element.items())):
        key, entry = _match_kv(ele, model_table)
        if entry is None:
            raise RuntimeError(f"The model {ele} is not defined in the simulator. Check if there is "
                               "a typo, or define the model by yourself.")
        class_ = _model_class(key, entry)

        kwargs = {**attr, **photonic.mode_info, 'omega': omega, 'time': t_value}
        if _entry_is_active(entry):
            kwargs['act'] = param_value[..., photonic.occur_order[ele]]
        instance = class_(**kwargs)

        in_cols = photonic.inward_node[ele]['ln'] + photonic.inward_node[ele]['rn']
        out_cols = photonic.outward_node[ele]['ln'] + photonic.outward_node[ele]['rn']
        num_constraint = len(in_cols)

        scatter = _scatter_with_graph(instance, kwargs.get('act'))
        if scatter.ndim == 3:
            scatter = scatter.unsqueeze(0)

        for i, col in enumerate(out_cols):
            rows.append(line + i)
            cols.append(col)
        values.append(-torch.ones((num_freq, num_constraint), dtype=cdtype, device=device))

        block = scatter.reshape(scatter.shape[0], num_freq, num_constraint * num_constraint)
        if block.shape[0] == 1:
            time_varying = False
        elif block.shape[0] == num_time:
            time_varying = not torch.equal(block, block[:1].expand_as(block))
        else:
            raise RuntimeError(
                f"Model '{class_.__name__}' returned a scatter matrix whose leading (time) axis has "
                f"length {block.shape[0]}, but the simulation has {num_time} time point(s).")

        if time_varying:
            dynamic.append({'rows': list(range(line, line + num_constraint)),
                            'in_cols': list(in_cols),
                            'S': scatter})
        else:
            for i in range(num_constraint):
                for col in in_cols:
                    rows.append(line + i)
                    cols.append(col)
            values.append(block[0])

        line += num_constraint

    if line != dim:
        raise RuntimeError(f"Internal error: assembled {line} rows for a system of size {dim}.")

    return _StaticSystem(rows, cols, torch.cat(values, dim=-1), dim), b, dynamic, dim


# ------------------------------------------------------------------------------ passive transfers

class _PassiveTransfer:
    """``u_0``, ``G``, ``z_0`` and ``D`` -- the whole time-invariant description of the circuit."""

    __slots__ = ('u0', 'G', 'z0', 'D', 'num_port', 'num_obs')

    def __init__(self, u0, G, z0, D):
        self.u0, self.G, self.z0, self.D = u0, G, z0, D
        self.num_port = u0.shape[-1]
        self.num_obs = z0.shape[-1]


def _passive_transfer(static: _StaticSystem, b: torch.Tensor, dynamic: List[Dict],
                      obs_cols: List[int]) -> _PassiveTransfer:
    num_freq, dim = b.shape
    cdtype, device = config['complex_dtype'], config['device']

    mod_rows = [r for block in dynamic for r in block['rows']]
    mod_cols = [c for block in dynamic for c in block['in_cols']]
    num_port = len(mod_rows)

    rhs = torch.zeros((num_freq, dim, 1 + num_port), dtype=cdtype, device=device)
    rhs[:, :, 0] = b
    for j, row in enumerate(mod_rows):
        rhs[:, row, 1 + j] = 1.0

    solution = static.solve(rhs)                       # (F, dim, 1 + M)
    x0, xr = solution[..., 0], solution[..., 1:]

    mod_index = torch.tensor(mod_cols, dtype=torch.long, device=device)
    obs_index = torch.tensor(obs_cols, dtype=torch.long, device=device)

    return _PassiveTransfer(u0=x0[:, mod_index],            # (F, M)
                            G=-xr[:, mod_index, :],         # (F, M, M)
                            z0=x0[:, obs_index],            # (F, O)
                            D=-xr[:, obs_index, :])         # (F, O, M)


# ------------------------------------------------------------------------------- impulse response

class _BandInterpolant:
    """A passive transfer sampled on the uniform ``.freq`` grid, plus its whole-band impulse response.

    ``taps[n] = c[n]`` is the inverse Fourier transform of the transfer over the *entire* band: it
    resolves delays to ``1/(F df)`` and is unaliased over ``|tau| < 1/(2 df)``.  The filters
    themselves are built per carrier from a narrower window (see :func:`_build_filters`); ``c[n]`` is
    kept because it is the right object for the wrap-around diagnostic, which asks whether the
    circuit's memory fits in that ``1/(2 df)`` window at all.
    """

    def __init__(self, transfer: torch.Tensor, omega: torch.Tensor):
        # transfer: (F, E) on the .freq grid
        num_freq = transfer.shape[0]
        self.num_freq = num_freq
        self.omega = omega
        self.d_omega = float(omega[1] - omega[0]) if num_freq > 1 else 0.0
        self.df = self.d_omega / (2 * pi)
        self.transfer = transfer

        self.taps = torch.fft.fft(transfer, dim=0) / num_freq            # (F, E)
        if self.df > 0.0:
            # tau_n = n / (F df), n wrapped into [-F/2, F/2): exactly torch.fft.fftfreq(F, d=df)
            self.tau = torch.fft.fftfreq(num_freq, d=self.df).to(
                dtype=config['real_dtype'], device=transfer.device)
        else:
            # a degenerate ``.freq`` line (one point, or start == stop) carries no band at all, so
            # there is no delay information to extract; every tap sits at tau = 0
            self.tau = torch.zeros(num_freq, dtype=config['real_dtype'], device=transfer.device)
        self.tau_window = float(self.tau.abs().max())     # +/- 1/(2 df), the unaliased delay range
        self.degenerate = not (self.df > 0.0)


def _next_pow2(value: int) -> int:
    return 1 if value <= 1 else 2 ** int(ceil(log2(value)))


#: Ceiling on the working set of the filter construction, ``F * L * E`` complex numbers.  The
#: envelope filters are inherently ``(carriers) x (time taps) x (transfer entries)``; a resonator
#: forces a fine ``.freq`` grid and a mesh many modulator ports, and the product can get out of hand
#: quietly.  Override with ``spipe.config['envelope_max_bytes']``.
MAX_FILTER_BYTES = 2 * 1024 ** 3


def _check_filter_budget(num_freq: int, num_taps: int, num_entry: int) -> None:
    itemsize = torch.empty(0, dtype=config['complex_dtype']).element_size()
    need = num_freq * num_taps * num_entry * itemsize
    budget = int(config.get('envelope_max_bytes', MAX_FILTER_BYTES))
    if need > budget:
        raise RuntimeError(
            f"mode='envelope' would need {need / 1024 ** 3:.2f} GB for the impulse-response "
            f"filters: {num_freq} optical frequencies x {num_taps} time taps x {num_entry} transfer "
            f"entries ({int(num_entry ** 0.5)}-ish modulator ports). The cost is "
            f"O(F * L * (M^2 + O*M)) with L ~ 1/(df * dt), so it grows with a finer .freq grid and "
            f"with the number of modulator ports. Coarsen .freq, simulate fewer modulators per run, "
            f"raise spipe.config['envelope_max_bytes'] (currently "
            f"{budget / 1024 ** 3:.2f} GB), or use mode='quasistatic'.")


def _dirichlet(x: torch.Tensor, length: int) -> torch.Tensor:
    """Length-``L`` Dirichlet (periodic sinc) kernel ``W(x)``.

        W(x) = (1/L) sum_{q=-L/2}^{L/2-1} exp(2 pi i q x / L)
             = sin(pi x) exp(-i pi x / L) / (L sin(pi x / L))

    ``W(0) = 1``, ``W`` vanishes at every other integer, and ``sum_p W(x - p) = 1`` identically over
    a full period -- which is what keeps the DC gain of a resampled filter exact.
    """
    x = x - length * torch.round(x / length)                       # W is L-periodic; wrap for range
    numer = torch.sin(pi * x)
    denom = length * torch.sin(pi * x / length)
    ratio = torch.where(x == 0, torch.ones_like(numer),
                        numer / torch.where(denom == 0, torch.ones_like(denom), denom))
    return ratio.to(config['complex_dtype']) * torch.exp(-1.j * pi * x / length).to(
        config['complex_dtype'])


def _build_filters(interp: '_BandInterpolant',
                   dt: float) -> Tuple[torch.Tensor, torch.Tensor, int, int, float]:
    """The impulse response of every carrier, on the transient time grid.

    A filter sampled at ``dt`` has a transfer that repeats with period ``1/dt``, so it is determined
    by exactly ``M = 1 / (df * dt)`` samples of ``H`` around its carrier -- no more and no fewer.
    Those samples are already on the ``.freq`` grid, so the construction is an inverse DFT of the
    ``M``-point window centred on ``omega_k``::

        h_k[p] = (1/M) sum_{q=-M/2}^{M/2-1} H(omega_k + q dOmega) exp(-2 pi i q p / M)

    with the taps landing at ``tau_p = p / (M df)``.  Nothing is interpolated: every value used is a
    solved sample.  ``q = 0`` alone survives the sum over ``p``, so ``sum_p h_k[p] = H_k`` exactly
    and the steady state of the envelope recursion is the quasi-static answer to the last bit.

    ``M`` has to be an integer while ``1/(df dt)`` need not be, so the taps come out on a grid of
    ``dt_eff = 1/(M df)`` and are stretched onto ``dt`` with the Dirichlet kernel.  That step is
    exact -- the ``M``-point sequence *is* band-limited by construction -- and it preserves the tap
    sum, so it costs nothing in either accuracy or steady state.

    **The window has to fit inside the sampled band.**  Carrier ``omega_k`` needs ``H`` over
    ``omega_k +/- pi/dt``; within ``M/2`` grid points of either end of ``.freq`` that range is simply
    not there.  Using the band periodically -- splicing its top onto its bottom, which is what a
    single whole-band FFT does -- fabricates a transfer with a discontinuity in it and produces an
    answer that is wrong by order one, not by a little.  Those carriers therefore keep the single DC
    tap, i.e. they are reported quasi-statically, and the caller is told how many and why.

    :return: ``(taps, p_grid, num_clipped, M, deviation)``; ``taps`` is ``(F, M, E)``, ``p_grid``
        the ascending integer tap offsets, ``num_clipped`` the number of carriers left
        quasi-static, and ``deviation`` the band deviation of :func:`_band_deviation`.
    """
    num_freq, num_entry = interp.transfer.shape
    df = interp.df
    device = interp.transfer.device
    window = int(round(1.0 / (df * dt))) if df > 0 else 0

    policy = str(config.get('envelope_edge', EDGE_POLICY))
    if policy not in ('zerofill', 'drop', 'quasistatic'):
        raise ValueError(f"config['envelope_edge'] must be 'zerofill', 'drop' or 'quasistatic', "
                         f"got {policy!r}.")
    # 'zerofill' works from a partial window, so it can still answer when the band is narrower than
    # one Nyquist period -- up to the point where so little of each window is known that the filter
    # is mostly an artefact of the fill.
    widest = num_freq * EDGE_WINDOW_FACTOR if policy == 'zerofill' else num_freq

    if window < 2 or window > widest:
        # Either the .freq resolution cannot represent any delay the time grid could show
        # (``window < 2`` means 1/df < 2 dt, i.e. the entire unaliased delay range +/- 1/(2 df) lies
        # inside half a time step), or the band is narrower than one Nyquist period so that no
        # carrier has a full window at all.  Both collapse to the single DC tap.
        p_grid = torch.zeros(1, dtype=torch.long, device=device)
        return interp.transfer.unsqueeze(1).clone(), p_grid, num_freq, window, 0.0

    _check_filter_budget(num_freq, window, num_entry)

    offset = window // 2                                   # position of the carrier in its window
    index = torch.arange(num_freq, device=device)
    columns = index.reshape(-1, 1) - offset + torch.arange(window, device=device).reshape(1, -1)
    in_band = (columns >= 0) & (columns < num_freq)        # which window samples actually exist
    inside = in_band.all(dim=1)                            # carriers with a complete window

    block = interp.transfer[columns.clamp(0, num_freq - 1)]                   # (F, M, E)
    deviation = _band_deviation(block, offset, in_band, float(interp.transfer.abs().max()))

    if policy == 'zerofill':
        # Missing window samples are set to zero rather than borrowed from elsewhere in the band.
        # The carrier's own sample (q = 0) always exists, and it is the only term that survives the
        # sum over p, so sum_p h_k[p] = H_k stays exact; the filter is then the true impulse
        # response smeared by the Dirichlet kernel of the sub-band that *is* known, which keeps the
        # delay instead of discarding it.
        block = block * in_band.unsqueeze(-1).to(block.dtype)

    block = torch.roll(block, -offset, dims=1)                                # carrier first
    taps = torch.fft.fftshift(torch.fft.fft(block, dim=1) / window, dim=1)    # (F, M, E), p ascending

    p_grid = torch.arange(-(window // 2), window - window // 2,
                          dtype=config['real_dtype'], device=device)

    stretch = 1.0 / (window * df * dt)                     # dt_eff / dt, always close to 1
    if abs(stretch - 1.0) * window > 1e-9:
        taps = torch.einsum('kqe,ql->kle', taps,
                            _dirichlet(p_grid.reshape(-1, 1) * stretch - p_grid.reshape(1, -1),
                                       window))

    num_clipped = int((~inside).sum())
    if num_clipped and policy != 'zerofill':
        taps[~inside] = 0.0
        if policy == 'quasistatic':
            taps[~inside, int((p_grid == 0).nonzero()[0])] = interp.transfer[~inside]

    return taps, p_grid.to(torch.long), num_clipped, window, deviation


def _trim(taps: torch.Tensor, p_grid: torch.Tensor, tol: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop leading/trailing taps below ``tol`` times the largest one."""
    magnitude = taps.abs().amax(dim=(0, 2))                                       # (L,)
    peak = float(magnitude.max()) if magnitude.numel() else 0.0
    if peak == 0.0:
        zero = torch.zeros(1, dtype=torch.long, device=p_grid.device)
        return taps[:, :1] * 0.0, zero
    keep = (magnitude > tol * peak).nonzero().flatten()
    lo, hi = int(keep[0]), int(keep[-1]) + 1
    return taps[:, lo:hi], p_grid[lo:hi]


# ------------------------------------------------------------------------------------ diagnostics

def _band_deviation(block: torch.Tensor, offset: int, in_band: torch.Tensor, peak: float) -> float:
    """How far the passive transfer moves across one carrier's modulation band.

        max over carriers, window samples and transfer entries of
            | H(omega_k + Omega) - H(omega_k) |  /  max |H|,     |Omega| <= pi/dt

    This is exactly what envelope mode adds on top of the quasi-static answer, which uses
    ``H(omega_k)`` for the whole band: it is both the right quantity to threshold and a bound on
    the error of collapsing every filter to its single DC tap.  For a network that is a pure delay
    ``tau`` it evaluates to ``~ pi tau / dt``, so the threshold reads directly as a fraction of a
    time step.

    It is measured on the ``M``-point window itself -- solved samples, nothing interpolated -- and
    only over window positions that actually exist, so a carrier near the edge of the band still
    contributes whatever part of its band is known rather than being ignored.  Measuring it on the
    *taps* instead (``sum_{p != 0} |h[p]|``) counts the fractional-delay kernel's sidelobes at full
    weight even though they cancel, and over-states the memory of a short circuit by more than an
    order of magnitude.
    """
    if peak == 0.0:
        return 0.0
    spread = (block - block[:, offset:offset + 1]).abs()
    return float(torch.where(in_band, spread.amax(dim=-1),
                             torch.zeros_like(spread[..., 0])).max()) / peak


def _bandwidth_report(interp: _BandInterpolant, dt: float, tau_estimate: float,
                      num_time: int, window_points: int, num_clipped: int) -> None:
    """``warnings.warn`` with concrete numbers when the ``.freq`` grid cannot carry the answer.

    Three independent ways the grid can be too small, each of which is invisible in the result:
    too little **bandwidth** (the ``M``-point window around a carrier runs off the end of the band,
    so that carrier has no impulse response), too little **resolution** (the response is longer than
    the unaliased delay window ``1/df`` and wraps), and too short a **record** (the convolution is
    truncated).  All three are reported with the band, the resolution and the estimated ``tau_max``
    spelled out, rather than being silently absorbed.
    """
    band = float(interp.omega[-1] - interp.omega[0]) / (2 * pi)
    df = interp.df
    window = interp.tau_window
    record = num_time * dt

    if window_points < 2:
        if tau_estimate > 0.01 * dt:
            warnings.warn(
                f"mode='envelope': the .freq grid is too coarse to resolve any optical memory on "
                f"this time grid. df = {df:.6g} Hz gives an unaliased delay range of only "
                f"+/- 1/(2 df) = {window:.6g} s, which is inside half a time step "
                f"(dt/2 = {dt / 2:.6g} s), so every delay the grid can represent rounds to zero and "
                f"the run is identical to mode='quasistatic'. The estimated maximum group delay is "
                f"tau_max = {tau_estimate:.6g} s = {tau_estimate / dt:.4g} dt; to see it, refine "
                f".freq to df <= {1.0 / (10.0 * dt):.6g} Hz "
                f"({int(ceil(band * 10.0 * dt)) + 1} points over the present {band:.6g} Hz band).",
                stacklevel=3)
        return

    if num_clipped:
        needed = window_points * df
        policy = str(config.get('envelope_edge', EDGE_POLICY))
        fraction = num_clipped / interp.num_freq
        remedy = {
            'zerofill':
                "Those carriers keep whatever part of their window exists and the rest is taken as "
                "zero: their DC gain -- hence their steady state -- is still exact, but their "
                "transient is the true impulse response smeared by the resolution of the sub-band "
                "that is known, so fast structure in them is blurred (measured on a 3 mm delay "
                "line, the detected 50% lag stays within 0.2% of ng L / c even at a 24% edge "
                "fraction).",
            'quasistatic':
                "config['envelope_edge'] = 'quasistatic': those carriers are given a single DC tap, "
                "i.e. ZERO optical memory. Because the detector adds carriers in power, that biases "
                "every reported delay LOW, by roughly the edge fraction -- here about "
                f"{100.0 * fraction:.3g}% -- and the result still looks entirely plausible. On a "
                "3 mm delay line the detected lag came out at 0.850 / 0.936 / 0.970 / 0.985 of the "
                "true delay at edge fractions of 24% / 12% / 6% / 3%. Use the default 'zerofill', "
                "or 'drop' to leave those carriers out of the sum altogether.",
            'drop':
                "config['envelope_edge'] = 'drop': those carriers are zeroed, so the photocurrent "
                f"is summed over {interp.num_freq - num_clipped} of {interp.num_freq} channels and "
                "is correspondingly low; the delay of what remains is right.",
        }[policy]
        everything = num_clipped >= interp.num_freq
        warnings.warn(
            f"mode='envelope': insufficient .freq bandwidth. A filter on a dt = {dt:.6g} s grid has "
            f"a transfer that repeats every 1/dt = {1.0 / dt:.6g} Hz, so each carrier needs the "
            f"passive transfer over the {needed:.6g} Hz around it -- "
            f"{window_points} points of the .freq grid (df = {df:.6g} Hz). The band only spans "
            f"{band:.6g} Hz over {interp.num_freq} points, so {num_clipped} of {interp.num_freq} "
            f"carriers ({100.0 * fraction:.3g}%) do not have a full window. " + remedy
            + (f" This is EVERY carrier: the band is narrower than one Nyquist period "
               f"({band:.6g} Hz < 1/dt = {1.0 / dt:.6g} Hz), so no carrier is fully resolved."
               if everything else "")
            + f" Widen the .freq band to at least {band + needed:.6g} Hz (keeping df), or increase "
              f"dt; the error falls with the edge fraction.", stacklevel=3)

    if df > 0 and window < tau_estimate:
        warnings.warn(
            f"mode='envelope': insufficient .freq resolution. df = {df:.6g} Hz "
            f"({interp.num_freq} points over {band:.6g} Hz) gives an unaliased delay window of "
            f"+/- 1/(2 df) = {window:.6g} s, but the estimated maximum optical group delay of the "
            f"network is tau_max = {tau_estimate:.6g} s (Photonic.max_group_delay(), a lower "
            f"bound: it is a longest-simple-path estimate and does not see the photon lifetime of "
            f"a resonator, which is longer than its round trip by the finesse). Any impulse "
            f"response longer than the window wraps around and reappears at short delays. Use at "
            f"least {int(ceil(band * 2.0 * tau_estimate)) + 1} .freq points (df <= "
            f"{1.0 / (2.0 * tau_estimate):.6g} Hz) for this circuit.", stacklevel=3)

    if df > 0 and window > 0 and record > 0 and window > record:
        warnings.warn(
            f"mode='envelope': the unaliased delay window of the .freq grid "
            f"(+/- {window:.6g} s) is longer than the whole transient record "
            f"({num_time} points x {dt:.6g} s = {record:.6g} s). The convolution is truncated to "
            f"the record, so any circuit memory beyond {record:.6g} s is not represented. "
            f"Simulate more time points if the circuit really is that slow.", stacklevel=3)


def _wraparound_report(interp: _BandInterpolant) -> None:
    """Measured (rather than estimated) check that the impulse response fits in the tau window.

    A response longer than ``1/df`` folds back onto short delays, which looks like an ordinary
    answer.  If an appreciable part of the tap energy sits within the outer quarter of the window --
    where a truncated exponential tail would land -- say so.

    Every carrier shares this test: the per-carrier response differs from ``c[n]`` only by a unit
    phase ramp, so ``|c_k[n]| = |c[n]|`` and the energy distribution over ``tau`` is the same one.
    """
    if interp.num_freq < 8:
        return
    energy = (interp.taps.abs() ** 2).sum(dim=-1)                                 # (N,)
    total = float(energy.sum())
    if total <= 0.0:
        return
    edge = interp.tau.abs() > 0.75 * interp.tau_window
    fraction = float(energy[edge].sum()) / total
    if fraction > 1e-2:
        warnings.warn(
            f"mode='envelope': {100.0 * fraction:.3g}% of the passive impulse-response energy lies "
            f"in the outer quarter of the unaliased delay window (|tau| > "
            f"{0.75 * interp.tau_window:.6g} s of +/- {interp.tau_window:.6g} s). The response is "
            f"probably longer than 1/df = {1.0 / interp.df:.6g} s and is wrapping around onto short "
            f"delays. Increase the number of .freq points (a high-Q resonator needs df well below "
            f"1/(2 tau_photon)).", stacklevel=3)


# ------------------------------------------------------------------------------------- recursion

def _block_scatter(dynamic: List[Dict], index: int, num_freq: int, num_port: int) -> torch.Tensor:
    """The block-diagonal modulator scatter matrix ``S(t_index)``, shape ``(F, M, M)``.

    Built one time point at a time: the dense ``(T, F, M, M)`` stack would be quadratic in the
    number of modulator ports, which a mesh circuit cannot afford.
    """
    out = torch.zeros((num_freq, num_port, num_port),
                      dtype=config['complex_dtype'], device=config['device'])
    offset = 0
    for block in dynamic:
        size = len(block['rows'])
        scatter = block['S']
        out[:, offset:offset + size, offset:offset + size] = \
            scatter[index if scatter.shape[0] > 1 else 0]
        offset += size
    return out


def _envelope_recursion(transfer: _PassiveTransfer, dynamic: List[Dict],
                        g_taps: torch.Tensor, g_offsets: torch.Tensor,
                        num_time: int) -> torch.Tensor:
    """March ``y(t) = S(t) u(t)``, ``u(t) = u_0 + sum_p g[p] y(t - p dt)`` forward in time.

    Taps at ``p <= 0`` are lumped into the instantaneous term.  A causal network has no ``p < 0``
    response; the small negative-``p`` weight the band-limited resampling produces is the tail of
    the fractional-delay kernel, and lumping it keeps ``sum_p g[p] = G`` exact, hence keeps the
    steady state exact.

    The history is pre-filled with the steady state of the first drive sample, i.e. the laser has
    been on forever and the circuit starts settled.  Consequently a *constant* drive reproduces the
    quasi-static answer at every sample, and every transient in the result is caused by the drive.

    :return: ``y`` of shape ``(T, F, M)``.
    """
    cdtype, device = config['complex_dtype'], config['device']
    num_freq, num_port = transfer.u0.shape

    causal = g_offsets > 0
    g_now = g_taps[:, ~causal].sum(dim=1)                                          # (F, M, M)
    g_hist = g_taps[:, causal]                                                     # (F, P, M, M)
    delays = g_offsets[causal]                                                     # (P,)
    depth = int(delays.max()) if delays.numel() else 0

    identity = torch.eye(num_port, dtype=cdtype, device=device).expand(num_freq, -1, -1)

    # steady state of the first sample -- the exact quasi-static solution there, because
    # sum_p g[p] == G to round-off
    scatter0 = _block_scatter(dynamic, 0, num_freq, num_port)
    u_ss = torch.linalg.solve(identity - torch.matmul(transfer.G, scatter0),
                              transfer.u0.unsqueeze(-1))
    y_ss = torch.matmul(scatter0, u_ss)                                            # (F, M, 1)

    history = y_ss.squeeze(-1).unsqueeze(0).expand(depth, num_freq, num_port).clone() \
        if depth else torch.zeros((0, num_freq, num_port), dtype=cdtype, device=device)
    y = torch.empty((num_time, num_freq, num_port), dtype=cdtype, device=device)
    store = torch.cat([history, y]) if depth else y
    base = depth

    for i in range(num_time):
        scatter = _block_scatter(dynamic, i, num_freq, num_port)
        rhs = transfer.u0.unsqueeze(-1)
        if depth:
            past = store[base + i - delays]                                        # (P, F, M)
            rhs = rhs + torch.einsum('fpab,pfb->fa', g_hist, past).unsqueeze(-1)
        u = torch.linalg.solve(identity - torch.matmul(g_now, scatter), rhs)
        store[base + i] = torch.matmul(scatter, u).squeeze(-1)

    return store[base:]


def _observe(transfer: _PassiveTransfer, y: torch.Tensor, d_taps: torch.Tensor,
             d_offsets: torch.Tensor) -> torch.Tensor:
    """``z(t) = z_0 + sum_p d[p] y(t - p dt)`` by FFT convolution.  Shape ``(T, F, O)``."""
    num_time = y.shape[0]
    p_min, p_max = int(d_offsets[0]), int(d_offsets[-1])
    pad_left, pad_right = max(p_max, 0), max(-p_min, 0)

    pieces = []
    if pad_left:
        pieces.append(y[:1].expand(pad_left, -1, -1))     # settled before the record started
    pieces.append(y)
    if pad_right:
        pieces.append(y[-1:].expand(pad_right, -1, -1))   # hold the last sample past the end
    padded = torch.cat(pieces) if len(pieces) > 1 else y

    length = padded.shape[0]
    num_tap = d_taps.shape[1]
    size = _next_pow2(length + num_tap)

    taps = d_taps.permute(1, 0, 2, 3)                                             # (P, F, O, M)
    spec = torch.fft.fft(taps, n=size, dim=0)                                     # (n, F, O, M)
    signal = torch.fft.fft(padded, n=size, dim=0)                                 # (n, F, M)
    conv = torch.fft.ifft(torch.einsum('nfoa,nfa->nfo', spec, signal), dim=0)

    start = pad_left - p_min
    return transfer.z0.unsqueeze(0) + conv[start:start + num_time]


# ------------------------------------------------------------------------------------ entry point

class EnvelopeModel:
    """Everything an envelope run needs that does not change from time step to time step.

    The passive transfers (``u_0``, ``G``, ``z_0``, ``D``) and the impulse responses built from
    them depend only on the circuit, the ``.freq`` grid and ``dt``.  The modulator scatter matrices
    do depend on the drive, and are held here too because they come out of the same assembly pass;
    that is why :func:`simulate_envelope` only reuses a cached model for an identical drive.
    """

    def __init__(self, photonic, t_value: torch.Tensor, param_value: Optional[torch.Tensor],
                 dt: float):
        self.dt = dt
        self.deviation, self.adiabatic = 0.0, True
        self.window, self.num_clipped = 0, 0
        static, b, self.dynamic, _ = _assemble(photonic, t_value, param_value)
        if not self.dynamic:
            # no device varies in time (a constant drive counts): the circuit is linear and time
            # invariant, so its steady state *is* the answer at every sample
            return

        self.obs_cols, self.detect_slice, self.prob_slices = _observation_columns(photonic)
        self.transfer = _passive_transfer(static, b, self.dynamic, self.obs_cols)

        num_freq = len(photonic.omega)
        num_port, num_obs = self.transfer.num_port, self.transfer.num_obs

        # G and D share one interpolant so the diagnostics see the whole passive network at once
        flat = torch.cat([self.transfer.G.reshape(num_freq, -1),
                          self.transfer.D.reshape(num_freq, -1)], dim=-1)
        self.interp = _BandInterpolant(flat, photonic.omega)

        if self.interp.degenerate:
            warnings.warn(
                f"mode='envelope': the .freq grid carries no bandwidth ({num_freq} point(s) "
                f"spanning {float(photonic.omega[-1] - photonic.omega[0]) / (2 * pi):.6g} Hz), so no "
                f"impulse response can be formed -- H(omega) is known at a single optical frequency "
                f"and an inverse FFT of one sample is a delta at tau = 0. The run falls back to "
                f"mode='quasistatic'. Give '.freq <start> <stop> <points>' a real band (stop > "
                f"start, at least 2 points, df <= 1/(2 tau_max) with tau_max = "
                f"{photonic.max_group_delay():.6g} s) to model the optical memory.", stacklevel=4)
            self.deviation, self.adiabatic = 0.0, True
            return

        taps, p_grid, self.num_clipped, self.window, self.deviation = \
            _build_filters(self.interp, dt)
        self.adiabatic = self.deviation < float(config.get('envelope_adiabatic_ratio',
                                                           ADIABATIC_RATIO))
        if self.adiabatic:
            if self.window < 2 or self.num_clipped >= num_freq:
                # Collapsed because the .freq grid cannot carry an impulse response at all, not
                # because the circuit is fast.  Both routes return the quasi-static numbers, which
                # look perfectly ordinary, so they have to be announced: `window < 2` is a grid too
                # coarse to resolve a delay, `num_clipped == F` a band narrower than one Nyquist
                # period, so that not one carrier has a full window.
                _bandwidth_report(self.interp, dt, photonic.max_group_delay(), len(t_value),
                                  self.window, self.num_clipped)
                _wraparound_report(self.interp)
            return

        _bandwidth_report(self.interp, dt, photonic.max_group_delay(), len(t_value),
                          self.window, self.num_clipped)
        _wraparound_report(self.interp)

        tol = float(config.get('envelope_tap_tol', TAP_TOL))
        taps, p_grid = _trim(taps, p_grid, tol)

        split = num_port * num_port
        self.g_taps = taps[..., :split].reshape(num_freq, -1, num_port, num_port)
        self.d_taps = taps[..., split:].reshape(num_freq, -1, num_obs, num_port)
        self.offsets = p_grid


def _observation_columns(photonic) -> Tuple[List[int], slice, List[Tuple[str, int]]]:
    """Columns of ``x`` the caller wants back: detector outputs, then each probe's (in, out) pair."""
    try:
        detect = [outward_map(photonic.node2ind[node]) for node in photonic.dout_node]
        probe = [(node, photonic.node2ind[node]) for node in photonic.middle_node]
    except KeyError:
        raise KeyError("At least one node required by photo detector and .prob syntax is not in the "
                       "circuit.")
    cols = list(detect)
    slices = []
    for node, index in probe:
        slices.append((node, len(cols)))
        cols.extend([inward_map(index), outward_map(index)])
    return cols, slice(0, len(detect)), slices


def simulate_envelope(photonic, t_value: Optional[torch.Tensor],
                      param_value: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """``Photonic.simulate(..., mode='envelope')``.

    Same arguments and the same two return values as the quasi-static
    :class:`~spipe.photonic.photonic.Simulate` path: ``res`` of shape ``(T, len(omega), n_pd)`` and
    the ``.prob`` dictionary.  Falls back to that path verbatim whenever the circuit has no optical
    memory the transient grid could represent, so the two modes then agree bit for bit.
    """
    # `need_grads` is forced on whenever the drive carries a graph: mode='envelope' is
    # differentiable through plain autograd, and it would be a trap for the fall-back branches --
    # which hand the run to the quasi-static adjoint -- to be the only ones that demand the flag.
    grad_wanted = photonic.need_grads or (param_value is not None and param_value.requires_grad)
    delegate = lambda: Simulate.apply(
        t_value, param_value, photonic.omega, photonic.node_has_ele, photonic.srce_node,
        photonic.node2ind, photonic.circuit_element, photonic.mod_element, photonic.mode_info,
        photonic.occur_order, (photonic.dout_node, photonic.middle_node), photonic.inward_node,
        photonic.outward_node, grad_wanted)

    if t_value is None or param_value is None or len(t_value) < 2:
        # nothing varies in time: the steady-state solve *is* the envelope answer
        return delegate()

    step = t_value[1:] - t_value[:-1]
    dt = float(step.mean())
    # The tolerance has to follow the dtype: differencing a grid that runs to 1e-8 s in float32
    # leaves ~1e-5 of relative jitter on a 1 ps step, which is round-off, not a non-uniform grid.
    jitter = 1e4 * torch.finfo(step.dtype).eps if step.is_floating_point() else 1e-9
    if dt <= 0 or float((step - dt).abs().max()) > jitter * dt:
        raise RuntimeError(
            "mode='envelope' needs a uniform transient time grid: the passive network's impulse "
            f"response is expressed on it. Got a grid whose step ranges from "
            f"{float(step.min()):.6g} s to {float(step.max()):.6g} s (mean {dt:.6g} s), which "
            f"varies by more than {jitter:.3g} of the step.")

    # The model holds the modulator scatter matrices as well as the passive transfers, so it is only
    # reusable for the *same* drive.  The drive is compared by value, not by identity: a caller that
    # rebuilds `param_value` every iteration (the electronic/photonic fixed point does) would
    # otherwise hand back a tensor at a recycled address and silently get the previous answer.
    #
    # The cache is bypassed entirely when the drive carries an autograd graph.  Two tensors that
    # are equal *by value* can sit on completely different graphs, so handing back a cached model
    # would attach the backward pass to the previous iteration's drive and silently leave this
    # one's `.grad` at None -- exactly the failure the cache is meant to avoid on the forward side.
    differentiable = bool(param_value.requires_grad)
    cache = getattr(photonic, '_envelope_cache', None)
    if (not differentiable and cache is not None and cache[0] == dt
            and cache[1].shape == t_value.shape and torch.equal(cache[1], t_value)
            and cache[2].shape == param_value.shape and torch.equal(cache[2], param_value)):
        model = cache[3]
    else:
        model = EnvelopeModel(photonic, t_value, param_value, dt)
        if not differentiable:
            photonic._envelope_cache = (dt, t_value.detach().clone(),
                                        param_value.detach().clone(), model)

    if model.adiabatic:
        # Nothing varies in time, or every impulse response has collapsed to a single tap holding
        # its exact DC value -- which is the quasi-static system.  Hand the run over rather than
        # rebuild it, so that the two modes agree to the last bit rather than to a tolerance.
        return delegate()

    # Record a graph only when a gradient was asked for. The device attributes are
    # nn.Parameters, which require grad, so without this every envelope result carried a graph
    # nobody wanted -- and `.numpy()` on it raised.
    with (contextlib.nullcontext() if differentiable else torch.no_grad()):
        y = _envelope_recursion(model.transfer, model.dynamic, model.g_taps, model.offsets,
                                len(t_value))
        z = _observe(model.transfer, y, model.d_taps, model.offsets)               # (T, F, O)

    res = z[..., model.detect_slice]
    middle = {node: z[..., start:start + 2] for node, start in model.prob_slices}
    _check_envelope_physics(photonic, t_value, param_value, res, model.offsets)
    return res, middle


def _check_envelope_physics(photonic, t_value, param_value, res, offsets) -> None:
    """Two checks on the envelope result, each against something that must be true.

    The recursion represents a delay shorter than the time step with a fractional-delay kernel
    whose taps oscillate. Inside a feedback loop -- a modulator in a ring -- that kernel can act
    as gain at some carriers and the recursion goes unstable: a ring with a 1.68 ps round trip
    reported 464x the input power at 1 ps steps, and at 2 ps steps carriers the user never looked
    at reached 4e15x, all with no error.

    Note what is NOT a valid test: "detected power <= launched power at every instant". A
    resonator stores energy and releases it when the input changes, so it can briefly emit more
    than it receives (tb11's ring does, 4x, correctly); and a sharp drive edge represented in a
    finite band overshoots (Gibbs), a bounded, known limitation of the method.

    1. **Runaway** -- an instability grows without bound; stored-energy release and Gibbs
       overshoot die away. So at the final sample the detected power must not exceed ten times
       both the launched power and the circuit's own steady-state output. Beyond that the
       numbers are meaningless, and it is an error.
    2. **Steady state** -- once the drive has been constant for longer than the network's memory,
       the envelope result must equal the quasi-static one (the module is built so that it
       does). A mismatch is a warning, with its size.

    Both use one quasi-static solve at the final time sample.
    """
    if res.numel() == 0 or not photonic.srce_node:
        return
    with torch.no_grad():
        launched = sum(v.abs() ** 2 for v in photonic.srce_node.values())          # (F,)
        reference, _ = Simulate.apply(
            t_value[-1:], param_value[-1:], photonic.omega, photonic.node_has_ele,
            photonic.srce_node, photonic.node2ind, photonic.circuit_element,
            photonic.mod_element, photonic.mode_info, photonic.occur_order,
            (photonic.dout_node, photonic.middle_node), photonic.inward_node,
            photonic.outward_node, False)
        final = (res.detach()[-1].abs() ** 2).sum(-1)                              # (F,)
        steady = (reference[0].abs() ** 2).sum(-1)                                 # (F,)
        bound = torch.maximum(launched.real, steady).clamp_min(1e-300)
        runaway = float((final / bound).max())
        if runaway > 10.0:
            raise RuntimeError(
                f"mode='envelope' has gone unstable: at the end of the record the detected power "
                f"is {runaway:.4g}x both the launched power and this circuit's steady-state "
                f"output, and still growing. No passive circuit does that; the numbers are "
                f"meaningless. This is a known limitation of envelope mode with a modulator "
                f"inside an optical loop (e.g. a ring): its fractional-delay kernel can act as "
                f"gain for some carriers. Use mode='quasistatic', or keep modulators out of "
                f"optical loops.")

        # The output depends on the drive up to the largest positive tap offset in the past;
        # once the drive has been constant that long, the envelope must have settled.
        memory = max(int(offsets[-1]), 0) if len(offsets) else 0
        drive = param_value.detach()
        moving = (drive != drive[-1:]).any(dim=-1).nonzero()
        still_since = int(moving[-1]) + 1 if len(moving) else 0
        if len(t_value) - 1 - still_since >= memory:
            scale = float(reference.abs().max())
            if scale > 0:
                mismatch = float((res.detach()[-1] - reference[0]).abs().max()) / scale
                if mismatch > 1e-3:
                    warnings.warn(
                        f"mode='envelope' did not settle to the steady state: with the drive "
                        f"constant for the last {len(t_value) - 1 - still_since} samples (longer "
                        f"than the network's {memory}-sample memory), the detected field still "
                        f"differs from the quasi-static solution by {mismatch:.3g} of its peak. "
                        f"The transient before it is not trustworthy either. Compare against "
                        f"mode='quasistatic', refine .freq, or change the time step.",
                        RuntimeWarning, stacklevel=3)


def _selftest_quasistatic(photonic, t_value, param_value) -> float:
    """Largest relative difference between the reduced algebra and the full quasi-static solve.

    Not used by the simulator; kept because it is the one check that the ``u_0/G/z_0/D`` reduction
    really is the same system :meth:`Simulate.forward` assembles.
    """
    static, b, dynamic, _ = _assemble(photonic, t_value, param_value)
    obs_cols, detect_slice, _ = _observation_columns(photonic)
    transfer = _passive_transfer(static, b, dynamic, obs_cols)
    num_freq, num_port = transfer.u0.shape
    identity = torch.eye(num_port, dtype=config['complex_dtype'],
                         device=config['device']).expand(num_freq, -1, -1)
    out = []
    for i in range(len(t_value)):
        scatter = _block_scatter(dynamic, i, num_freq, num_port)
        u = torch.linalg.solve(identity - torch.matmul(transfer.G, scatter),
                               transfer.u0.unsqueeze(-1))
        y = torch.matmul(scatter, u)
        out.append(transfer.z0 + torch.matmul(transfer.D, y).squeeze(-1))
    reduced = torch.stack(out)[..., detect_slice]
    reference, _ = Simulate.apply(
        t_value, param_value, photonic.omega, photonic.node_has_ele, photonic.srce_node,
        photonic.node2ind, photonic.circuit_element, photonic.mod_element, photonic.mode_info,
        photonic.occur_order, (photonic.dout_node, photonic.middle_node), photonic.inward_node,
        photonic.outward_node, False)
    scale = float(reference.abs().max())
    return float((reduced - reference).abs().max()) / (scale if scale else 1.0)
