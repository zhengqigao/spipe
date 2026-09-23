"""Device base classes and the MNA stamping context.

Every device type is a :class:`DeviceGroup` subclass that owns *all* instances
of that type in the circuit.  Instances are evaluated in a vectorised way
(one batched torch expression per device type), which is what makes a pure
PyTorch SPICE fast enough to be useful.

The system is written as a semi-explicit DAE

.. math::

    f(x, t) + \\frac{d}{dt} q(x) = 0

where ``f`` collects the *static* (resistive / algebraic) contributions and
``q`` collects *charges* (capacitors) and *fluxes* (inductors).  Integration
is applied to ``q`` only, which is what makes the scheme charge conserving.

A group therefore only has to implement :meth:`DeviceGroup.load`, adding its
contributions to ``ctx.f`` / ``ctx.q`` and (optionally) the corresponding
Jacobians ``df/dx`` and ``dq/dx``.  The solver never needs to know which
devices exist.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

DTYPE = torch.float64

__all__ = ["DTYPE", "LoadContext", "DeviceGroup", "Device", "pnjlim", "limexp", "GMIN_DEFAULT"]

GMIN_DEFAULT = 1e-12


class Device:
    """One device instance.

    Attributes
    ----------
    name : str
        instance name, lowercase (hierarchical for sub-circuit instances).
    nodes : list[str]
        connection node names.
    params : dict[str, torch.Tensor]
        float64 **leaf** tensors -- these are what ``Netlist.param()`` returns
        and what gradients are taken with respect to.
    """

    __slots__ = ("name", "nodes", "params", "model", "extra")

    def __init__(self, name, nodes, params, model=None, extra=None):
        self.name = name
        self.nodes = list(nodes)
        self.params: Dict[str, torch.Tensor] = params
        self.model = model
        self.extra = extra or {}

    def __repr__(self):  # pragma: no cover - debug helper
        return "<%s %s %s>" % (type(self).__name__, self.name, self.nodes)


class LoadContext:
    """Accumulator handed to :meth:`DeviceGroup.load`.

    Vectors have length ``n+1``; index ``n`` is the *ground bin* -- stamps that
    would land on ground are written there and discarded afterwards.  This
    removes all branching on "is this node ground?" from the device code.
    """

    __slots__ = ("x", "xold", "t", "n", "need_jac", "gmin", "src_scale", "dc",
                 "limiting", "limited", "_fi", "_fv", "_qi", "_qv", "_ji",
                 "_jv", "_ki", "_kv")

    def __init__(self, x, t, n, need_jac=True, xold=None, gmin=0.0,
                 src_scale=1.0, dc=False, limiting=True):
        self.x = x
        self.xold = x if xold is None else xold
        self.t = t
        self.n = n
        self.need_jac = need_jac
        self.gmin = gmin
        self.src_scale = src_scale
        self.dc = dc
        self.limiting = limiting
        #: set by a device when its voltage limiter actually changed a bias;
        #: Newton must not declare convergence while this is true
        self.limited = False
        # contributions are collected and scattered once, in finish()
        self._fi = []
        self._fv = []
        self._qi = []
        self._qv = []
        self._ji = []      # (rows, cols) pairs for df/dx
        self._jv = []
        self._ki = []      # (rows, cols) pairs for dq/dx
        self._kv = []

    # -- accumulation helpers ------------------------------------------------
    def add_f(self, idx: torch.Tensor, vals: torch.Tensor) -> None:
        """Add *vals* into the residual rows *idx*."""
        self._fi.append(idx)
        self._fv.append(vals)

    def add_q(self, idx: torch.Tensor, vals: torch.Tensor) -> None:
        """Add *vals* into the charge/flux rows *idx*."""
        self._qi.append(idx)
        self._qv.append(vals)

    def add_jf(self, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor) -> None:
        if self.need_jac:
            self._ji.append((rows, cols))
            self._jv.append(vals)

    def add_jq(self, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor) -> None:
        if self.need_jac:
            self._ki.append((rows, cols))
            self._kv.append(vals)

    def _vec(self, idx, vals):
        n = self.n
        out = torch.zeros(n + 1, dtype=DTYPE)
        if not idx:
            return out[:n]
        if len(idx) == 1:
            return out.index_add(0, idx[0], vals[0])[:n]
        return out.index_add(0, torch.cat(idx), torch.cat(vals))[:n]

    def _mat(self, idx, vals):
        n = self.n
        out = torch.zeros(n + 1, n + 1, dtype=DTYPE)
        if idx:
            rows = torch.cat([r for r, _ in idx])
            cols = torch.cat([c for _, c in idx])
            v = torch.cat(vals)
            out.index_put_((rows, cols), v.detach(), accumulate=True)
        return out[:n, :n]

    def magnitudes(self):
        """Row-wise sum of |contribution| -- the SPICE convergence scale.

        The residual test needs the size of the *terms* that make up each
        KCL row, not the size of the Jacobian: a diode biased far into
        forward conduction has an enormous conductance but the currents that
        must cancel are what decides convergence.
        """
        n = self.n
        fm = torch.zeros(n + 1, dtype=DTYPE)
        if self._fi:
            fm = fm.index_add(0, torch.cat(self._fi) if len(self._fi) > 1
                              else self._fi[0],
                              (torch.cat(self._fv) if len(self._fv) > 1
                               else self._fv[0]).detach().abs())
        qm = torch.zeros(n + 1, dtype=DTYPE)
        if self._qi:
            qm = qm.index_add(0, torch.cat(self._qi) if len(self._qi) > 1
                              else self._qi[0],
                              (torch.cat(self._qv) if len(self._qv) > 1
                               else self._qv[0]).detach().abs())
        return fm[:n], qm[:n]

    def finish(self):
        """Drop the ground bin and return ``(f, q, Jf, Jq)``."""
        f = self._vec(self._fi, self._fv)
        q = self._vec(self._qi, self._qv)
        if not self.need_jac:
            return f, q, None, None
        return f, q, self._mat(self._ji, self._jv), self._mat(self._ki, self._kv)


class DeviceGroup:
    """Base class for a vectorised collection of same-type devices.

    Subclasses must define

    ``letter``
        the SPICE device letter they handle.
    ``PARAMS``
        ``{name: (default, description)}`` -- the documented parameter table.
    ``build(elem, ctx)``
        classmethod creating one :class:`Device` from a parsed ``Element``.
    ``load(ctx)``
        stamp all instances.
    """

    letter = "?"
    PARAMS: Dict[str, tuple] = {}
    #: number of extra MNA branch unknowns each instance needs
    n_branch = 0
    #: True if the group can contribute to the charge/flux vector q(x)
    has_charge = False

    def __init__(self, devices: Sequence[Device], circuit):
        self.devices: List[Device] = list(devices)
        self.circuit = circuit
        self.names = [d.name for d in self.devices]
        self.N = len(self.devices)
        self._pcache: Dict[str, torch.Tensor] = {}
        self._idx_cache: Dict[str, torch.Tensor] = {}

    # -- indexing helpers ----------------------------------------------------
    def node_idx(self, k: int) -> torch.Tensor:
        """``(N,)`` long tensor of global indices for the *k*-th node of every
        instance (ground maps to the ground bin)."""
        key = "n%d" % k
        t = self._idx_cache.get(key)
        if t is None:
            t = torch.tensor(
                [self.circuit.node_index(d.nodes[k]) for d in self.devices],
                dtype=torch.long,
            )
            self._idx_cache[key] = t
        return t

    def branch_idx(self, k: int = 0) -> torch.Tensor:
        key = "b%d" % k
        t = self._idx_cache.get(key)
        if t is None:
            t = torch.tensor(
                [self.circuit.branch_index(d.name, k) for d in self.devices],
                dtype=torch.long,
            )
            self._idx_cache[key] = t
        return t

    # -- parameter helpers ---------------------------------------------------
    def p(self, name: str) -> torch.Tensor:
        """Stacked ``(N,)`` parameter tensor.

        When :attr:`circuit.attach_params` is true the stack keeps the autograd
        graph back to the per-instance leaf tensors (used by the adjoint and by
        the autograd path); otherwise a detached cache is returned, which is
        what the Newton loop uses.
        """
        if self.circuit.attach_params:
            return torch.stack([d.params[name] for d in self.devices])
        t = self._pcache.get(name)
        if t is None:
            t = torch.stack([d.params[name] for d in self.devices]).detach()
            self._pcache[name] = t
        return t

    def invalidate_cache(self):
        self._pcache.clear()

    # -- to be provided by subclasses ---------------------------------------
    def load(self, ctx: LoadContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def breakpoints(self, tstart, tstop):
        """Time points that the transient analysis must land on exactly."""
        return []

    def __repr__(self):  # pragma: no cover
        return "<%s x%d>" % (type(self).__name__, self.N)


# --------------------------------------------------------------------------
# convergence aids
# --------------------------------------------------------------------------

def limexp(x: torch.Tensor, xmax: float = 80.0) -> torch.Tensor:
    """``exp`` linearised above *xmax*.

    Keeps the model exact for every physically sensible bias while making the
    Newton iteration immune to float64 overflow when an intermediate iterate
    runs away.  ``exp(80) ~ 5.5e34``, so with a typical ``IS`` of 1e-14 the
    linearisation only ever engages far outside the solution region.
    """
    lin = x > xmax
    xs = torch.where(lin, torch.full_like(x, xmax), x)
    e = torch.exp(xs)
    return torch.where(lin, e * (1.0 + x - xmax), e)


def pnjlim(vnew: torch.Tensor, vold: torch.Tensor, vt: torch.Tensor,
           vcrit: torch.Tensor) -> torch.Tensor:
    """Berkeley SPICE ``pnjlim`` junction voltage limiter (vectorised).

    Limiting is inactive once the iteration is converged
    (``|vnew - vold| <= 2*vt``), so the converged solution satisfies the
    *unmodified* device equations.
    """
    tiny = torch.finfo(vnew.dtype).tiny
    dv = vnew - vold
    act = (vnew > vcrit) & (dv.abs() > 2.0 * vt)
    arg = 1.0 + dv / vt
    v_pos = torch.where(
        arg > 0.0,
        vold + vt * torch.log(torch.clamp(arg, min=tiny)),
        vcrit,
    )
    v_neg = vt * torch.log(torch.clamp(vnew / vt, min=tiny))
    return torch.where(act, torch.where(vold > 0.0, v_pos, v_neg), vnew)


def fetlim(vnew: torch.Tensor, vold: torch.Tensor, vto: torch.Tensor,
           maxstep: float = 2.0) -> torch.Tensor:
    """Simple MOSFET gate-voltage limiter: bound the per-iteration change.

    The Shichman-Hodges current is polynomial, so the aggressive Berkeley
    ``fetlim`` is unnecessary; bounding |dv| keeps the iteration out of the
    far-field where the ``(1+lambda*vds)`` factor can change sign.
    """
    step = torch.clamp(vnew - vold, min=-maxstep, max=maxstep)
    return vold + step
