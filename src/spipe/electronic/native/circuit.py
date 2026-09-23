"""The public :class:`Netlist` object -- SPIPE's native SPICE engine."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from . import analyses as _an
from .adjoint import OpAutograd, TranAutograd, adjoint_sweep, op_adjoint
from .mna import CircuitError, MNASystem
from .newton import ConvergenceError, NewtonOptions
from .parser import parse_netlist
from .results import DCResult, OpResult, TranResult
from .units import SpiceSyntaxError

DTYPE = torch.float64

__all__ = ["Netlist", "UnknownParameterError", "CircuitError",
           "ConvergenceError", "SpiceSyntaxError"]


class UnknownParameterError(KeyError, ValueError):
    """Raised by :meth:`Netlist.param` for an unknown device or parameter.

    Inherits from both ``KeyError`` and ``ValueError`` so that either idiom
    catches it.
    """

    def __str__(self):  # keep the message readable (KeyError repr adds quotes)
        return self.args[0] if self.args else ""


_OPTION_KEYS = ("reltol", "abstol", "vntol", "maxiter", "dxmax", "gmin",
                "lte_reltol", "lte_abstol", "lte_reltol_nl")


class Netlist:
    """A parsed circuit, ready for analysis.

    >>> ckt = Netlist.from_string('''
    ... V1 in 0 PULSE(0 1 0 1p 1p 10n 20n)
    ... R1 in out 1k
    ... C1 out 0 1p
    ... ''')
    >>> res = ckt.tran(1e-12, 5e-9)
    >>> float(res.v('out')[-1])                       # doctest: +SKIP
    0.9932...

    Every numeric device parameter is a float64 leaf tensor, so

    >>> r = ckt.param('R1', 'R'); _ = r.requires_grad_(True)
    >>> res = ckt.tran(1e-12, 5e-9)
    >>> res.v('out').sum().backward()
    >>> r.grad is not None
    True
    """

    # ------------------------------------------------------------------
    def __init__(self, text: str = "", basedir: Optional[str] = None,
                 **options):
        self.text = text
        self.parsed = parse_netlist(text, basedir)
        self.sys = MNASystem(self.parsed, options)
        self.opts = NewtonOptions()
        self.method = "trap"
        self.nsub = None
        self.adaptive = True
        self.max_nsub = 64
        self.options: Dict[str, object] = {
            "method": self.method, "nsub": self.nsub,
            "adaptive": self.adaptive, "max_nsub": self.max_nsub,
            "temp": self.sys.temp_c,
        }
        for k in _OPTION_KEYS:
            self.options[k] = getattr(self.opts, k)
        for k, v in self.parsed.options.items():
            if k in self.options:
                self.options[k] = v
        self.options.update({k: v for k, v in options.items()
                             if k in self.options})
        self._last_tran: Optional[dict] = None
        self._last_rec = None

    # ------------------------------------------------------------------
    @classmethod
    def from_string(cls, text: str, **options) -> "Netlist":
        """Parse a netlist held in a string."""
        return cls(text, None, **options)

    @classmethod
    def from_file(cls, path: str, **options) -> "Netlist":
        """Parse a netlist file (``.include`` is resolved relative to it)."""
        with open(path, "r") as fh:
            text = fh.read()
        return cls(text, os.path.dirname(os.path.abspath(path)), **options)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> List[str]:
        """Node names, ground excluded, in netlist order of first appearance."""
        return self.sys.nodes

    @property
    def devices(self) -> List[str]:
        """Instance names of every device in the (flattened) circuit."""
        return list(self.sys.device_by_name)

    @property
    def thermal_voltage(self) -> float:
        """``kT/q`` at the simulation temperature (default 27 degC)."""
        return self.sys.vt

    def _nodemap(self) -> Dict[str, int]:
        return {nm: self.sys.node_index(nm) for nm in self.sys.nodes}

    def _branchmap(self) -> Dict[str, int]:
        out = {}
        for (name, k), off in self.sys._branch_map.items():
            if k == 0:
                out[name] = self.sys.n_nodes + off
        return out

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------
    def param(self, device: str, name: str) -> torch.Tensor:
        """The float64 **leaf tensor** holding one device parameter.

        >>> w = ckt.param('M1', 'W')      # doctest: +SKIP
        >>> w.requires_grad_(True)        # doctest: +SKIP

        Raises
        ------
        UnknownParameterError
            for an unknown device or an unknown parameter of a known device.
        """
        key = str(name)
        dev = self.sys.device_by_name.get(str(device).lower())
        if dev is not None:
            avail = dev.params
        else:
            avail = None
            for blk in self.sys.blocks:
                if str(getattr(blk, "name", "")).lower() == str(device).lower():
                    avail = (blk.parameters() or {}) if hasattr(blk, "parameters") else {}
                    break
            if avail is None:
                known = ", ".join(sorted(self.sys.device_by_name)
                                  + [str(getattr(b, "name", "?"))
                                     for b in self.sys.blocks]) or "(none)"
                raise UnknownParameterError(
                    "unknown device %r; the circuit contains: %s"
                    % (device, known))
        for cand in (key, key.upper(), key.lower()):
            if cand in avail:
                return avail[cand]
        raise UnknownParameterError(
            "device %r has no parameter %r; available parameters: %s"
            % (device, name, ", ".join(sorted(avail)) or "(none)"))

    def params(self) -> Dict[Tuple[str, str], torch.Tensor]:
        """Every device parameter, keyed by ``(device, parameter)``."""
        out = {}
        for name, dev in self.sys.device_by_name.items():
            for pname, tens in dev.params.items():
                out[(name, pname)] = tens
        for blk in self.sys.blocks:
            if hasattr(blk, "parameters"):
                for pname, tens in (blk.parameters() or {}).items():
                    out[(getattr(blk, "name", "block"), pname)] = tens
        return out

    def _leaves(self, only_grad=True) -> List[torch.Tensor]:
        seen, out = set(), []
        for t in self.params().values():
            if id(t) in seen:
                continue
            seen.add(id(t))
            if (not only_grad) or t.requires_grad:
                out.append(t)
        return out

    # ------------------------------------------------------------------
    # external blocks
    # ------------------------------------------------------------------
    def add_external_block(self, block) -> None:
        """Inject an external block's equations into this circuit's MNA system.

        See :mod:`spipe.electronic.native.external` for the contract.  The
        block owns ``block.n_states`` extra unknowns, couples to the circuit
        nodes listed in ``block.nodes`` (creating them if they do not exist)
        and supplies ``residual_and_jacobian(x, xdot, t)``.  Neither the MNA
        assembly nor the Newton loop needs any change to support a new block.
        """
        self.sys.add_external_block(block)
        self._last_rec = None

    # ------------------------------------------------------------------
    # option plumbing
    # ------------------------------------------------------------------
    def _sync_options(self) -> NewtonOptions:
        for k in _OPTION_KEYS:
            v = self.options.get(k)
            if v is not None:
                setattr(self.opts, k, type(getattr(self.opts, k))(v))
        self.method = str(self.options.get("method", self.method)).lower()
        self.nsub = self.options.get("nsub", self.nsub)
        self.adaptive = bool(self.options.get("adaptive", self.adaptive))
        self.max_nsub = int(self.options.get("max_nsub", self.max_nsub))
        self.sys.invalidate_param_cache()
        return self.opts

    # ------------------------------------------------------------------
    # analyses
    # ------------------------------------------------------------------
    def op(self) -> OpResult:
        """DC operating point.

        Returns a ``dict`` mapping node name -> float64 scalar tensor (with
        extra ``.v()``/``.i()`` accessors).  The result is differentiable via
        the DC adjoint.
        """
        opts = self._sync_options()
        rec = _an.solve_op(self.sys, opts, t=0.0, dc=True)
        self._last_op = rec
        x = rec.x
        leaves = self._leaves()
        if leaves:
            holder = {"sys": self.sys, "rec": rec, "params": leaves, "x": x}
            x = OpAutograd.apply(holder, *leaves)
        return OpResult(self.sys.nodes, x, self._nodemap(), self._branchmap())

    def tran(self, tstep=None, tstop=None, tstart=0.0, uic=False,
             method=None, nsub=None, adaptive=None, max_nsub=None) -> TranResult:
        """Transient analysis.

        Parameters
        ----------
        tstep, tstop : float
            print step and stop time; taken from the netlist's ``.tran`` card
            when omitted.
        tstart : float
            first output time (the circuit is always integrated from 0).
        uic : bool
            skip the operating point and start from ``.ic`` / ``IC=`` values.
        method : {'trap', 'be'}
            integration formula (trapezoidal with a backward-Euler start is
            the default; backward Euler is always used for the first step and
            immediately after a source breakpoint).
        nsub : int
            force a fixed number of internal steps per print interval
            (disables the LTE-driven step control).
        adaptive : bool
            enable LTE-driven step control (default ``True``).
        """
        opts = self._sync_options()
        if tstep is None or tstop is None:
            if not self.parsed.tran:
                raise CircuitError(
                    "tran() needs tstep and tstop, or a '.tran' card in the "
                    "netlist")
            ts, tp, t0, u = self.parsed.tran
            tstep = ts if tstep is None else tstep
            tstop = tp if tstop is None else tstop
            if tstart == 0.0:
                tstart = t0
            uic = uic or u
        args = dict(tstep=float(tstep), tstop=float(tstop),
                    tstart=float(tstart), uic=bool(uic),
                    method=(method or self.method),
                    nsub=(nsub if nsub is not None else self.nsub),
                    adaptive=(self.adaptive if adaptive is None else adaptive),
                    max_nsub=(self.max_nsub if max_nsub is None else max_nsub))
        rec = _an.run_tran(self.sys, opts, **args)
        self._last_tran = args
        self._last_rec = rec
        return self._wrap_tran(rec)

    def _wrap_tran(self, rec) -> TranResult:
        idx = torch.tensor(rec.out_idx, dtype=torch.long)
        Xout = rec.X.index_select(0, idx)
        leaves = self._leaves()
        if leaves:
            holder = {"sys": self.sys, "rec": rec, "params": leaves,
                      "Xout": Xout}
            Xout = TranAutograd.apply(holder, *leaves)
        return TranResult(rec.t_out, Xout, self.sys.nodes, self._nodemap(),
                          self._branchmap(), record=rec)

    def dc(self, source=None, start=None, stop=None, step=None) -> DCResult:
        """``.dc`` sweep of one independent source."""
        opts = self._sync_options()
        if source is None:
            if not self.parsed.dc_sweeps:
                raise CircuitError("dc() needs a source, or a '.dc' card")
            source, start, stop, step = self.parsed.dc_sweeps[0]
        sweep, X = _an.run_dc(self.sys, opts, source, start, stop, step)
        return DCResult(sweep, X, self.sys.nodes, self._nodemap(),
                        self._branchmap(), source=str(source))

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------
    def adjoint_grad(self, objective, params=None, tstep=None, tstop=None,
                     tstart=None, uic=None, reuse=False) -> Dict[Tuple[str, str], torch.Tensor]:
        """Gradient of *objective* by the explicit time-domain adjoint.

        Parameters
        ----------
        objective : callable
            takes the :class:`~.results.TranResult` and returns a scalar
            tensor.
        params : sequence of ``(device, parameter)``
            which parameters to differentiate with respect to; defaults to
            every parameter that currently has ``requires_grad=True``, and to
            *all* parameters if none does.
        tstep, tstop, tstart, uic :
            transient settings; default to the arguments of the most recent
            :meth:`tran` call, else to the netlist ``.tran`` card.
        reuse : bool
            reuse the most recent transient solution instead of re-running it.

        Returns
        -------
        dict
            ``{(device, parameter): float64 scalar tensor}``.

        Notes
        -----
        This is a *single backward sweep* over the stored time steps -- one
        transposed linear solve and one vector-Jacobian product per step --
        not repeated forward solves.
        """
        opts = self._sync_options()
        if params is None:
            keys = [k for k, v in self.params().items() if v.requires_grad]
            if not keys:
                keys = list(self.params())
        else:
            keys = [tuple(k) for k in params]
        leaves = [self.param(d, p) for d, p in keys]

        if reuse and self._last_rec is not None:
            rec = self._last_rec
        else:
            if self._last_tran:
                args = dict(self._last_tran)
            elif self.parsed.tran:
                ts, tp, t0, u = self.parsed.tran
                args = dict(tstep=ts, tstop=tp, tstart=t0, uic=u,
                            method=self.method, nsub=self.nsub,
                            adaptive=self.adaptive, max_nsub=self.max_nsub)
            else:
                args = dict(tstep=None, tstop=None, tstart=0.0, uic=False,
                            method=self.method, nsub=self.nsub,
                            adaptive=self.adaptive, max_nsub=self.max_nsub)
            if tstep is not None:
                args["tstep"] = float(tstep)
            if tstop is not None:
                args["tstop"] = float(tstop)
            if tstart is not None:
                args["tstart"] = float(tstart)
            if uic is not None:
                args["uic"] = bool(uic)
            if args["tstep"] is None or args["tstop"] is None:
                raise CircuitError(
                    "adjoint_grad() needs tstep/tstop (or a previous tran() "
                    "call, or a '.tran' card in the netlist)")
            rec = _an.run_tran(self.sys, opts, **args)
            self._last_tran = args
            self._last_rec = rec

        idx = torch.tensor(rec.out_idx, dtype=torch.long)
        Xout = rec.X.index_select(0, idx).detach().clone().requires_grad_(True)
        res = TranResult(rec.t_out, Xout, self.sys.nodes, self._nodemap(),
                         self._branchmap(), record=rec)
        obj = objective(res)
        if not isinstance(obj, torch.Tensor) or obj.numel() != 1:
            raise ValueError("objective must return a scalar tensor")
        gX, = torch.autograd.grad(obj, Xout)
        grads = adjoint_sweep(self.sys, rec, gX, leaves)
        return {k: g for k, g in zip(keys, grads)}

    def op_adjoint_grad(self, objective, params=None) -> Dict[Tuple[str, str], torch.Tensor]:
        """Adjoint gradient of a DC-operating-point objective."""
        opts = self._sync_options()
        if params is None:
            keys = [k for k, v in self.params().items() if v.requires_grad]
            if not keys:
                keys = list(self.params())
        else:
            keys = [tuple(k) for k in params]
        leaves = [self.param(d, p) for d, p in keys]
        rec = _an.solve_op(self.sys, opts, t=0.0, dc=True)
        x = rec.x.detach().clone().requires_grad_(True)
        res = OpResult(self.sys.nodes, x, self._nodemap(), self._branchmap())
        obj = objective(res)
        gx, = torch.autograd.grad(obj, x)
        grads = op_adjoint(self.sys, rec, gx, leaves)
        return {k: g for k, g in zip(keys, grads)}

    # ------------------------------------------------------------------
    def run(self):
        """Run whatever analyses the netlist's control cards request."""
        out = {}
        if self.parsed.has_op or not (self.parsed.tran or self.parsed.dc_sweeps):
            out["op"] = self.op()
        if self.parsed.tran:
            out["tran"] = self.tran()
        if self.parsed.dc_sweeps:
            out["dc"] = self.dc()
        return out

    def __repr__(self):  # pragma: no cover
        return "<Netlist %d devices, %d nodes, %d unknowns>" % (
            len(self.sys.device_by_name), len(self.sys.nodes), self.sys.n)
