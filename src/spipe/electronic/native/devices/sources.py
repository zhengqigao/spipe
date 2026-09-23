"""Independent sources and their time functions.

Supported functions: ``DC``, ``PULSE``, ``PWL``, ``SIN`` and (as a bonus)
``EXP``.  Every waveform parameter is an individually addressable float64 leaf
tensor, e.g. ``ckt.param('V1', 'V2')`` for the pulse high level.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from ..units import SpiceSyntaxError, eval_expr, is_number
from .base import DTYPE, Device, DeviceGroup

__all__ = ["VoltageSource", "CurrentSource", "make_waveform"]

_FUNCS = ("dc", "pulse", "pwl", "sin", "sine", "exp", "sffm", "ac")


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


def _flatten_args(args) -> List[str]:
    """Split ``PULSE(0 5 1n)`` style tokens into flat lowercase tokens."""
    out: List[str] = []
    for a in args:
        s = str(a).strip()
        if "(" in s:
            head, _, tail = s.partition("(")
            if head:
                out.append(head)
            tail = tail.rstrip(")")
            out.extend(tok for tok in tail.replace(",", " ").split() if tok)
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------
# waveforms
# --------------------------------------------------------------------------

class Waveform:
    """Base class for a source time-function."""

    kind = "dc"
    names: List[str] = []

    def value(self, t, p):  # pragma: no cover - abstract
        raise NotImplementedError

    def dc_value(self, p):
        """Value used for the DC operating point (the t=0 value by default)."""
        return self.value(0.0, p)

    def value_batch(self, t, p):
        """``(K,)`` values at the K times in *t*.  Overridden where the scalar
        form takes a Python branch on ``t``."""
        return torch.broadcast_to(self.value(t, p), t.shape)

    def breakpoints(self, p, tstart, tstop):
        return []


class DCWave(Waveform):
    """Constant source.  Parameter: ``DC``."""

    kind = "dc"
    names = ["DC"]

    def value(self, t, p):
        return p["DC"]


class PulseWave(Waveform):
    """``PULSE(V1 V2 TD TR TF PW PER)``.

    Parameters ``V1 V2 TD TR TF PW PER``; ``TR``/``TF`` default to the
    transient print step and ``PW``/``PER`` to ``tstop`` as in SPICE.
    """

    kind = "pulse"
    names = ["V1", "V2", "TD", "TR", "TF", "PW", "PER"]

    def value(self, t, p):
        tt = torch.as_tensor(t, dtype=DTYPE) - p["TD"]
        per = p["PER"]
        tr, tf, pw = p["TR"], p["TF"], p["PW"]
        perf = float(per.detach())
        if perf > 0 and float(tt.detach()) > 0:
            k = math.floor(float(tt.detach()) / perf)
            if k > 0:
                tt = tt - k * per
        zero = torch.zeros((), dtype=DTYPE)
        v1, v2 = p["V1"], p["V2"]
        trf = torch.clamp(tr, min=1e-300)
        tff = torch.clamp(tf, min=1e-300)
        v_rise = v1 + (v2 - v1) * torch.clamp(tt, min=zero) / trf
        v_fall = v2 + (v1 - v2) * (tt - tr - pw) / tff
        out = torch.where(tt < 0.0, v1,
              torch.where(tt < tr, v_rise,
              torch.where(tt < tr + pw, v2,
              torch.where(tt < tr + pw + tf, v_fall, v1))))
        return out

    def value_batch(self, t, p):
        tt = t - p["TD"]
        per, tr, tf, pw = p["PER"], p["TR"], p["TF"], p["PW"]
        if float(per.detach()) > 0:
            tt = tt - torch.floor(torch.clamp(tt, min=0.0) / per) * per
        v1, v2 = p["V1"], p["V2"]
        trf = torch.clamp(tr, min=1e-300)
        tff = torch.clamp(tf, min=1e-300)
        v_rise = v1 + (v2 - v1) * torch.clamp(tt, min=0.0) / trf
        v_fall = v2 + (v1 - v2) * (tt - tr - pw) / tff
        return torch.where(tt < 0.0, v1,
               torch.where(tt < tr, v_rise,
               torch.where(tt < tr + pw, v2,
               torch.where(tt < tr + pw + tf, v_fall, v1))))

    def breakpoints(self, p, tstart, tstop):
        f = {k: float(v.detach()) for k, v in p.items()}
        td, tr, tf, pw, per = f["TD"], f["TR"], f["TF"], f["PW"], f["PER"]
        edges = [0.0, tr, tr + pw, tr + pw + tf]
        bps = []
        if per > 0 and (tr + pw + tf) > 0:
            k = 0
            while td + k * per <= tstop and k < 1000000:
                base = td + k * per
                for e in edges:
                    tb = base + e
                    if tstart <= tb <= tstop:
                        bps.append(tb)
                k += 1
                if per <= 0:
                    break
        else:
            for e in edges:
                tb = td + e
                if tstart <= tb <= tstop:
                    bps.append(tb)
        return bps


class PWLWave(Waveform):
    """``PWL(t0 v0 t1 v1 ...)`` piecewise linear, constant outside the range.

    Parameters are named ``T0 V0 T1 V1 ...``.
    """

    kind = "pwl"

    def __init__(self, npts):
        self.npts = npts
        self.names = []
        for i in range(npts):
            self.names += ["T%d" % i, "V%d" % i]

    def value(self, t, p):
        tt = torch.as_tensor(t, dtype=DTYPE)
        ts = [p["T%d" % i] for i in range(self.npts)]
        vs = [p["V%d" % i] for i in range(self.npts)]
        out = vs[0]
        for i in range(self.npts - 1):
            dt = torch.clamp(ts[i + 1] - ts[i], min=1e-300)
            seg = vs[i] + (vs[i + 1] - vs[i]) * (tt - ts[i]) / dt
            inside = (tt >= ts[i]) & (tt < ts[i + 1])
            out = torch.where(inside, seg, out)
        out = torch.where(tt >= ts[-1], vs[-1], out)
        out = torch.where(tt < ts[0], vs[0], out)
        return out

    def breakpoints(self, p, tstart, tstop):
        return [
            float(p["T%d" % i].detach())
            for i in range(self.npts)
            if tstart <= float(p["T%d" % i].detach()) <= tstop
        ]


class SinWave(Waveform):
    """``SIN(VO VA FREQ TD THETA PHASE)`` -- damped sine, phase in degrees."""

    kind = "sin"
    names = ["VO", "VA", "FREQ", "TD", "THETA", "PHASE"]

    def value(self, t, p):
        tt = torch.as_tensor(t, dtype=DTYPE) - p["TD"]
        ph = p["PHASE"] * (math.pi / 180.0)
        arg = 2.0 * math.pi * p["FREQ"] * torch.clamp(tt, min=0.0) + ph
        damp = torch.exp(-torch.clamp(tt, min=0.0) * p["THETA"])
        v = p["VO"] + p["VA"] * damp * torch.sin(arg)
        v0 = p["VO"] + p["VA"] * torch.sin(ph)
        return torch.where(tt < 0.0, v0, v)

    def breakpoints(self, p, tstart, tstop):
        td = float(p["TD"].detach())
        return [td] if tstart <= td <= tstop else []


class ExpWave(Waveform):
    """``EXP(V1 V2 TD1 TAU1 TD2 TAU2)``."""

    kind = "exp"
    names = ["V1", "V2", "TD1", "TAU1", "TD2", "TAU2"]

    def value(self, t, p):
        tt = torch.as_tensor(t, dtype=DTYPE)
        v1, v2 = p["V1"], p["V2"]
        td1, tau1, td2, tau2 = p["TD1"], p["TAU1"], p["TD2"], p["TAU2"]
        t1 = torch.clamp(tt - td1, min=0.0)
        t2 = torch.clamp(tt - td2, min=0.0)
        r1 = 1.0 - torch.exp(-t1 / torch.clamp(tau1, min=1e-300))
        r2 = 1.0 - torch.exp(-t2 / torch.clamp(tau2, min=1e-300))
        return v1 + (v2 - v1) * (torch.where(tt < td1, torch.zeros_like(r1), r1)
                                 - torch.where(tt < td2, torch.zeros_like(r2), r2))

    def breakpoints(self, p, tstart, tstop):
        out = []
        for k in ("TD1", "TD2"):
            tb = float(p[k].detach())
            if tstart <= tb <= tstop:
                out.append(tb)
        return out


_PULSE_DEFAULTS = {"V1": 0.0, "V2": 0.0, "TD": 0.0, "TR": None, "TF": None,
                   "PW": None, "PER": None}
_SIN_DEFAULTS = {"VO": 0.0, "VA": 0.0, "FREQ": None, "TD": 0.0,
                 "THETA": 0.0, "PHASE": 0.0}
_EXP_DEFAULTS = {"V1": 0.0, "V2": 0.0, "TD1": 0.0, "TAU1": None,
                 "TD2": None, "TAU2": None}


def make_waveform(elem):
    """Build ``(waveform, params, defaulted)`` from a parsed source card."""
    toks = _flatten_args(elem.args)
    scope = elem.params
    dc_val = None
    kind = None
    vals: List[str] = []

    # KEY=VALUE forms such as `V1 a 0 DC=1`
    for k in ("dc", "value"):
        if k in elem.kwargs:
            dc_val = eval_expr(elem.kwargs[k], scope)

    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t == "dc":
            i += 1
            if i < len(toks):
                dc_val = eval_expr(toks[i], scope)
                i += 1
            continue
        if t == "ac":
            i += 1
            while i < len(toks) and is_number(toks[i]):
                i += 1
            continue
        if t in ("pulse", "pwl", "sin", "sine", "exp", "sffm"):
            kind = "sin" if t == "sine" else t
            i += 1
            vals = []
            while i < len(toks) and not toks[i].lower() in _FUNCS:
                vals.append(toks[i])
                i += 1
            continue
        if kind is None and dc_val is None:
            try:
                dc_val = eval_expr(toks[i], scope)
            except SpiceSyntaxError:
                pass
        i += 1

    defaulted = set()
    if kind is None:
        if dc_val is None:
            raise SpiceSyntaxError("source %s has no value: %r" % (elem.name, elem.source))
        return DCWave(), {"DC": _leaf(dc_val)}, defaulted

    nums = [eval_expr(v, scope) for v in vals]

    if kind == "pulse":
        wf = PulseWave()
        p = {}
        order = wf.names
        for j, nm in enumerate(order):
            if j < len(nums):
                p[nm] = _leaf(nums[j])
            else:
                d = _PULSE_DEFAULTS[nm]
                defaulted.add(nm)
                p[nm] = _leaf(0.0 if d is None else d)
        if dc_val is not None:
            p["DC"] = _leaf(dc_val)
        return wf, p, defaulted

    if kind == "pwl":
        if len(nums) < 2 or len(nums) % 2:
            raise SpiceSyntaxError("PWL needs time/value pairs: %r" % (elem.source,))
        n = len(nums) // 2
        wf = PWLWave(n)
        p = {}
        for j in range(n):
            p["T%d" % j] = _leaf(nums[2 * j])
            p["V%d" % j] = _leaf(nums[2 * j + 1])
        if dc_val is not None:
            p["DC"] = _leaf(dc_val)
        return wf, p, defaulted

    if kind == "sin":
        wf = SinWave()
        p = {}
        for j, nm in enumerate(wf.names):
            if j < len(nums):
                p[nm] = _leaf(nums[j])
            else:
                d = _SIN_DEFAULTS[nm]
                defaulted.add(nm)
                p[nm] = _leaf(0.0 if d is None else d)
        if dc_val is not None:
            p["DC"] = _leaf(dc_val)
        return wf, p, defaulted

    if kind == "exp":
        wf = ExpWave()
        p = {}
        for j, nm in enumerate(wf.names):
            if j < len(nums):
                p[nm] = _leaf(nums[j])
            else:
                defaulted.add(nm)
                p[nm] = _leaf(0.0)
        if dc_val is not None:
            p["DC"] = _leaf(dc_val)
        return wf, p, defaulted

    raise SpiceSyntaxError("unsupported source function %r" % kind)


# --------------------------------------------------------------------------
# source device groups
# --------------------------------------------------------------------------

class _SourceGroup(DeviceGroup):
    def __init__(self, devices, circuit):
        super().__init__(devices, circuit)
        self._cache_key = None
        self._cache_val = None

    def invalidate_cache(self):
        super().invalidate_cache()
        self._cache_key = None
        self._cache_val = None

    def set_tran_defaults(self, tstep, tstop):
        """Apply the SPICE defaults for omitted ``PULSE``/``SIN`` parameters."""
        for d in self.devices:
            wf = d.extra["wf"]
            miss = d.extra["defaulted"]
            if isinstance(wf, PulseWave):
                for nm, val in (("TR", tstep), ("TF", tstep),
                                ("PW", tstop), ("PER", tstop)):
                    if nm in miss:
                        with torch.no_grad():
                            d.params[nm].copy_(torch.tensor(float(val), dtype=DTYPE))
            elif isinstance(wf, SinWave):
                if "FREQ" in miss and tstop > 0:
                    with torch.no_grad():
                        d.params["FREQ"].copy_(torch.tensor(1.0 / tstop, dtype=DTYPE))
        self.invalidate_cache()
        self._cache_key = None

    def values(self, ctx):
        """``(N,)`` tensor of source values at ``ctx.t`` (scaled for source stepping)."""
        return self.values_at(ctx.t, ctx.dc, ctx.src_scale)

    def values_at(self, t, dc=False, src_scale=1.0):
        """``(N,)`` tensor of source values at time *t*."""
        attach = self.circuit.attach_params
        key = (t, dc, attach)
        if not attach and self._cache_key == key:
            vals = self._cache_val
        else:
            out = []
            for d in self.devices:
                wf = d.extra["wf"]
                p = d.params
                if not attach:
                    p = {k: v.detach() for k, v in p.items()}
                if dc:
                    # pure .op: use the DC value if the card gives one, else the
                    # t=0 value of the transient waveform (SPICE behaviour)
                    out.append(p["DC"] if "DC" in p else wf.value(0.0, p))
                else:
                    out.append(wf.value(t, p))
            vals = torch.stack(out)
            if not attach:
                self._cache_key = key
                self._cache_val = vals
        return vals if src_scale == 1.0 else vals * src_scale

    def values_batch(self, t):
        """``(K, N)`` source values at the K times in the 1-D tensor *t*.

        Evaluating a whole time window in one shot is what lets the linear
        transient kernel avoid a per-step Python call into every source.
        """
        out = []
        for d in self.devices:
            wf = d.extra["wf"]
            p = {k: v.detach() for k, v in d.params.items()}
            out.append(wf.value_batch(t, p))
        return torch.stack(out, dim=1)

    def breakpoints(self, tstart, tstop):
        out = []
        for d in self.devices:
            out.extend(d.extra["wf"].breakpoints(d.params, tstart, tstop))
        return out


class VoltageSource(_SourceGroup):
    """Independent voltage source.  Adds one branch-current unknown.

    Card: ``Vname n+ n- [DC <val>] [PULSE(...)|PWL(...)|SIN(...)|EXP(...)]``

    ``res.i('V1')`` returns the branch current, positive flowing from ``n+``
    through the source to ``n-`` (SPICE sign convention).
    """

    letter = "v"
    n_branch = 1
    PARAMS = {"<waveform>": (None, "see make_waveform(); e.g. DC, V1..PER, VO..PHASE")}

    @classmethod
    def build(cls, elem, circuit):
        wf, p, defaulted = make_waveform(elem)
        return Device(elem.name, elem.nodes, p,
                      extra={"wf": wf, "defaulted": defaulted})

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        br = self.branch_idx()
        i = ctx.x[br]
        ctx.add_f(n1, i)
        ctx.add_f(n2, -i)
        ctx.add_f(br, ctx.x[n1] - ctx.x[n2] - self.values(ctx))
        if ctx.need_jac:
            one = torch.ones(self.N, dtype=DTYPE)
            rows = torch.cat([n1, n2, br, br])
            cols = torch.cat([br, br, n1, n2])
            ctx.add_jf(rows, cols, torch.cat([one, -one, one, -one]))


class CurrentSource(_SourceGroup):
    """Independent current source; positive current flows from ``n+`` to ``n-``
    *inside* the source (SPICE convention)."""

    letter = "i"
    PARAMS = {"<waveform>": (None, "see make_waveform()")}

    @classmethod
    def build(cls, elem, circuit):
        wf, p, defaulted = make_waveform(elem)
        return Device(elem.name, elem.nodes, p,
                      extra={"wf": wf, "defaulted": defaulted})

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        val = self.values(ctx)
        ctx.add_f(n1, val)
        ctx.add_f(n2, -val)
