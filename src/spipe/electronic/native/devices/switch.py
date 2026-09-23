"""Voltage- and current-controlled switches (Tier 3).

Uses the smooth (C1) log-conductance interpolation between ``VOFF`` and
``VON`` that PSpice/LTspice use, which keeps Newton well behaved.
"""

from __future__ import annotations

import math

import torch

from ..units import eval_expr
from .base import DTYPE, Device, DeviceGroup

__all__ = ["VCSwitch", "CCSwitch"]

_SW_DEFAULTS = {"RON": 1.0, "ROFF": 1e12, "VON": 1.0, "VOFF": 0.0}
_CSW_DEFAULTS = {"RON": 1.0, "ROFF": 1e12, "ION": 1e-3, "IOFF": 0.0}


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


def _switch_conductance(ctrl, gon, goff, con, coff):
    """Smooth conductance ramp; returns ``(g, dg/dctrl)``."""
    lgon, lgoff = torch.log(gon), torch.log(goff)
    lm = lgon - lgoff
    lr = 0.5 * (lgon + lgoff)
    span = con - coff
    span = torch.where(span.abs() < 1e-30, torch.full_like(span, 1e-30), span)
    vd = (ctrl - 0.5 * (con + coff)) / span
    vdc = torch.clamp(vd, -0.5, 0.5)
    g = torch.exp(lr + 1.5 * lm * vdc - 2.0 * lm * vdc ** 3)
    dg = torch.where(
        (vd > -0.5) & (vd < 0.5),
        g * (1.5 * lm - 6.0 * lm * vdc * vdc) / span,
        torch.zeros_like(g),
    )
    return g, dg


class VCSwitch(DeviceGroup):
    """Voltage-controlled switch ``S``.

    Card: ``Sname n+ n- nc+ nc- <model>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``RON``      1.0      on resistance [ohm]
    ``ROFF``     1e12     off resistance [ohm]
    ``VON``      1.0      control voltage for fully on [V]
    ``VOFF``     0.0      control voltage for fully off [V]
    ============ ======== ==================================================

    ``VT``/``VH`` model cards are translated to ``VON = VT+VH``,
    ``VOFF = VT-VH``.
    """

    letter = "s"
    PARAMS = {k: (v, "") for k, v in _SW_DEFAULTS.items()}

    @classmethod
    def build(cls, elem, circuit):
        mp = circuit.model_params(elem.model, ("sw", "vswitch", "switch"), elem.name)
        vals = dict(_SW_DEFAULTS)
        low = {k.lower(): v for k, v in mp.items()}
        if "vt" in low:
            vh = float(low.get("vh", 0.0)) or 1e-3
            vt = float(low["vt"])
            vals["VON"] = vt + vh
            vals["VOFF"] = vt - vh
        for k in ("ron", "roff", "von", "voff"):
            if k in low:
                vals[k.upper()] = float(low[k])
        for k, v in elem.kwargs.items():
            if k.upper() in vals:
                vals[k.upper()] = eval_expr(v, elem.params)
        return Device(elem.name, elem.nodes, {k: _leaf(v) for k, v in vals.items()},
                      model=elem.model)

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        cp, cm = self.node_idx(2), self.node_idx(3)
        gon = 1.0 / torch.clamp(self.p("RON"), min=1e-30)
        goff = 1.0 / torch.clamp(self.p("ROFF"), min=1e-30)
        vc = ctx.x[cp] - ctx.x[cm]
        g, dg = _switch_conductance(vc, gon, goff, self.p("VON"), self.p("VOFF"))
        v = ctx.x[n1] - ctx.x[n2]
        i = g * v
        ctx.add_f(n1, i)
        ctx.add_f(n2, -i)
        if ctx.need_jac:
            didc = dg * v
            rows = torch.cat([n1, n1, n2, n2, n1, n1, n2, n2])
            cols = torch.cat([n1, n2, n1, n2, cp, cm, cp, cm])
            vals = torch.cat([g, -g, -g, g, didc, -didc, -didc, didc])
            ctx.add_jf(rows, cols, vals)


class CCSwitch(DeviceGroup):
    """Current-controlled switch ``W``.

    Card: ``Wname n+ n- <Vcontrol> <model>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``RON``      1.0      on resistance [ohm]
    ``ROFF``     1e12     off resistance [ohm]
    ``ION``      1e-3     control current for fully on [A]
    ``IOFF``     0.0      control current for fully off [A]
    ============ ======== ==================================================
    """

    letter = "w"
    PARAMS = {k: (v, "") for k, v in _CSW_DEFAULTS.items()}

    @classmethod
    def build(cls, elem, circuit):
        mp = circuit.model_params(elem.model, ("csw", "iswitch", "switch"), elem.name)
        vals = dict(_CSW_DEFAULTS)
        low = {k.lower(): v for k, v in mp.items()}
        if "it" in low:
            ih = float(low.get("ih", 0.0)) or 1e-9
            it = float(low["it"])
            vals["ION"] = it + ih
            vals["IOFF"] = it - ih
        for k in ("ron", "roff", "ion", "ioff"):
            if k in low:
                vals[k.upper()] = float(low[k])
        for k, v in elem.kwargs.items():
            if k.upper() in vals:
                vals[k.upper()] = eval_expr(v, elem.params)
        return Device(elem.name, elem.nodes, {k: _leaf(v) for k, v in vals.items()},
                      model=elem.model, extra={"ctrl": elem.ctrl})

    def ctrl_idx(self):
        t = self._idx_cache.get("ctrl")
        if t is None:
            t = torch.tensor([self.circuit.ctrl_branch_index(d.extra["ctrl"], d.name)
                              for d in self.devices], dtype=torch.long)
            self._idx_cache["ctrl"] = t
        return t

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        cb = self.ctrl_idx()
        gon = 1.0 / torch.clamp(self.p("RON"), min=1e-30)
        goff = 1.0 / torch.clamp(self.p("ROFF"), min=1e-30)
        ic = ctx.x[cb]
        g, dg = _switch_conductance(ic, gon, goff, self.p("ION"), self.p("IOFF"))
        v = ctx.x[n1] - ctx.x[n2]
        i = g * v
        ctx.add_f(n1, i)
        ctx.add_f(n2, -i)
        if ctx.need_jac:
            didc = dg * v
            rows = torch.cat([n1, n1, n2, n2, n1, n2])
            cols = torch.cat([n1, n2, n1, n2, cb, cb])
            vals = torch.cat([g, -g, -g, g, didc, -didc])
            ctx.add_jf(rows, cols, vals)
