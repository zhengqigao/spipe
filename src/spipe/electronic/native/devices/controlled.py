"""Linear controlled sources: E (VCVS), G (VCCS), F (CCCS), H (CCVS)."""

from __future__ import annotations

import torch

from ..units import SpiceSyntaxError, eval_expr
from .base import DTYPE, Device, DeviceGroup

__all__ = ["VCVS", "VCCS", "CCCS", "CCVS"]


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


def _gain(elem, key):
    if key in elem.kwargs:
        return eval_expr(elem.kwargs[key], elem.params)
    if elem.args:
        a0 = str(elem.args[0]).lower()
        if a0 in ("poly", "table", "value", "laplace", "freq"):
            raise SpiceSyntaxError(
                "%s: POLY/TABLE/VALUE controlled sources are not supported "
                "by the native engine (card: %r)" % (elem.name, elem.source)
            )
        return eval_expr(elem.args[0], elem.params)
    raise SpiceSyntaxError("controlled source %s has no gain: %r" % (elem.name, elem.source))


class VCVS(DeviceGroup):
    """Voltage-controlled voltage source ``E``.

    Card: ``Ename n+ n- nc+ nc- <gain>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``GAIN``     --       voltage gain (V/V)
    ============ ======== ==================================================
    """

    letter = "e"
    n_branch = 1
    PARAMS = {"GAIN": (None, "voltage gain [V/V]")}

    @classmethod
    def build(cls, elem, circuit):
        return Device(elem.name, elem.nodes, {"GAIN": _leaf(_gain(elem, "gain"))})

    def load(self, ctx):
        np_, nm = self.node_idx(0), self.node_idx(1)
        cp, cm = self.node_idx(2), self.node_idx(3)
        br = self.branch_idx()
        k = self.p("GAIN")
        i = ctx.x[br]
        ctx.add_f(np_, i)
        ctx.add_f(nm, -i)
        ctx.add_f(br, ctx.x[np_] - ctx.x[nm] - k * (ctx.x[cp] - ctx.x[cm]))
        if ctx.need_jac:
            one = torch.ones(self.N, dtype=DTYPE)
            rows = torch.cat([np_, nm, br, br, br, br])
            cols = torch.cat([br, br, np_, nm, cp, cm])
            ctx.add_jf(rows, cols, torch.cat([one, -one, one, -one, -k, k]))


class VCCS(DeviceGroup):
    """Voltage-controlled current source ``G``.

    Card: ``Gname n+ n- nc+ nc- <gm>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``GM``       --       transconductance (A/V)
    ============ ======== ==================================================
    """

    letter = "g"
    PARAMS = {"GM": (None, "transconductance [A/V]")}

    @classmethod
    def build(cls, elem, circuit):
        return Device(elem.name, elem.nodes, {"GM": _leaf(_gain(elem, "gm"))})

    def load(self, ctx):
        np_, nm = self.node_idx(0), self.node_idx(1)
        cp, cm = self.node_idx(2), self.node_idx(3)
        gm = self.p("GM")
        i = gm * (ctx.x[cp] - ctx.x[cm])
        ctx.add_f(np_, i)
        ctx.add_f(nm, -i)
        if ctx.need_jac:
            rows = torch.cat([np_, np_, nm, nm])
            cols = torch.cat([cp, cm, cp, cm])
            ctx.add_jf(rows, cols, torch.cat([gm, -gm, -gm, gm]))


class CCCS(DeviceGroup):
    """Current-controlled current source ``F``.

    Card: ``Fname n+ n- <Vcontrol> <gain>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``GAIN``     --       current gain (A/A)
    ============ ======== ==================================================
    """

    letter = "f"
    PARAMS = {"GAIN": (None, "current gain [A/A]")}

    @classmethod
    def build(cls, elem, circuit):
        return Device(elem.name, elem.nodes, {"GAIN": _leaf(_gain(elem, "gain"))},
                      extra={"ctrl": elem.ctrl})

    def ctrl_idx(self):
        t = self._idx_cache.get("ctrl")
        if t is None:
            t = torch.tensor(
                [self.circuit.ctrl_branch_index(d.extra["ctrl"], d.name)
                 for d in self.devices], dtype=torch.long)
            self._idx_cache["ctrl"] = t
        return t

    def load(self, ctx):
        np_, nm = self.node_idx(0), self.node_idx(1)
        cb = self.ctrl_idx()
        k = self.p("GAIN")
        i = k * ctx.x[cb]
        ctx.add_f(np_, i)
        ctx.add_f(nm, -i)
        if ctx.need_jac:
            ctx.add_jf(torch.cat([np_, nm]), torch.cat([cb, cb]), torch.cat([k, -k]))


class CCVS(DeviceGroup):
    """Current-controlled voltage source ``H``.

    Card: ``Hname n+ n- <Vcontrol> <transresistance>``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``R``        --       transresistance (V/A)
    ============ ======== ==================================================
    """

    letter = "h"
    n_branch = 1
    PARAMS = {"R": (None, "transresistance [V/A]")}

    @classmethod
    def build(cls, elem, circuit):
        return Device(elem.name, elem.nodes, {"R": _leaf(_gain(elem, "r"))},
                      extra={"ctrl": elem.ctrl})

    def ctrl_idx(self):
        t = self._idx_cache.get("ctrl")
        if t is None:
            t = torch.tensor(
                [self.circuit.ctrl_branch_index(d.extra["ctrl"], d.name)
                 for d in self.devices], dtype=torch.long)
            self._idx_cache["ctrl"] = t
        return t

    def load(self, ctx):
        np_, nm = self.node_idx(0), self.node_idx(1)
        br = self.branch_idx()
        cb = self.ctrl_idx()
        r = self.p("R")
        i = ctx.x[br]
        ctx.add_f(np_, i)
        ctx.add_f(nm, -i)
        ctx.add_f(br, ctx.x[np_] - ctx.x[nm] - r * ctx.x[cb])
        if ctx.need_jac:
            one = torch.ones(self.N, dtype=DTYPE)
            rows = torch.cat([np_, nm, br, br, br])
            cols = torch.cat([br, br, np_, nm, cb])
            ctx.add_jf(rows, cols, torch.cat([one, -one, one, -one, -r]))
