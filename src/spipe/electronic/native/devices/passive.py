"""Linear passive devices: resistor, capacitor, inductor."""

from __future__ import annotations

import torch

from ..units import SpiceSyntaxError, eval_expr
from .base import DTYPE, Device, DeviceGroup

__all__ = ["Resistor", "Capacitor", "Inductor"]

_RMIN = 1e-12


def _value_of(elem, key, default=None):
    """Positional value, ``KEY=value`` or ``.param`` expression."""
    if key in elem.kwargs:
        return eval_expr(elem.kwargs[key], elem.params)
    if elem.args:
        return eval_expr(elem.args[0], elem.params)
    if default is not None:
        return float(default)
    raise SpiceSyntaxError("device %s has no value: %r" % (elem.name, elem.source))


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


class Resistor(DeviceGroup):
    """Linear resistor.

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``R``        --       resistance in ohm (positional or ``R=``)
    ``M``        1.0      parallel instance multiplier
    ============ ======== ==================================================

    Card: ``Rname n+ n- <value> | R=<expr> [M=<mult>]``
    """

    letter = "r"
    PARAMS = {"R": (None, "resistance [ohm]"), "M": (1.0, "parallel multiplier")}

    @classmethod
    def build(cls, elem, circuit):
        r = _value_of(elem, "r")
        m = eval_expr(elem.kwargs.get("m", "1"), elem.params)
        return Device(elem.name, elem.nodes, {"R": _leaf(r), "M": _leaf(m)})

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        r = self.p("R")
        r = torch.where(r.abs() < _RMIN, torch.full_like(r, _RMIN), r)
        g = self.p("M") / r
        v = ctx.x[n1] - ctx.x[n2]
        i = g * v
        ctx.add_f(n1, i)
        ctx.add_f(n2, -i)
        if ctx.need_jac:
            rows = torch.cat([n1, n1, n2, n2])
            cols = torch.cat([n1, n2, n1, n2])
            vals = torch.cat([g, -g, -g, g])
            ctx.add_jf(rows, cols, vals)


class Capacitor(DeviceGroup):
    """Linear capacitor, integrated in *charge* form ``q = C*(v+ - v-)``.

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``C``        --       capacitance in farad (positional or ``C=``)
    ``IC``       0.0      initial voltage used when ``uic=True``
    ``M``        1.0      parallel instance multiplier
    ============ ======== ==================================================
    """

    letter = "c"
    has_charge = True
    PARAMS = {"C": (None, "capacitance [F]"), "IC": (0.0, "initial voltage [V]"),
              "M": (1.0, "parallel multiplier")}

    @classmethod
    def build(cls, elem, circuit):
        c = _value_of(elem, "c")
        ic = eval_expr(elem.kwargs.get("ic", "0"), elem.params)
        m = eval_expr(elem.kwargs.get("m", "1"), elem.params)
        return Device(elem.name, elem.nodes,
                      {"C": _leaf(c), "IC": _leaf(ic), "M": _leaf(m)})

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        c = self.p("C") * self.p("M")
        v = ctx.x[n1] - ctx.x[n2]
        qq = c * v
        ctx.add_q(n1, qq)
        ctx.add_q(n2, -qq)
        if ctx.need_jac:
            rows = torch.cat([n1, n1, n2, n2])
            cols = torch.cat([n1, n2, n1, n2])
            vals = torch.cat([c, -c, -c, c])
            ctx.add_jq(rows, cols, vals)


class Inductor(DeviceGroup):
    """Linear inductor.  Adds one branch-current unknown per instance and is
    integrated in *flux* form ``phi = L*i``.

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``L``        --       inductance in henry (positional or ``L=``)
    ``IC``       0.0      initial current used when ``uic=True``
    ============ ======== ==================================================
    """

    letter = "l"
    has_charge = True
    PARAMS = {"L": (None, "inductance [H]"), "IC": (0.0, "initial current [A]")}
    n_branch = 1

    @classmethod
    def build(cls, elem, circuit):
        l = _value_of(elem, "l")
        ic = eval_expr(elem.kwargs.get("ic", "0"), elem.params)
        return Device(elem.name, elem.nodes, {"L": _leaf(l), "IC": _leaf(ic)})

    def load(self, ctx):
        n1, n2 = self.node_idx(0), self.node_idx(1)
        br = self.branch_idx()
        i = ctx.x[br]
        l = self.p("L")
        ctx.add_f(n1, i)
        ctx.add_f(n2, -i)
        # branch equation:  (v+ - v-) - L di/dt = 0
        ctx.add_f(br, ctx.x[n1] - ctx.x[n2])
        ctx.add_q(br, -l * i)
        if ctx.need_jac:
            one = torch.ones_like(l)
            rows = torch.cat([n1, n2, br, br])
            cols = torch.cat([br, br, n1, n2])
            vals = torch.cat([one, -one, one, -one])
            ctx.add_jf(rows, cols, vals)
            ctx.add_jq(br, br, -l)
