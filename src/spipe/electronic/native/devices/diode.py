"""Junction diode (Shockley) with series resistance and junction charge."""

from __future__ import annotations

import torch

from ..units import SpiceSyntaxError, eval_expr
from .base import DTYPE, Device, DeviceGroup, limexp, pnjlim

__all__ = ["Diode"]

_ALIASES = {
    "is": "IS", "n": "N", "rs": "RS",
    "cjo": "CJO", "cj0": "CJO", "cj": "CJO",
    "vj": "VJ", "pb": "VJ", "phi": "VJ",
    "m": "M", "mj": "M",
    "tt": "TT", "fc": "FC",
}

_DEFAULTS = {"IS": 1e-14, "N": 1.0, "RS": 0.0, "CJO": 0.0, "VJ": 1.0,
             "M": 0.5, "TT": 0.0, "FC": 0.5, "AREA": 1.0}


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


class Diode(DeviceGroup):
    """Shockley junction diode ``D``.

    Card: ``Dname anode cathode <model> [area] [AREA=..]``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``IS``       1e-14    saturation current [A]
    ``N``        1.0      emission coefficient
    ``RS``       0.0      ohmic series resistance [ohm]
    ``CJO``      0.0      zero-bias junction capacitance [F]
    ``VJ``       1.0      junction potential [V]
    ``M``        0.5      grading coefficient
    ``TT``       0.0      transit time [s] (diffusion charge)
    ``FC``       0.5      forward-bias depletion-capacitance coefficient
    ``AREA``     1.0      area multiplier
    ============ ======== ==================================================

    Current: ``I = AREA*IS*(exp(Vd/(N*Vt)) - 1) + gmin*Vd``.

    Charge (integrated in charge form, hence conserving):
    depletion charge from the standard SPICE junction expression with the
    ``FC`` linearisation above ``FC*VJ``, plus ``TT*I``.

    A non-zero ``RS`` introduces an internal node ``<name>#i``.
    """

    letter = "d"
    has_charge = True
    PARAMS = {k: (v, "") for k, v in _DEFAULTS.items()}

    @classmethod
    def build(cls, elem, circuit):
        mp = circuit.model_params(elem.model, ("d",), elem.name)
        vals = dict(_DEFAULTS)
        for k, v in mp.items():
            key = _ALIASES.get(k)
            if key:
                vals[key] = float(v)
        if elem.args:
            try:
                vals["AREA"] = eval_expr(elem.args[0], elem.params)
            except SpiceSyntaxError:
                pass
        for k, v in elem.kwargs.items():
            key = _ALIASES.get(k, k.upper())
            if key in vals:
                vals[key] = eval_expr(v, elem.params)
        nodes = list(elem.nodes)
        if vals["RS"] > 0.0:
            nodes.append(elem.name + "#i")
        else:
            nodes.append(elem.nodes[0])
        params = {k: _leaf(v) for k, v in vals.items()}
        return Device(elem.name, nodes, params, model=elem.model,
                      extra={"has_rs": vals["RS"] > 0.0})

    # ------------------------------------------------------------------
    def _rs_mask(self):
        t = self._idx_cache.get("rsmask")
        if t is None:
            t = torch.tensor([i for i, d in enumerate(self.devices)
                              if d.extra["has_rs"]], dtype=torch.long)
            self._idx_cache["rsmask"] = t
        return t

    def load(self, ctx):
        na, nc, ni = self.node_idx(0), self.node_idx(1), self.node_idx(2)
        isat = self.p("IS") * self.p("AREA")
        nn = self.p("N")
        vt = self.circuit.vt * nn
        cjo = self.p("CJO") * self.p("AREA")
        vj, mg, tt, fc = self.p("VJ"), self.p("M"), self.p("TT"), self.p("FC")

        vd = ctx.x[ni] - ctx.x[nc]
        if ctx.limiting:
            vcrit = vt * torch.log(vt / (1.4142135623730951 * isat.clamp(min=1e-300)))
            vold = ctx.xold[ni] - ctx.xold[nc]
            vlim = pnjlim(vd, vold, vt, vcrit)
            if bool((vlim != vd).any()):
                ctx.limited = True
            vd = vlim

        ex = limexp(vd / vt)
        gmin = ctx.gmin
        cur = isat * (ex - 1.0) + gmin * vd
        gd = isat * ex / vt + gmin

        ctx.add_f(ni, cur)
        ctx.add_f(nc, -cur)
        if ctx.need_jac:
            rows = torch.cat([ni, ni, nc, nc])
            cols = torch.cat([ni, nc, ni, nc])
            ctx.add_jf(rows, cols, torch.cat([gd, -gd, -gd, gd]))

        # ---- junction charge --------------------------------------------
        if bool((cjo != 0).any()) or bool((tt != 0).any()):
            fcvj = fc * vj
            lo = vd < fcvj
            arg = torch.clamp(1.0 - vd / vj, min=1e-12)
            q_lo = cjo * vj / (1.0 - mg) * (1.0 - arg ** (1.0 - mg))
            c_lo = cjo * arg ** (-mg)
            omf = 1.0 - fc
            f1 = vj / (1.0 - mg) * (1.0 - omf ** (1.0 - mg))
            f2 = omf ** (1.0 + mg)
            f3 = 1.0 - fc * (1.0 + mg)
            dv = vd - fcvj
            q_hi = cjo * (f1 + (f3 * dv + mg / (2.0 * vj) * (vd * vd - fcvj * fcvj)) / f2)
            c_hi = cjo * (f3 + mg * vd / vj) / f2
            qj = torch.where(lo, q_lo, q_hi) + tt * cur
            cj = torch.where(lo, c_lo, c_hi) + tt * gd
            ctx.add_q(ni, qj)
            ctx.add_q(nc, -qj)
            if ctx.need_jac:
                rows = torch.cat([ni, ni, nc, nc])
                cols = torch.cat([ni, nc, ni, nc])
                ctx.add_jq(rows, cols, torch.cat([cj, -cj, -cj, cj]))

        # ---- series resistance -------------------------------------------
        sel = self._rs_mask()
        if sel.numel():
            rs = self.p("RS")[sel] / self.p("AREA")[sel]
            g = 1.0 / rs
            a, i2 = na[sel], ni[sel]
            ir = g * (ctx.x[a] - ctx.x[i2])
            ctx.add_f(a, ir)
            ctx.add_f(i2, -ir)
            if ctx.need_jac:
                rows = torch.cat([a, a, i2, i2])
                cols = torch.cat([a, i2, a, i2])
                ctx.add_jf(rows, cols, torch.cat([g, -g, -g, g]))
