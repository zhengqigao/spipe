"""Bipolar junction transistor -- reduced Gummel-Poon model (Tier 3)."""

from __future__ import annotations

import warnings

import torch

from ..units import SpiceSyntaxError, eval_expr
from .base import DTYPE, Device, DeviceGroup, limexp, pnjlim

__all__ = ["BJT"]

#: (model, ignored-parameter set) pairs already warned about, so a model used by many
#: instances warns once rather than once per device.
_WARNED_IGNORED = set()

_DEFAULTS = {
    "IS": 1e-16, "BF": 100.0, "BR": 1.0, "NF": 1.0, "NR": 1.0,
    "VAF": 0.0, "VAR": 0.0,
    "CJE": 0.0, "VJE": 0.75, "MJE": 0.33,
    "CJC": 0.0, "VJC": 0.75, "MJC": 0.33,
    "TF": 0.0, "TR": 0.0, "AREA": 1.0,
}

_ALIASES = {
    "is": "IS", "bf": "BF", "br": "BR", "nf": "NF", "nr": "NR",
    "vaf": "VAF", "va": "VAF", "var": "VAR", "vb": "VAR",
    "cje": "CJE", "vje": "VJE", "pe": "VJE", "mje": "MJE", "me": "MJE",
    "cjc": "CJC", "vjc": "VJC", "pc": "VJC", "mjc": "MJC", "mc": "MJC",
    "tf": "TF", "tr": "TR",
}


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


def _depl_charge(cj, vj, m, v):
    """Depletion charge and capacitance with the usual FC=0.5 linearisation."""
    fc = 0.5
    fcvj = fc * vj
    lo = v < fcvj
    arg = torch.clamp(1.0 - v / vj, min=1e-12)
    q_lo = cj * vj / (1.0 - m) * (1.0 - arg ** (1.0 - m))
    c_lo = cj * arg ** (-m)
    omf = 1.0 - fc
    f1 = vj / (1.0 - m) * (1.0 - omf ** (1.0 - m))
    f2 = omf ** (1.0 + m)
    f3 = 1.0 - fc * (1.0 + m)
    dv = v - fcvj
    q_hi = cj * (f1 + (f3 * dv + m / (2.0 * vj) * (v * v - fcvj * fcvj)) / f2)
    c_hi = cj * (f3 + m * v / vj) / f2
    return torch.where(lo, q_lo, q_hi), torch.where(lo, c_lo, c_hi)


class BJT(DeviceGroup):
    """Bipolar transistor ``Q`` (reduced Gummel-Poon).

    Card: ``Qname collector base emitter [substrate] <model> [area]``

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``IS``       1e-16    transport saturation current [A]
    ``BF``       100      ideal forward beta
    ``BR``       1        ideal reverse beta
    ``NF``       1.0      forward emission coefficient
    ``NR``       1.0      reverse emission coefficient
    ``VAF``      0 (inf)  forward Early voltage [V]
    ``VAR``      0 (inf)  reverse Early voltage [V]
    ``CJE/VJE/MJE``  0/0.75/0.33  base-emitter depletion capacitance
    ``CJC/VJC/MJC``  0/0.75/0.33  base-collector depletion capacitance
    ``TF``/``TR``    0/0      forward / reverse transit time [s]
    ``AREA``     1.0      area multiplier
    ============ ======== ==================================================

    Not modelled: high-level injection (``IKF``/``IKR``), series resistances
    ``RB``/``RC``/``RE``, substrate junction, temperature dependence.
    """

    letter = "q"
    has_charge = True
    PARAMS = {k: (v, "") for k, v in _DEFAULTS.items()}

    @classmethod
    def build(cls, elem, circuit):
        mp = circuit.model_params(elem.model, ("npn", "pnp"), elem.name)
        mtype = circuit.model_type(elem.model)

        # A BJT card's LEVEL selects the compact model, and this engine has exactly one:
        # Gummel-Poon, which is LEVEL=1 (or no LEVEL) in every SPICE dialect. LEVEL=9 and 12
        # are VBIC, which foundry PDKs such as IHP SG13G2 use. Accepting those and quietly
        # solving Gummel-Poon instead is how a VBIC HBT that carries 593 uA came out at
        # 187 uA -- 3.2x wrong, no warning. The MOSFET builder has always refused an
        # unsupported LEVEL; the BJT now does the same.
        if "level" in mp:
            try:
                level = float(mp["level"])
            except (TypeError, ValueError):
                level = None
            if level != 1.0:
                raise SpiceSyntaxError(
                    "device %s uses .model %s with LEVEL=%s; the native engine implements "
                    "the Gummel-Poon BJT (LEVEL=1) only. LEVEL=9/12 is VBIC -- simulate that "
                    "with spice_exe='xyce' or 'hspice'."
                    % (elem.name, elem.model, mp["level"]))

        vals = dict(_DEFAULTS)
        ignored = []
        for k, v in mp.items():
            key = _ALIASES.get(k)
            if key:
                try:
                    vals[key] = float(v)
                except (TypeError, ValueError):
                    pass
            elif k != "level":
                ignored.append(k.upper())

        # Parameters this model does not implement (IKF, RB, RC, RE, the VBIC set, ...) are
        # still ignored -- a real foundry card carries dozens that are harmless to drop, and
        # refusing them would make ordinary decks unusable. But they are no longer ignored
        # *silently*: RB=100 on a real deck changes the answer, and the user should know.
        if ignored:
            key = (elem.model, tuple(sorted(ignored)))
            if key not in _WARNED_IGNORED:
                _WARNED_IGNORED.add(key)
                warnings.warn(
                    "BJT .model %s: the native engine's Gummel-Poon model does not implement "
                    "%s, so %s ignored. The result may differ from a simulator that does "
                    "(for series resistances and high-level injection, noticeably)."
                    % (elem.model, ", ".join(sorted(ignored)),
                       "it is" if len(ignored) == 1 else "they are"),
                    RuntimeWarning, stacklevel=2)
        if elem.args:
            try:
                vals["AREA"] = eval_expr(elem.args[0], elem.params)
            except SpiceSyntaxError:
                pass
        for k, v in elem.kwargs.items():
            key = _ALIASES.get(k, k.upper())
            if key in vals:
                vals[key] = eval_expr(v, elem.params)
        nodes = list(elem.nodes[:3])
        ptype = -1.0 if mtype.startswith("pnp") else 1.0
        return Device(elem.name, nodes, {k: _leaf(v) for k, v in vals.items()},
                      model=elem.model, extra={"type": ptype})

    def _type_vec(self):
        t = self._idx_cache.get("type")
        if t is None:
            t = torch.tensor([d.extra["type"] for d in self.devices], dtype=DTYPE)
            self._idx_cache["type"] = t
        return t

    def load(self, ctx):
        nc, nb, ne = self.node_idx(0), self.node_idx(1), self.node_idx(2)
        ty = self._type_vec()
        area = self.p("AREA")
        isat = self.p("IS") * area
        bf, br = self.p("BF"), self.p("BR")
        nf, nr = self.p("NF"), self.p("NR")
        vaf, var = self.p("VAF"), self.p("VAR")
        vtf = self.circuit.vt * nf
        vtr = self.circuit.vt * nr

        vbe = ty * (ctx.x[nb] - ctx.x[ne])
        vbc = ty * (ctx.x[nb] - ctx.x[nc])
        if ctx.limiting:
            vcrit_f = vtf * torch.log(vtf / (1.4142135623730951 * (isat / bf).clamp(min=1e-300)))
            vcrit_r = vtr * torch.log(vtr / (1.4142135623730951 * (isat / br).clamp(min=1e-300)))
            obe = ty * (ctx.xold[nb] - ctx.xold[ne])
            obc = ty * (ctx.xold[nb] - ctx.xold[nc])
            lbe = pnjlim(vbe, obe, vtf, vcrit_f)
            lbc = pnjlim(vbc, obc, vtr, vcrit_r)
            if bool((lbe != vbe).any()) or bool((lbc != vbc).any()):
                ctx.limited = True
            vbe, vbc = lbe, lbc

        exf = limexp(vbe / vtf)
        exr = limexp(vbc / vtr)
        ifwd = isat * (exf - 1.0)
        irev = isat * (exr - 1.0)
        gif = isat * exf / vtf
        gir = isat * exr / vtr

        gmin = ctx.gmin
        ibe = ifwd / bf + gmin * vbe
        ibc = irev / br + gmin * vbc
        gbe = gif / bf + gmin
        gbc = gir / br + gmin

        # base charge (Early effect only)
        zer = torch.zeros_like(vbe)
        has_f = vaf > 0
        has_r = var > 0
        vaf_s = torch.where(has_f, vaf, torch.ones_like(vaf))
        var_s = torch.where(has_r, var, torch.ones_like(var))
        inv = (torch.where(has_f, -vbc / vaf_s, zer)
               + torch.where(has_r, -vbe / var_s, zer))
        u = torch.clamp(1.0 + inv, min=1e-4)
        qb = 1.0 / u
        dqb_dvbc = torch.where(has_f, qb * qb / vaf_s, torch.zeros_like(qb))
        dqb_dvbe = torch.where(has_r, qb * qb / var_s, torch.zeros_like(qb))

        diff = ifwd - irev
        ict = diff * qb
        gmf = gif * qb + diff * dqb_dvbe
        gmr = -gir * qb + diff * dqb_dvbc

        i_c = ty * (ict - ibc)
        i_b = ty * (ibe + ibc)
        i_e = -(i_c + i_b)
        ctx.add_f(nc, i_c)
        ctx.add_f(nb, i_b)
        ctx.add_f(ne, i_e)

        if ctx.need_jac:
            # d(I)/d(vbe), d(I)/d(vbc) for the three terminal currents
            rows, cols, vals = [], [], []
            for row, dbe, dbc in (
                (nc, gmf, gmr - gbc),
                (nb, gbe, gbc),
                (ne, -(gmf + gbe), -gmr),
            ):
                rows += [row, row, row]
                cols += [nb, ne, nc]
                vals += [dbe + dbc, -dbe, -dbc]
            ctx.add_jf(torch.cat(rows), torch.cat(cols), torch.cat(vals))

        # ---- charges -----------------------------------------------------
        cje, cjc = self.p("CJE") * area, self.p("CJC") * area
        tf, tr = self.p("TF"), self.p("TR")
        if bool((cje != 0).any()) or bool((cjc != 0).any()) or \
           bool((tf != 0).any()) or bool((tr != 0).any()):
            qe, ce = _depl_charge(cje, self.p("VJE"), self.p("MJE"), vbe)
            qc, cc = _depl_charge(cjc, self.p("VJC"), self.p("MJC"), vbc)
            qbe = qe + tf * ifwd
            qbc = qc + tr * irev
            cbe = ce + tf * gif
            cbc = cc + tr * gir
            ctx.add_q(nb, ty * (qbe + qbc))
            ctx.add_q(ne, -ty * qbe)
            ctx.add_q(nc, -ty * qbc)
            if ctx.need_jac:
                rows = torch.cat([nb, nb, nb, ne, ne, nc, nc])
                cols = torch.cat([nb, ne, nc, nb, ne, nb, nc])
                vals = torch.cat([cbe + cbc, -cbe, -cbc,
                                  -cbe, cbe, -cbc, cbc])
                ctx.add_jq(rows, cols, vals)
