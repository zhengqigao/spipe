"""MOSFET model.

``LEVEL=1`` (Shichman-Hodges) is the only level implemented: square law with
body effect, channel-length modulation, the drain/source role swap for
``Vds < 0``, linear overlap / junction capacitances and charge-conserving
Meyer intrinsic capacitances.  EKV is **not** implemented; a ``LEVEL`` other
than 1 is rejected rather than silently mis-simulated.
"""

from __future__ import annotations

import torch

from ..units import SpiceSyntaxError, eval_expr
from .base import DTYPE, Device, DeviceGroup

__all__ = ["Mosfet"]

EPS_OX = 3.9 * 8.854214871e-12  # F/m

_MODEL_DEFAULTS = {
    "LEVEL": 1.0,
    "VTO": 0.0, "KP": 2.0e-5, "LAMBDA": 0.0, "GAMMA": 0.0, "PHI": 0.6,
    "TOX": 0.0, "COX": 0.0, "UO": 0.0, "CAPOP": 3.0,
    "CGSO": 0.0, "CGDO": 0.0, "CGBO": 0.0,
    "CBD": 0.0, "CBS": 0.0,
}

_INST_DEFAULTS = {"W": 1.0e-4, "L": 1.0e-4, "M": 1.0, "AD": 0.0, "AS": 0.0}

_MODEL_ALIASES = {
    "level": "LEVEL", "vto": "VTO", "vt0": "VTO", "kp": "KP",
    "lambda": "LAMBDA", "la": "LAMBDA", "gamma": "GAMMA", "phi": "PHI",
    "tox": "TOX", "cox": "COX", "uo": "UO", "u0": "UO", "capop": "CAPOP",
    "cgso": "CGSO", "cgdo": "CGDO", "cgbo": "CGBO",
    "cbd": "CBD", "cbs": "CBS",
}


def _leaf(v):
    return torch.tensor(float(v), dtype=DTYPE)


class Mosfet(DeviceGroup):
    """MOSFET ``M`` (level 1 Shichman-Hodges by default).

    Card: ``Mname drain gate source bulk <model> W=.. L=.. [M=..] [AD=..] [AS=..]``

    Model parameters (``.model nch NMOS (...)`` / ``PMOS``)

    ============ ======== ==================================================
    parameter    default  description
    ============ ======== ==================================================
    ``LEVEL``    1        only level 1 (Shichman-Hodges) is implemented
    ``VTO``      0.0      zero-bias threshold voltage [V] (signed: negative
                          for a normal PMOS)
    ``KP``       2e-5     transconductance parameter [A/V^2]
    ``LAMBDA``   0.0      channel-length modulation [1/V]
    ``GAMMA``    0.0      body-effect factor [V^1/2]
    ``PHI``      0.6      surface potential [V]
    ``TOX``      0        oxide thickness [m]; sets ``COX`` if ``COX`` unset
    ``COX``      0        oxide capacitance per area [F/m^2]; 0 disables the
                          intrinsic (Meyer) capacitances
    ``CAPOP``    3        intrinsic capacitance model: 0 = none, 2 = Meyer
                          with the Berkeley sub-threshold ramp, 3 = Meyer
                          with no gate-bulk term (what Xyce 7.10 does)
    ``UO``       0        surface mobility [cm^2/Vs]; with ``TOX`` it sets KP
    ``CGSO``     0.0      gate-source overlap capacitance per metre of W
    ``CGDO``     0.0      gate-drain overlap capacitance per metre of W
    ``CGBO``     0.0      gate-bulk overlap capacitance per metre of L
    ``CBD``      0.0      bulk-drain capacitance [F]
    ``CBS``      0.0      bulk-source capacitance [F]
    ============ ======== ==================================================

    Instance parameters: ``W`` (1e-4 m), ``L`` (1e-4 m), ``M`` (multiplier),
    ``AD``/``AS`` (accepted, unused).

    Both polarities and all three regions (cutoff / triode / saturation) are
    implemented, with the drain/source role swap for ``Vds < 0`` so that the
    model is symmetric.  Not implemented: bulk junction diodes, ``RD``/``RS``
    series resistance, short-channel effects, subthreshold conduction
    (level 1 has none by definition).
    """

    letter = "m"
    has_charge = True
    PARAMS = dict(
        {k: (v, "") for k, v in _MODEL_DEFAULTS.items()},
        **{k: (v, "") for k, v in _INST_DEFAULTS.items()},
    )

    @classmethod
    def build(cls, elem, circuit):
        mp = circuit.model_params(elem.model, ("nmos", "pmos", "nch", "pch"), elem.name)
        mtype = circuit.model_type(elem.model)
        vals = dict(_MODEL_DEFAULTS)
        vals.update(_INST_DEFAULTS)
        for k, v in mp.items():
            key = _MODEL_ALIASES.get(k)
            if key:
                try:
                    vals[key] = float(v)
                except (TypeError, ValueError):
                    pass
        for k, v in elem.kwargs.items():
            key = k.upper()
            if key in _INST_DEFAULTS:
                vals[key] = eval_expr(v, elem.params)
            elif _MODEL_ALIASES.get(k):
                vals[_MODEL_ALIASES[k]] = eval_expr(v, elem.params)
        # positional  W L  (rare, but legal in some dialects)
        if elem.args and "W" not in elem.kwargs:
            try:
                vals["W"] = eval_expr(elem.args[0], elem.params)
                if len(elem.args) > 1:
                    vals["L"] = eval_expr(elem.args[1], elem.params)
            except SpiceSyntaxError:
                pass

        if vals["COX"] == 0.0 and vals["TOX"] > 0.0:
            vals["COX"] = EPS_OX / vals["TOX"]
        if "kp" not in mp and vals["UO"] > 0.0 and vals["COX"] > 0.0:
            vals["KP"] = vals["UO"] * 1e-4 * vals["COX"]

        if int(vals["LEVEL"]) != 1:
            raise SpiceSyntaxError(
                "device %s uses .model %s with LEVEL=%g; the native engine "
                "implements MOSFET LEVEL=1 (Shichman-Hodges) only"
                % (elem.name, elem.model, vals["LEVEL"]))
        ptype = -1.0 if mtype.startswith("p") else 1.0
        params = {k: _leaf(v) for k, v in vals.items()}
        return Device(elem.name, elem.nodes, params, model=elem.model,
                      extra={"type": ptype, "level": int(vals["LEVEL"])})

    # ------------------------------------------------------------------
    def _type_vec(self):
        t = self._idx_cache.get("type")
        if t is None:
            t = torch.tensor([d.extra["type"] for d in self.devices], dtype=DTYPE)
            self._idx_cache["type"] = t
        return t

    def load(self, ctx):
        nd, ng, ns, nb = (self.node_idx(0), self.node_idx(1),
                          self.node_idx(2), self.node_idx(3))
        ty = self._type_vec()
        w, l, mult = self.p("W"), self.p("L"), self.p("M")
        kp, vto, lam = self.p("KP"), self.p("VTO"), self.p("LAMBDA")
        # SPICE sign convention: VTO is given signed (negative for a normal
        # enhancement PMOS); the type-normalised model uses ty*VTO.
        vto = ty * vto
        gamma, phi = self.p("GAMMA"), self.p("PHI")
        cox = self.p("COX")

        # ---- terminal voltages (source referenced, type-normalised) ------
        vgs_r = ty * (ctx.x[ng] - ctx.x[ns])
        vds_r = ty * (ctx.x[nd] - ctx.x[ns])
        vbs_r = ty * (ctx.x[nb] - ctx.x[ns])
        if ctx.limiting:
            og = ty * (ctx.xold[ng] - ctx.xold[ns])
            od = ty * (ctx.xold[nd] - ctx.xold[ns])
            ob = ty * (ctx.xold[nb] - ctx.xold[ns])
            lg = og + torch.clamp(vgs_r - og, -2.0, 2.0)
            ld = od + torch.clamp(vds_r - od, -2.0, 2.0)
            lb = ob + torch.clamp(vbs_r - ob, -2.0, 2.0)
            if (bool((lg != vgs_r).any()) or bool((ld != vds_r).any())
                    or bool((lb != vbs_r).any())):
                ctx.limited = True
            vgs_r, vds_r, vbs_r = lg, ld, lb

        fwd = vds_r >= 0.0
        # internal drain / source node indices
        dd = torch.where(fwd, nd, ns)
        ss = torch.where(fwd, ns, nd)
        vds = torch.where(fwd, vds_r, -vds_r)
        vgs = torch.where(fwd, vgs_r, vgs_r - vds_r)
        vbs = torch.where(fwd, vbs_r, vbs_r - vds_r)

        # ---- threshold voltage -------------------------------------------
        phic = torch.clamp(phi, min=1e-6)
        sq_phi = torch.sqrt(phic)
        neg = vbs <= 0.0
        arg = torch.clamp(phic - vbs, min=1e-12)
        sarg_n = torch.sqrt(arg)
        dsarg_n = -0.5 / sarg_n
        den = 1.0 + vbs / (2.0 * phic)
        den = torch.clamp(den, min=1e-6)
        sarg_p = sq_phi / den
        dsarg_p = -sq_phi / (2.0 * phic * den * den)
        sarg = torch.where(neg, sarg_n, sarg_p)
        dsarg = torch.where(neg, dsarg_n, dsarg_p)
        vth = vto + gamma * (sarg - sq_phi)
        dadvb = -gamma * dsarg          # d(vgs - vth)/d vbs

        beta = kp * (w / torch.clamp(l, min=1e-30)) * mult
        vgst = vgs - vth

        cut = vgst <= 0.0
        tri = (~cut) & (vds < vgst)

        lamf = 1.0 + lam * vds
        id_tri = beta * (vgst - 0.5 * vds) * vds * lamf
        id_sat = 0.5 * beta * vgst * vgst * lamf
        gm_tri = beta * vds * lamf
        gm_sat = beta * vgst * lamf
        gds_tri = beta * ((vgst - vds) * lamf + lam * (vgst - 0.5 * vds) * vds)
        gds_sat = 0.5 * beta * vgst * vgst * lam

        zero = torch.zeros_like(vgst)
        ids = torch.where(cut, zero, torch.where(tri, id_tri, id_sat))
        gm = torch.where(cut, zero, torch.where(tri, gm_tri, gm_sat))
        gds = torch.where(cut, zero, torch.where(tri, gds_tri, gds_sat))
        gmbs = gm * dadvb

        cur = ty * ids
        ctx.add_f(dd, cur)
        ctx.add_f(ss, -cur)
        if ctx.need_jac:
            gsum = gm + gds + gmbs
            rows = torch.cat([dd, dd, dd, dd, ss, ss, ss, ss])
            cols = torch.cat([dd, ng, ss, nb, dd, ng, ss, nb])
            vals = torch.cat([gds, gm, -gsum, gmbs,
                              -gds, -gm, gsum, -gmbs])
            ctx.add_jf(rows, cols, vals)

        # ---- linear overlap / junction capacitances -----------------------
        cgso = self.p("CGSO") * w * mult
        cgdo = self.p("CGDO") * w * mult
        cgbo = self.p("CGBO") * l * mult
        cbd = self.p("CBD") * mult
        cbs = self.p("CBS") * mult
        for cval, na_, nb_ in ((cgso, ng, ns), (cgdo, ng, nd), (cgbo, ng, nb),
                               (cbd, nb, nd), (cbs, nb, ns)):
            if not bool((cval != 0).any()):
                continue
            qq = cval * (ctx.x[na_] - ctx.x[nb_])
            ctx.add_q(na_, qq)
            ctx.add_q(nb_, -qq)
            if ctx.need_jac:
                rows = torch.cat([na_, na_, nb_, nb_])
                cols = torch.cat([na_, nb_, na_, nb_])
                ctx.add_jq(rows, cols, torch.cat([cval, -cval, -cval, cval]))

        # ---- intrinsic (Meyer) charges ------------------------------------
        capop = self.p("CAPOP")
        if bool((cox != 0).any()) and bool((capop != 0).any()):
            self._load_intrinsic(ctx, cox * w * l * mult * (capop != 0),
                                 ty, vgst, vds, dadvb, phic, capop,
                                 ng, dd, ss, nb)

    # ------------------------------------------------------------------
    def _load_intrinsic(self, ctx, cch, ty, a, vds, dadvb, phi, capop,
                        ng, dd, ss, nb):
        """Charge-conserving Meyer intrinsic capacitances.

        In strong inversion the gate charge is the analytic antiderivative of
        the Meyer capacitances, ``Qg = (2/3)C (a + b - a b/(a+b))`` with
        ``a = vgs - vth`` and ``b = a - vds``; it is split symmetrically
        between the source and drain ends, so ``Qgs = Qgd`` at ``vds = 0``
        and ``Qgd = 0`` in saturation, exactly as Meyer's ``Cgd`` does.

        Below threshold the SPICE Meyer depletion/accumulation ramp is used:

        ============================ ======================================
        region                       capacitances
        ============================ ======================================
        ``vgst <= -PHI``             ``Cgb = Cox``
        ``-PHI < vgst <= -PHI/2``    ``Cgb = -vgst*Cox/PHI``
        ``-PHI/2 < vgst <= 0``       ``Cgb = -vgst*Cox/PHI``,
                                     ``Cgs = (4/3)vgst*Cox/PHI + (2/3)Cox``
        ============================ ======================================

        which is integrated analytically here so that the model stays charge
        based.  The total gate capacitance is continuous across all four
        regions.
        """
        b = a - vds
        on = a > 0.0
        tri = on & (b > 0.0)
        d = torch.clamp(a + b, min=1e-30)
        c23 = (2.0 / 3.0) * cch
        ns_ = a * a + 0.5 * a * b
        nd_ = b * b + 0.5 * a * b
        qgs_t = c23 * ns_ / d
        qgd_t = c23 * nd_ / d
        # derivatives w.r.t. a and b
        dqgs_da = c23 * ((2.0 * a + 0.5 * b) * d - ns_) / (d * d)
        dqgs_db = c23 * (0.5 * a * d - ns_) / (d * d)
        dqgd_db = c23 * ((2.0 * b + 0.5 * a) * d - nd_) / (d * d)
        dqgd_da = c23 * (0.5 * b * d - nd_) / (d * d)

        zero = torch.zeros_like(a)
        # ---- sub-threshold: depletion then accumulation ------------------
        ph = torch.clamp(phi, min=1e-6)
        dep = a > -0.5 * ph                     # -PHI/2 < vgst <= 0
        mid = (~dep) & (a > -ph)                # -PHI   < vgst <= -PHI/2
        qgs_dep = cch * ((2.0 / 3.0) * a * a / ph + (2.0 / 3.0) * a)
        qgb_dep = -cch * 0.5 * a * a / ph
        qgs_mid = -cch * ph / 6.0
        qgb_mid = -cch * ph / 8.0 - cch / (2.0 * ph) * (a * a - 0.25 * ph * ph)
        qgb_acc = -cch * 0.5 * ph + cch * (a + ph)
        qgs_off = torch.where(dep, qgs_dep, qgs_mid)
        qgb_off = torch.where(dep, qgb_dep, torch.where(mid, qgb_mid, qgb_acc))
        dqgs_off = torch.where(
            dep, cch * ((4.0 / 3.0) * a / ph + 2.0 / 3.0), zero)
        dqgb_off = torch.where(dep, -cch * a / ph,
                               torch.where(mid, -cch * a / ph, cch))

        nogb = capop == 3.0           # Xyce-like: no intrinsic gate-bulk term
        qgb_off = torch.where(nogb, zero, qgb_off)
        dqgb_off = torch.where(nogb, zero, dqgb_off)

        qgs = torch.where(on, torch.where(tri, qgs_t, c23 * a), qgs_off)
        qgd = torch.where(on, torch.where(tri, qgd_t, zero), zero)
        qgb = torch.where(on, zero, qgb_off)

        A1 = torch.where(on, torch.where(tri, dqgs_da, c23), dqgs_off)
        B1 = torch.where(on, torch.where(tri, dqgs_db, zero), zero)
        A2 = torch.where(on, torch.where(tri, dqgd_da, zero), zero)
        B2 = torch.where(on, torch.where(tri, dqgd_db, zero), zero)
        A3 = torch.where(on, zero, dqgb_off)

        ctx.add_q(ng, ty * (qgs + qgd + qgb))
        ctx.add_q(ss, -ty * qgs)
        ctx.add_q(dd, -ty * qgd)
        ctx.add_q(nb, -ty * qgb)

        if not ctx.need_jac:
            return
        # d/d(vgs)=A+B, d/d(vds)=-B, d/d(vbs)=(A+B)*dadvb, d/d(vs)=-(sum)
        for (A, B, other) in ((A1, B1, ss), (A2, B2, dd), (A3, zero, nb)):
            dg = A + B
            ddr = -B
            db = dg * dadvb
            dsr = -(dg + ddr + db)
            rows = torch.cat([ng, ng, ng, ng, other, other, other, other])
            cols = torch.cat([ng, dd, ss, nb, ng, dd, ss, nb])
            vals = torch.cat([dg, ddr, dsr, db, -dg, -ddr, -dsr, -db])
            ctx.add_jq(rows, cols, vals)
