"""Newton-Raphson solver with damping, gmin stepping and source stepping."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .mna import CircuitError

DTYPE = torch.float64

__all__ = ["NewtonOptions", "ConvergenceError", "newton_solve", "dc_solve"]


class ConvergenceError(RuntimeError):
    """Newton failed to converge; the message names the worst unknown."""


class NewtonOptions:
    """Numerical tolerances.

    ============= ========= =============================================
    option        default   meaning
    ============= ========= =============================================
    ``reltol``    1e-11     relative tolerance on unknowns and residual
    ``abstol``    1e-15     absolute current tolerance [A]
    ``vntol``     1e-11     absolute voltage tolerance [V]
    ``maxiter``   100       Newton iterations per solve
    ``dxmax``     1e2       maximum voltage change per iteration [V]
    ``gmin``      0.0       conductance added across nonlinear junctions
    ``lte_reltol`` 1e-8     transient local-truncation-error tolerance
    ``lte_abstol`` 1e-14    absolute LTE floor
    ``lte_reltol_nl`` 1e-5  LTE tolerance for circuits containing nonlinear
                            devices, where extra steps are ~10x dearer
    ============= ========= =============================================
    """

    __slots__ = ("reltol", "abstol", "vntol", "maxiter", "dxmax", "gmin",
                 "lte_reltol", "lte_abstol", "lte_reltol_nl")

    def __init__(self, reltol=1e-11, abstol=1e-15, vntol=1e-11, maxiter=100,
                 dxmax=1e2, gmin=0.0, lte_reltol=1e-8, lte_abstol=1e-14,
                 lte_reltol_nl=1e-5):
        self.reltol = reltol
        self.abstol = abstol
        self.vntol = vntol
        self.maxiter = maxiter
        self.dxmax = dxmax
        self.gmin = gmin
        self.lte_reltol = lte_reltol
        self.lte_abstol = lte_abstol
        self.lte_reltol_nl = lte_reltol_nl

    def copy(self, **kw):
        """A copy of these options with the given fields overridden."""
        o = NewtonOptions(self.reltol, self.abstol, self.vntol, self.maxiter,
                          self.dxmax, self.gmin, self.lte_reltol,
                          self.lte_abstol, self.lte_reltol_nl)
        for k, v in kw.items():
            setattr(o, k, v)
        return o


def _lin_solve(J, rhs, sys, context):
    try:
        out = torch.linalg.solve(J, rhs)
        if torch.isfinite(out).all():
            return out
    except Exception:
        pass
    _raise_singular(J, sys, context)


def _raise_singular(J, sys, context):
    """Diagnose a singular MNA matrix and name the unknowns responsible."""
    bad = sys.zero_rows(J)
    if bad:
        raise CircuitError(
            "singular MNA matrix during %s: %s %s no connection to the rest "
            "of the circuit (check for an unconnected node or a missing DC "
            "path to ground)"
            % (context, ", ".join(bad), "has" if len(bad) == 1 else "have"))
    try:
        _, S, Vh = torch.linalg.svd(J)
        v = Vh[-1].abs()
        order = torch.argsort(v, descending=True)
        names = [sys.unknown_name(int(i)) for i in order[:4]
                 if float(v[i]) > 0.05]
    except Exception:
        names = []
    raise CircuitError(
        "singular MNA matrix during %s: %s not independently determined -- "
        "the circuit has no DC path to ground for them, a loop of voltage "
        "sources, or a capacitor-only cutset"
        % (context, ", ".join(names) if names else "some unknowns are"))


def newton_solve(assemble: Callable, x0: torch.Tensor, opts: NewtonOptions,
                 sys, context="DC operating point", x_scale=None):
    """Solve ``G(x) = 0``.

    ``assemble(x, xold) -> (G, J)``.  ``xold`` is the previous iterate, handed
    to the devices so that they can apply SPICE voltage limiting; limiting is
    inactive once converged, so the returned solution satisfies the unmodified
    device equations.

    Returns ``(x, iterations)``.
    """
    x = x0.clone()
    n = x.numel()
    nnode = sys.n_nodes
    atol = torch.full((n,), opts.abstol, dtype=DTYPE)
    atol[:nnode] = opts.vntol
    # residual units: amperes on node rows, volts on branch rows
    rabs = torch.full((n,), opts.abstol, dtype=DTYPE)
    rabs[nnode:nnode + sys.n_branch] = opts.vntol
    last = None
    prev_dx = float("inf")
    stall = 0
    for it in range(opts.maxiter):
        G, J, scale, limited = assemble(x, x if last is None else last)
        if not torch.isfinite(G).all() or not torch.isfinite(J).all():
            raise ConvergenceError(
                "non-finite residual during %s at iteration %d" % (context, it)
            )
        # residual test first: at convergence the limiters are inactive, so
        # this is the residual of the unmodified device equations
        # A limiter that actually fired means this residual belongs to the
        # *modified* device equations, so convergence may not be declared yet
        # (Berkeley SPICE does exactly this).
        small_res = (it > 0 and not limited
                     and bool((G.abs() <= rabs + opts.reltol * scale).all()))
        dx = _lin_solve(J, -G, sys, context)
        if not torch.isfinite(dx).all():
            raise ConvergenceError("non-finite Newton step during %s" % context)
        ok_dx = bool((dx.abs() <= atol + opts.reltol * (x + dx).abs()).all())
        # An ill-conditioned step (a very small h, say) can leave the Newton
        # update at the round-off floor without ever meeting `ok_dx`.  If the
        # update has stopped shrinking while the residual is already inside
        # tolerance, the solution is as good as float64 allows.
        dxn = float(dx.abs().max())
        stall = stall + 1 if dxn > 0.5 * prev_dx else 0
        prev_dx = dxn
        if small_res and (ok_dx or stall >= 3):
            return x + dx, it + 1
        mx = float(dx.abs().max())
        if mx > opts.dxmax:
            dx = dx * (opts.dxmax / mx)
        last = x
        x = x + dx
    # report the offending unknown
    G, J, scale, _ = assemble(x, x)
    rtol = rabs + opts.reltol * scale
    err = (G.abs() / torch.clamp(rtol, min=1e-300))
    k = int(err.argmax())
    raise ConvergenceError(
        "Newton did not converge during %s after %d iterations: worst unknown "
        "%s, residual %.3e (tolerance %.3e)"
        % (context, opts.maxiter, sys.unknown_name(k), float(G[k]), float(rtol[k]))
    )


def dc_solve(make_assemble: Callable, x0: torch.Tensor, opts: NewtonOptions,
             sys, context="DC operating point"):
    """DC solve with automatic fallbacks.

    ``make_assemble(gmin, gshunt, src_scale) -> assemble`` builds the residual
    closure for a given continuation setting.  Strategy:

    1. plain Newton;
    2. **gmin stepping** -- a shunt conductance to ground is ramped
       ``1e-3 -> 1e-12 -> 0``, each solve warm-starting the next;
    3. **source stepping** -- every independent source is ramped from 0 to
       its full value.
    """
    first = None
    try:
        return newton_solve(make_assemble(opts.gmin, 0.0, 1.0), x0, opts, sys, context)
    except (ConvergenceError, CircuitError) as exc:
        first = exc

    # ---- gmin stepping ---------------------------------------------------
    x = x0.clone()
    try:
        g = 1e-3
        while g > 1e-13:
            x, _ = newton_solve(make_assemble(opts.gmin, g, 1.0), x, opts, sys,
                                context + " (gmin=%g)" % g)
            g *= 0.1
        return newton_solve(make_assemble(opts.gmin, 0.0, 1.0), x, opts, sys, context)
    except (ConvergenceError, CircuitError):
        pass

    # ---- source stepping -------------------------------------------------
    x = torch.zeros_like(x0)
    s = 0.0
    ds = 0.1
    guard = 0
    while s < 1.0 and guard < 200:
        guard += 1
        trial = min(1.0, s + ds)
        try:
            xn, _ = newton_solve(make_assemble(max(opts.gmin, 1e-12), 0.0, trial),
                                 x, opts, sys, context + " (source=%.3f)" % trial)
        except (ConvergenceError, CircuitError):
            ds *= 0.25
            if ds < 1e-6:
                break
            continue
        x, s = xn, trial
        ds = min(ds * 1.5, 0.25)
    if s >= 1.0:
        return newton_solve(make_assemble(opts.gmin, 0.0, 1.0), x, opts, sys, context)

    raise type(first)(str(first))
