"""Time-domain adjoint sensitivity analysis.

The transient solve is a sequence of implicit steps

    G_n(x_n, x_{n-1}, d_{n-1}, xdot_{n-1}; p) = 0                    (n = 1..M)
    d_n    = alpha_n (q(x_n;p) - q(x_{n-1};p)) + beta_n d_{n-1}
    xdot_n = alpha_n (x_n - x_{n-1})           + beta_n xdot_{n-1}

For an objective ``Phi = sum_n phi(x_n, p)`` the *discrete* adjoint is a
single backward sweep: at each step one transposed linear solve with the
already-assembled Jacobian, plus one vector-Jacobian product for the
parameter and history terms.  No forward solve is ever repeated, so the cost
of a gradient with respect to *any number* of parameters is one extra sweep.

Because the sweep differentiates exactly the discretised equations, it agrees
with reverse-mode autograd through the same solve to round-off, and with
central finite differences to the accuracy of the difference formula.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

DTYPE = torch.float64

__all__ = ["adjoint_sweep", "op_adjoint", "TranAutograd", "OpAutograd"]


class _require_grad:
    """Temporarily mark parameter leaves as requiring grad.

    ``adjoint_grad`` differentiates with respect to parameters that the user
    never had to flag, so the sweep flips the flag itself and restores it.
    """

    def __init__(self, params):
        self.params = [p for p in params if not p.requires_grad]

    def __enter__(self):
        for p in self.params:
            p.requires_grad_(True)
        return self

    def __exit__(self, *exc):
        for p in self.params:
            p.requires_grad_(False)
            p.grad = None
        return False


def _grad(outputs, inputs, grad_outputs, retain):
    if not inputs:
        return []
    g = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs,
                            retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(i) if gi is None else gi for i, gi in zip(inputs, g)]


def adjoint_sweep(sys, rec, gXout, params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Backward sweep over a :class:`~.analyses.TranRecord`.

    Parameters
    ----------
    sys : MNASystem
    rec : TranRecord
    gXout : (N_out, n) tensor
        ``dPhi/dX`` at the recorded output points.
    params : sequence of float64 leaf tensors

    Returns
    -------
    list[torch.Tensor]
        ``dPhi/dp`` for each entry of *params*, in the same order.
    """
    params = list(params)
    with _require_grad(params):
        return _adjoint_sweep(sys, rec, gXout, params)


def _adjoint_sweep(sys, rec, gXout, params):
    n = rec.n
    X = rec.X
    M = X.shape[0]
    has_blocks = bool(sys.blocks)
    grads = [torch.zeros_like(p) for p in params]

    gXall = torch.zeros(M, n, dtype=DTYPE)
    for j, i in enumerate(rec.out_idx):
        gXall[i] = gXall[i] + gXout[j]

    gx = gXall[M - 1].clone()
    gd = torch.zeros(n, dtype=DTYPE)
    gxd = torch.zeros(n, dtype=DTYPE)

    for k in range(M - 1, 0, -1):
        alpha = rec.alphas[k]
        beta = rec.betas[k]
        t = rec.times[k]
        tp = rec.times[k - 1]

        # --- Jacobian of the step (values only) --------------------------
        ev = sys.eval(X[k], t,
                      xdot=rec.XD[k] if (has_blocks and rec.XD is not None) else None,
                      need_jac=True, limiting=False, gmin=rec.gmin, dc=False)
        J = ev["Jf"] + alpha * ev["Jq"]
        if ev["Jbx"] is not None:
            J = J + ev["Jbx"] + alpha * ev["Jbd"]

        # --- differentiable replay of the step ---------------------------
        xn = X[k].detach().clone().requires_grad_(True)
        xp = X[k - 1].detach().clone().requires_grad_(True)
        dp = rec.D[k - 1].detach().clone().requires_grad_(True)
        xdp = ((rec.XD[k - 1] if rec.XD is not None
                else torch.zeros(n, dtype=DTYPE))
               .detach().clone().requires_grad_(True))

        # grad mode is off inside autograd.Function.backward, so re-enable it
        sys.attach_params = True
        try:
            with torch.enable_grad():
                xd = (alpha * (xn - xp) + beta * xdp) if has_blocks else None
                en = sys.eval(xn, t, xdot=xd, need_jac=False, limiting=False,
                              gmin=rec.gmin, dc=False)
                epv = sys.eval(xp, tp, need_jac=False, limiting=False,
                               gmin=rec.gmin, dc=False, charge_only=True)
                dn = alpha * (en["q"] - epv["q"]) + beta * dp
                G = en["f"] + dn
                if en["rb"] is not None:
                    G = G + en["rb"]
        finally:
            sys.attach_params = False

        outs = [dn]
        gouts = [gd]
        if has_blocks:
            outs.append(xd)
            gouts.append(gxd)

        # contribution of the auxiliary states to dPhi/dx_n
        extra = _grad(outs, [xn], gouts, retain=True)
        gx = gx + extra[0]

        lam = torch.linalg.solve(J.transpose(0, 1), gx)

        hist = [xp, dp] + ([xdp] if has_blocks else [])
        res = _grad([G] + outs, hist + params, [-lam] + gouts, retain=False)
        gx = gXall[k - 1] + res[0]
        gd = res[1]
        off = 2
        if has_blocks:
            gxd = res[2]
            off = 3
        for j in range(len(params)):
            grads[j] = grads[j] + res[off + j]

    # ---- the initial operating point -------------------------------------
    op = rec.op
    ev = sys.eval(X[0], op.t, need_jac=True, limiting=False, gmin=op.gmin,
                  dc=op.dc)
    J0 = ev["Jf"]
    if ev["Jbx"] is not None:
        J0 = J0 + ev["Jbx"]
    mask = torch.ones(n, dtype=DTYPE)
    if op.pinned:
        idx = torch.tensor([p[0] for p in op.pinned], dtype=torch.long)
        J0 = J0.clone()
        J0[idx, :] = 0.0
        J0[idx, idx] = 1.0
        mask[idx] = 0.0
    qmask = rec.qmask if rec.qmask is not None else torch.zeros(n, dtype=DTYPE)
    x0 = X[0].detach().clone().requires_grad_(True)
    sys.attach_params = True
    try:
        with torch.enable_grad():
            e0 = sys.eval(x0, op.t, need_jac=False, limiting=False,
                          gmin=op.gmin, dc=op.dc)
            F0 = e0["f"]
            if e0["rb"] is not None:
                F0 = F0 + e0["rb"]
            d0 = -F0 * qmask      # consistent dq/dt at the initial point
            G0 = F0 * mask
    finally:
        sys.attach_params = False

    gx = gx + _grad([d0], [x0], [gd], retain=True)[0]
    lam0 = torch.linalg.solve(J0.transpose(0, 1), gx)
    if params:
        res = _grad([G0, d0], params, [-lam0, gd], retain=False)
        for j in range(len(params)):
            grads[j] = grads[j] + res[j]
    return grads


def op_adjoint(sys, oprec, gx, params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Adjoint of a single DC solve: one transposed solve plus one VJP."""
    params = list(params)
    if not params:
        return []
    with _require_grad(params):
        return _op_adjoint(sys, oprec, gx, params)


def _op_adjoint(sys, oprec, gx, params):
    n = sys.n
    ev = sys.eval(oprec.x, oprec.t, need_jac=True, limiting=False,
                  gmin=oprec.gmin, dc=oprec.dc)
    J = ev["Jf"]
    if ev["Jbx"] is not None:
        J = J + ev["Jbx"]
    mask = torch.ones(n, dtype=DTYPE)
    if oprec.pinned:
        idx = torch.tensor([p[0] for p in oprec.pinned], dtype=torch.long)
        J = J.clone()
        J[idx, :] = 0.0
        J[idx, idx] = 1.0
        mask[idx] = 0.0
    lam = torch.linalg.solve(J.transpose(0, 1), gx)
    x0 = oprec.x.detach().clone().requires_grad_(True)
    sys.attach_params = True
    try:
        with torch.enable_grad():
            e0 = sys.eval(x0, oprec.t, need_jac=False, limiting=False,
                          gmin=oprec.gmin, dc=oprec.dc)
            G0 = e0["f"]
            if e0["rb"] is not None:
                G0 = G0 + e0["rb"]
            G0 = G0 * mask
    finally:
        sys.attach_params = False
    return _grad([G0], params, [-lam], retain=False)


class TranAutograd(torch.autograd.Function):
    """Makes ``TranResult.x`` differentiable; ``backward`` *is* the adjoint."""

    @staticmethod
    def forward(ctx, holder, *params):
        ctx.holder = holder
        return holder["Xout"]

    @staticmethod
    def backward(ctx, gXout):
        h = ctx.holder
        grads = adjoint_sweep(h["sys"], h["rec"], gXout.contiguous(), h["params"])
        return (None,) + tuple(grads)


class OpAutograd(torch.autograd.Function):
    """Makes ``OpResult`` values differentiable via the DC adjoint."""

    @staticmethod
    def forward(ctx, holder, *params):
        ctx.holder = holder
        return holder["x"]

    @staticmethod
    def backward(ctx, gx):
        h = ctx.holder
        grads = op_adjoint(h["sys"], h["rec"], gx.contiguous(), h["params"])
        return (None,) + tuple(grads)
