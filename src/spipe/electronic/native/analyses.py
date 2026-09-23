"""Analyses: DC operating point, transient, DC sweep.

The transient integrator writes every step in the uniform form

    G_n(x_n) = f(x_n, t_n) + d_n + r_blk(x_n, xdot_n, t_n) = 0
    d_n      = alpha_n * (q(x_n) - q(x_{n-1})) + beta_n * d_{n-1}
    xdot_n   = alpha_n * (x_n - x_{n-1})       + beta_n * xdot_{n-1}

with ``(alpha, beta) = (1/h, 0)`` for backward Euler and ``(2/h, -1)`` for
trapezoidal.  Writing both the charge derivative and the external-block state
derivative through the *same* recursion is what lets the adjoint in
:mod:`.adjoint` be a single, generic backward sweep.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .mna import CircuitError, MNASystem
from .newton import ConvergenceError, NewtonOptions, dc_solve, newton_solve

DTYPE = torch.float64

__all__ = ["TranRecord", "OpRecord", "solve_op", "run_tran", "run_dc"]


class OpRecord:
    """Everything the adjoint needs about a DC solve."""

    __slots__ = ("x", "t", "dc", "gmin", "pinned", "iters")

    def __init__(self, x, t, dc, gmin, pinned, iters):
        self.x = x
        self.t = t
        self.dc = dc
        self.gmin = gmin
        self.pinned = pinned
        self.iters = iters


class TranRecord:
    """Full transient history -- the tape the adjoint sweep replays."""

    __slots__ = ("times", "alphas", "betas", "hs", "restarts", "X", "D", "XD",
                 "out_idx", "t_out", "op", "uic", "gmin", "nsub", "method",
                 "n", "newton_iters", "lte", "qmask", "be_restart")

    def __init__(self):
        self.times: List[float] = []
        self.alphas: List[float] = []
        self.betas: List[float] = []
        self.hs: List[float] = []
        self.restarts: List[bool] = []
        self.X: Optional[torch.Tensor] = None
        self.D: Optional[torch.Tensor] = None
        self.XD: Optional[torch.Tensor] = None
        self.out_idx: List[int] = []
        self.t_out: Optional[torch.Tensor] = None
        self.op: Optional[OpRecord] = None
        self.uic = False
        self.gmin = 0.0
        self.nsub = 1
        self.method = "trap"
        self.n = 0
        self.newton_iters = 0
        self.lte = 0.0
        self.qmask = None
        self.be_restart = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _apply_pins(G, J, scale, x, pinned):
    if not pinned:
        return G, J, scale
    idx = torch.tensor([p[0] for p in pinned], dtype=torch.long)
    val = torch.tensor([p[1] for p in pinned], dtype=DTYPE)
    G = G.clone()
    G[idx] = x[idx] - val
    J = J.clone()
    J[idx, :] = 0.0
    J[idx, idx] = 1.0
    scale = scale.clone()
    scale[idx] = x[idx].abs() + val.abs()
    return G, J, scale


def _dc_assembler(sys: MNASystem, t, dc, pinned):
    def make(gmin, gshunt, src_scale):
        def assemble(x, xold):
            ev = sys.eval(x, t, xdot=None, need_jac=True, xold=xold,
                          gmin=gmin, gshunt=gshunt, src_scale=src_scale,
                          dc=dc, limiting=True)
            G, J = ev["f"], ev["Jf"]
            scale = ev["fmag"]
            if gshunt:
                scale = scale + gshunt * x.abs()
            if ev["rb"] is not None:
                G = G + ev["rb"]
                J = J + ev["Jbx"]
            G, J, scale = _apply_pins(G, J, scale, x, pinned)
            return G, J, scale, ev["limited"]
        return assemble
    return make


def solve_op(sys: MNASystem, opts: NewtonOptions, t=0.0, dc=True, x0=None,
             pinned=None, context="DC operating point") -> OpRecord:
    """Solve the operating point (with gmin / source-stepping fallbacks)."""
    if x0 is None:
        x0 = sys.block_initial_state()
        for nm, val in sys.parsed.nodeset.items():
            try:
                x0[sys.node_index(nm)] = float(val)
            except CircuitError:
                pass
    x, iters = dc_solve(_dc_assembler(sys, t, dc, pinned), x0, opts, sys, context)
    return OpRecord(x.detach(), t, dc, opts.gmin, pinned, iters)


def _ic_pins(sys: MNASystem) -> List[Tuple[int, float]]:
    """Initial-condition constraints used by ``uic``.

    Follows SPICE ``UIC`` semantics: ``.ic v(node)=`` pins a node, every
    capacitor with one terminal on ground pins the other terminal to its
    ``IC=`` value (0 by default), and every inductor pins its branch current
    to its ``IC=`` value (0 by default).  ``.ic`` wins over a device ``IC=``.
    A capacitor that floats between two live nodes cannot be expressed as a
    single-unknown constraint and is skipped.
    """
    pins: Dict[int, float] = {}
    for nm, val in sys.parsed.ic.items():
        try:
            idx = sys.node_index(nm)
        except CircuitError:
            continue
        if idx < sys.n_nodes:
            pins[idx] = float(val)
    for grp in sys.groups:
        letter = getattr(grp, "letter", "")
        if letter == "c":
            for dev in grp.devices:
                ic = float(dev.params["IC"].detach())
                ai = sys.node_index(dev.nodes[0])
                bi = sys.node_index(dev.nodes[1])
                if bi == sys.ground and ai < sys.n_nodes:
                    pins.setdefault(ai, ic)
                elif ai == sys.ground and bi < sys.n_nodes:
                    pins.setdefault(bi, -ic)
        elif letter == "l":
            for dev in grp.devices:
                ic = float(dev.params["IC"].detach())
                pins.setdefault(sys.branch_index(dev.name, 0), ic)
    return sorted(pins.items())


# --------------------------------------------------------------------------
# transient
# --------------------------------------------------------------------------

_JUMP_EPS = 1e-6      # pre-breakpoint sliver, as a fraction of the local step


def _time_grid(sys, tstep, tstop, tstart, nsub, be_restart=False):
    """Deterministic internal time grid.

    Print points and source breakpoints are *exact* grid points; each interval
    between consecutive such points is split into ``nsub`` uniform internal
    steps.  Where a source is genuinely discontinuous, a sliver step is
    inserted just before the breakpoint so that the discontinuity is never
    integrated over: the step landing on ``t_bp - delta`` sees the old value
    and the sliver step carries the jump.  The sequence depends only on the
    netlist's timing constants, never on the solution, which is what keeps
    gradients comparable with central finite differences.
    """
    tstep = float(tstep)
    tstop = float(tstop)
    npr = int(math.floor(tstop / tstep + 1e-9))
    prints = [k * tstep for k in range(npr + 1)]
    if prints[-1] < tstop * (1 - 1e-12):
        prints.append(tstop)
    prints[-1] = min(prints[-1], tstop)

    bps = [b for b in sys.breakpoints(0.0, tstop) if 0.0 < b < tstop]
    tol = tstep * 1e-9
    marks = sorted(set(prints) | set(bps))
    merged = [marks[0]]
    is_print = [True]
    prset = set(prints)
    for m in marks[1:]:
        if m - merged[-1] <= tol:
            if m in prset:
                merged[-1] = m
                is_print[-1] = True
            continue
        merged.append(m)
        is_print.append(m in prset)

    # classify breakpoints: jump (discontinuous value) vs kink (continuous)
    bpset = set(bps)
    jumps = set()
    for k in range(1, len(merged)):
        tb = merged[k]
        if tb not in bpset:
            continue
        d = (merged[k] - merged[k - 1]) * _JUMP_EPS
        if d <= 0:
            continue
        try:
            v0 = sys.source_values(tb - 2.0 * d)
            v1 = sys.source_values(tb - d)
            v2 = sys.source_values(tb)
        except Exception:
            continue
        if not v2.numel():
            continue
        # A genuine jump shows a step between the last probe and the
        # breakpoint that a finite slope cannot explain: d1 >> d2.  A steep
        # but finite ramp has d1 ~ d2 and must NOT get a sliver step, or the
        # integrator ends up with an absurdly small, ill-conditioned step.
        d1 = float((v2 - v1).abs().max())
        d2 = float((v1 - v0).abs().max())
        if d1 > 1e-9 and d1 > 100.0 * d2:
            jumps.add(k)

    times = [0.0]
    restarts = [bool(be_restart)]
    hnom = [0.0]
    out_idx = []
    if 0.0 >= tstart - tol:
        out_idx.append(0)
    for k in range(1, len(merged)):
        ta, tb = merged[k - 1], merged[k]
        span = tb - ta
        if k in jumps:
            span = span * (1.0 - _JUMP_EPS)
        rst = be_restart and (k - 1) in jumps
        # A plain print-to-print interval is `tstep` long by construction; the
        # difference of two grid marks wobbles in the last ulps and would
        # otherwise present a different step size (and a different LU) at
        # nearly every step.
        nominal = span
        if (is_print[k] and is_print[k - 1] and k not in jumps
                and abs(span - tstep) <= 1e-6 * tstep):
            nominal = tstep
        hstep = nominal / nsub
        for j in range(1, nsub + 1):
            times.append(ta + span * j / nsub)
            restarts.append(bool(rst and j == 1))
            hnom.append(hstep)
        if k in jumps:
            times.append(tb)
            restarts.append(False)
            hnom.append(tb - (ta + span))
        if is_print[k] and tb >= tstart - tol:
            out_idx.append(len(times) - 1)
    return times, restarts, out_idx, _snap_steps(hnom)


def _snap_steps(hnom, rtol=1e-11):
    """Collapse step lengths that differ only by float round-off.

    ``k*tstep`` differences wobble in the last ulps, so a nominally uniform
    grid otherwise presents a different ``h`` -- and therefore a different
    integration coefficient and a different LU factorisation -- at almost
    every step.  Snapping at 1e-11 relative (the wobble is ~1e-13) restores
    exact uniformity, which is what the discretisation meant in the first
    place, and lets whole runs of steps be solved as one batch.
    """
    canon = []
    out = []
    for v in hnom:
        hit = None
        for c in canon:
            if abs(v - c) <= rtol * abs(c):
                hit = c
                break
        if hit is None:
            canon.append(v)
            hit = v
        out.append(hit)
    return out


def _scan_affine(B, C, w0, L):
    """Solve the affine recursion ``w_j = B w_{j-1} + C[j]`` for ``j = 0..K-1``.

    With ``L > 1`` this is a blocked (parallel-prefix) scan: the powers
    ``B^0..B^L`` are formed once, the per-block forcing terms are accumulated
    with two batched matrix products, and only ``K/L`` steps stay sequential.
    That is what removes the Python-level per-timestep cost from a linear
    transient.
    """
    K, n = C.shape
    if L <= 1 or K < 4 * L:
        W = torch.empty(K, n, dtype=DTYPE)
        w = w0
        for j in range(K):
            w = B @ w + C[j]
            W[j] = w
        return W

    nb = (K + L - 1) // L
    pad = nb * L - K
    if pad:
        C = torch.cat([C, torch.zeros(pad, n, dtype=DTYPE)])
    Cb = C.view(nb, L, n)

    P = torch.empty(L + 1, n, n, dtype=DTYPE)
    P[0] = torch.eye(n, dtype=DTYPE)
    for i in range(1, L + 1):
        P[i] = P[i - 1] @ B

    # block boundaries: w_{(b+1)L-1} = B^L w_{bL-1} + sum_i B^{L-1-i} C[b,i]
    S = torch.einsum('iac,bic->ba', torch.flip(P[:L], (0,)), Cb)
    BL = P[L]
    WB = torch.empty(nb, n, dtype=DTYPE)
    w = w0
    for b in range(nb):
        WB[b] = w
        w = BL @ w + S[b]

    # inside each block, in parallel across blocks
    Ltri = torch.zeros(L, n, L, n, dtype=DTYPE)
    for j in range(L):
        for i in range(j + 1):
            Ltri[j, :, i, :] = P[j - i]
    T = (Cb.reshape(nb, L * n) @ Ltri.reshape(L * n, L * n).transpose(0, 1))
    W = torch.einsum('jac,bc->bja', P[1:L + 1], WB) + T.view(nb, L, n)
    return W.reshape(nb * L, n)[:K]


def _linear_runs(times, restarts, hnom, method, M):
    """Maximal runs of consecutive steps sharing ``(alpha, beta)``."""
    runs = []
    k = 1
    while k < M:
        h = hnom[k]
        be = bool(restarts[k]) or method == "be"
        j = k + 1
        while (j < M and hnom[j] == h
               and (bool(restarts[j]) or method == "be") == be):
            j += 1
        alpha = (1.0 / h) if be else (2.0 / h)
        beta = 0.0 if be else -1.0
        runs.append((k, j, alpha, beta, h))
        k = j
    return runs


_LIN_CHUNK = 1 << 14


def _linear_transient(sys, X, D, times, restarts, hnom, method, Jf0, Jq0,
                      q0, d0):
    """Vectorised transient for a circuit with no nonlinear device.

    Because ``f(x,t) = Jf x + f(0,t)`` and ``q(x) = Jq x``, the step

        (Jf + alpha Jq) x_k = -f(0,t_k) + alpha q_{k-1} - beta d_{k-1}

    collapses onto a single carried vector ``w_k = alpha q_k - beta d_k``:

        x_k = A^-1 (w_{k-1} - f_k)
        w_k = B w_{k-1} - N f_k,   B = N + beta I,  N = (1-beta) alpha Jq A^-1

    ``B`` and ``N`` are constant for a run of equal step sizes, so the whole
    run is one affine scan plus one batched triangular solve.  ``N`` is built
    by *solving* with the factorisation, never by forming ``inv(A)``.
    """
    n = X.shape[1]
    M = X.shape[0]
    lu_cache = {}
    mat_cache = {}
    L = 32 if n <= 8 else (8 if n <= 24 else 1)
    tall = torch.tensor(times, dtype=DTYPE)
    eye = torch.eye(n, dtype=DTYPE)
    q_prev, d_prev = q0, d0

    for k0, k1, alpha, beta, h in _linear_runs(times, restarts, hnom, method, M):
        lu = lu_cache.get(alpha)
        if lu is None:
            if len(lu_cache) > 64:
                lu_cache.clear()
                mat_cache.clear()
            A = Jf0 + alpha * Jq0
            try:
                lu = torch.linalg.lu_factor(A)
            except Exception:
                bad = sys.zero_rows(A)
                raise CircuitError(
                    "singular MNA matrix in the transient step at t=%g: %s"
                    % (times[k0], ", ".join(bad) if bad else "(no zero rows)"))
            lu_cache[alpha] = lu
            # N = (1-beta) alpha Jq A^-1, obtained as (A^T)^-1 Jq^T transposed
            N = torch.linalg.lu_solve(lu[0], lu[1], Jq0.transpose(0, 1),
                                      adjoint=True).transpose(0, 1)
            N = ((1.0 - beta) * alpha) * N
            mat_cache[alpha] = (N, N + beta * eye)
        N, B = mat_cache[alpha]

        w = alpha * q_prev - beta * d_prev
        for c0 in range(k0, k1, _LIN_CHUNK):
            c1 = min(c0 + _LIN_CHUNK, k1)
            F0 = sys.source_residual_batch(tall[c0:c1])
            C = -(F0 @ N.transpose(0, 1))
            W = _scan_affine(B, C, w, L)
            Wprev = torch.cat([w.unsqueeze(0), W[:-1]])
            Xc = torch.linalg.lu_solve(
                lu[0], lu[1], (Wprev - F0).transpose(0, 1)).transpose(0, 1)
            Qc = Xc @ Jq0.transpose(0, 1)
            # d_k = -(f(x_k, t_k)) follows from the DAE itself and is exact to
            # round-off, whereas alpha*(q_k - q_{k-1}) + beta*d_{k-1} would
            # accumulate; both are algebraically the same at the solution.
            Dc = -(Xc @ Jf0.transpose(0, 1) + F0)
            X[c0:c1] = Xc
            D[c0:c1] = Dc
            # re-anchor the carried state on the freshly solved x rather than
            # letting the scan's own recursion drift over a long run
            q_prev, d_prev = Qc[-1], Dc[-1]
            w = alpha * q_prev - beta * d_prev
    if not torch.isfinite(X).all():
        raise ConvergenceError("non-finite solution in the transient analysis")


def _tran_once(sys: MNASystem, opts: NewtonOptions, tstep, tstop, tstart,
               uic, method, nsub, be_restart=False) -> TranRecord:
    rec = TranRecord()
    rec.uic = uic
    rec.gmin = opts.gmin
    rec.nsub = nsub
    rec.method = method
    n = sys.n
    rec.n = n

    times, restarts, out_idx, hnom = _time_grid(sys, tstep, tstop, tstart,
                                                nsub, be_restart)
    M = len(times)

    # ---- initial state ---------------------------------------------------
    pins = _ic_pins(sys) if uic else None
    op = solve_op(sys, opts, t=0.0, dc=False, pinned=pins,
                  context="transient operating point")
    rec.op = op
    x = op.x

    ev = sys.eval(x, 0.0, need_jac=True, limiting=False, gmin=opts.gmin)
    q_prev = ev["q"].detach()
    # dq/dt at t0 follows from the DAE itself: f + dq/dt = 0.  For a proper
    # operating point f is zero, so d0 = 0; with `uic` it is the consistent
    # initial charge derivative.  Rows that carry no charge get zero.
    qmask = (ev["Jq"].abs().sum(dim=1) > 0).to(DTYPE)
    rec.qmask = qmask
    d0 = -(ev["f"] + (ev["rb"] if ev["rb"] is not None else 0.0)) * qmask

    has_blocks = bool(sys.blocks)
    X = torch.empty(M, n, dtype=DTYPE)
    D = torch.zeros(M, n, dtype=DTYPE)
    XD = torch.zeros(M, n, dtype=DTYPE) if has_blocks else None
    X[0] = x
    D[0] = d0
    rec.times = [0.0]
    rec.alphas = [0.0]
    rec.betas = [0.0]
    rec.hs = [0.0]
    rec.restarts = [bool(restarts[0])]

    d_prev = d0
    xd_prev = torch.zeros(n, dtype=DTYPE)
    x_prev = x
    total_iters = 0

    # ---- step metadata (identical for both paths) ------------------------
    for k in range(1, M):
        h = hnom[k]
        if h <= 0:
            raise CircuitError("non-increasing transient time grid at t=%g"
                               % times[k])
        use_be = bool(restarts[k]) or method == "be"
        rec.times.append(times[k])
        rec.alphas.append((1.0 / h) if use_be else (2.0 / h))
        rec.betas.append(0.0 if use_be else -1.0)
        rec.hs.append(h)
        rec.restarts.append(bool(restarts[k]))

    linear = sys.is_linear
    if linear:
        # No nonlinear device: the whole transient is one affine recursion
        # plus batched triangular solves -- see _linear_transient().
        _linear_transient(sys, X, D, times, restarts, hnom, method,
                          ev["Jf"], ev["Jq"], q_prev, d0)
        total_iters = M - 1
    else:
        for k in range(1, M):
            t = times[k]
            h = hnom[k]
            alpha, beta = rec.alphas[k], rec.betas[k]

            def assemble(xx, xold, _t=t, _a=alpha, _b=beta, _xp=x_prev,
                         _qp=q_prev, _dp=d_prev, _xdp=xd_prev):
                xd = (_a * (xx - _xp) + _b * _xdp) if has_blocks else None
                e = sys.eval(xx, _t, xdot=xd, need_jac=True, xold=xold,
                             gmin=opts.gmin, dc=False, limiting=True)
                G = e["f"] + _a * (e["q"] - _qp) + _b * _dp
                J = e["Jf"] + _a * e["Jq"]
                scale = (e["fmag"] + abs(_a) * (e["qmag"] + _qp.abs())
                         + abs(_b) * _dp.abs())
                if e["rb"] is not None:
                    G = G + e["rb"]
                    J = J + e["Jbx"] + _a * e["Jbd"]
                return G, J, scale, e["limited"]

            try:
                xn, it = newton_solve(assemble, x_prev, opts, sys,
                                      context="transient step t=%.6g s" % t)
            except ConvergenceError as exc:
                raise ConvergenceError(
                    "%s (step %d of %d, h=%.3g s, method=%s)"
                    % (exc, k, M - 1, h,
                       "BE" if rec.betas[k] == 0.0 else "TRAP"))
            total_iters += it
            e = sys.eval(xn, t, need_jac=False, limiting=False, gmin=opts.gmin)
            q_n = e["q"].detach()
            d_n = alpha * (q_n - q_prev) + beta * d_prev
            X[k] = xn
            D[k] = d_n
            if has_blocks:
                XD[k] = alpha * (xn - x_prev) + beta * xd_prev
                xd_prev = XD[k]
            x_prev, q_prev, d_prev = xn, q_n, d_n

    rec.X, rec.D, rec.XD = X, D, XD
    rec.out_idx = out_idx
    rec.t_out = torch.tensor([times[i] for i in out_idx], dtype=DTYPE)
    rec.newton_iters = total_iters
    rec.lte = _worst_lte(sys, rec, opts)
    return rec


def _lte_target(sys, opts):
    """LTE tolerance and refinement cap for this circuit.

    Linear circuits go through the cached-factorisation fast path, so extra
    steps are almost free and a tight target is affordable.  Circuits with
    nonlinear devices cost a full Newton solve per step, so they use the
    looser ``lte_reltol_nl`` and a smaller refinement cap.
    """
    if sys.is_linear:
        return opts.lte_reltol, 64
    return max(opts.lte_reltol, opts.lte_reltol_nl), 8


def _worst_lte(sys, rec, opts) -> float:
    """Worst normalised local truncation error over the run.

    Trapezoidal LTE is ``-h^3/12 * x'''``; on a uniform sub-grid the third
    divided difference gives ``LTE ~ |D3 x| / 12``.
    """
    X = rec.X
    M = X.shape[0]
    if M < 5:
        return 0.0
    nn = sys.n_nodes
    if nn == 0:
        return 0.0
    hs = rec.hs
    rst = rec.restarts
    idx = []
    for k in range(3, M):
        if rst[k] or rst[k - 1] or rst[k - 2]:
            continue
        h0 = hs[k]
        if abs(hs[k - 1] - h0) > 1e-12 * h0 or abs(hs[k - 2] - h0) > 1e-12 * h0:
            continue
        idx.append(k)
    if not idx:
        return 0.0
    i = torch.tensor(idx, dtype=torch.long)
    d3 = (X[i, :nn] - 3.0 * X[i - 1, :nn] + 3.0 * X[i - 2, :nn] - X[i - 3, :nn]).abs() / 12.0
    ref = torch.maximum(X[i, :nn].abs(), X[i - 1, :nn].abs())
    rtol, _ = _lte_target(sys, opts)
    tol = opts.lte_abstol + rtol * ref
    return float((d3 / tol).max())


def run_tran(sys: MNASystem, opts: NewtonOptions, tstep, tstop, tstart=0.0,
             uic=False, method="trap", nsub=None, max_nsub=16,
             adaptive=True, be_restart=False) -> TranRecord:
    """Run a transient analysis.

    Step control: the internal grid subdivides every print/breakpoint interval
    into ``nsub`` uniform steps.  When ``adaptive`` is true (default) ``nsub``
    is raised -- globally, so that the step sequence stays a deterministic
    function of the netlist rather than of the solution -- until the worst
    local truncation error meets ``lte_reltol``/``lte_abstol``.
    """
    sys.set_tran_defaults(tstep, tstop)
    if float(tstop) <= 0:
        raise CircuitError("tran needs tstop > 0 (got %r)" % (tstop,))
    if float(tstep) <= 0:
        raise CircuitError("tran needs tstep > 0 (got %r)" % (tstep,))
    ns = int(nsub) if nsub else 1
    rec = _tran_once(sys, opts, tstep, tstop, tstart, uic, method, ns, be_restart)
    rec.be_restart = be_restart
    if not adaptive or nsub:
        return rec
    _, cap = _lte_target(sys, opts)
    max_nsub = min(int(max_nsub), cap)
    guard = 0
    while rec.lte > 1.0 and ns < max_nsub and guard < 8:
        guard += 1
        grow = max(2.0, rec.lte ** (1.0 / 3.0))
        ns = min(max_nsub, int(2 ** math.ceil(math.log2(ns * grow))))
        rec = _tran_once(sys, opts, tstep, tstop, tstart, uic, method, ns,
                         be_restart)
        rec.be_restart = be_restart
    return rec


# --------------------------------------------------------------------------
# DC sweep
# --------------------------------------------------------------------------

def run_dc(sys: MNASystem, opts: NewtonOptions, source: str, start, stop, step):
    """Sweep the DC value of an independent source."""
    name = str(source).lower()
    dev = sys.device_by_name.get(name)
    if dev is None or name[0] not in ("v", "i"):
        raise CircuitError(
            "'.dc' can only sweep an independent V or I source; %r is not one"
            % source
        )
    from .devices.sources import DCWave
    saved_wf = dev.extra.get("wf")
    saved_params = dict(dev.params)
    grp = sys.group_of_device[name]

    npts = int(round(abs((float(stop) - float(start)) / float(step)))) + 1
    vals = [float(start) + i * float(step) for i in range(npts)]
    xs = []
    x0 = None
    try:
        for v in vals:
            dev.extra["wf"] = DCWave()
            dev.params = {"DC": torch.tensor(float(v), dtype=DTYPE)}
            grp.invalidate_cache()
            grp._cache_key = None
            rec = solve_op(sys, opts, t=0.0, dc=True, x0=x0,
                           context="DC sweep %s=%g" % (source, v))
            x0 = rec.x.clone()
            xs.append(rec.x)
    finally:
        dev.extra["wf"] = saved_wf
        dev.params = saved_params
        grp.invalidate_cache()
        grp._cache_key = None
    return torch.tensor(vals, dtype=DTYPE), torch.stack(xs)
