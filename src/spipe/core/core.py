from typing import List, Tuple, Optional, Dict, Any, Callable, Mapping
from spipe.electronic.electronic import Electronic
from spipe.photonic.photonic import Photonic
import torch
from spipe import config
from spipe.utils import extract
import logging
import time
import math
import warnings
__all__ = ['Circuit', 'solve_fixed_point', 'FixedPointError', 'FixedPointDivergence',
           'FixedPointNotConverged', 'CouplingJacobianError', 'solve_coupling_system']

logger = logging.getLogger(__name__)


def _parse_file(file_path: str) -> Tuple[List[str], List[str], torch.Tensor]:
    '''
    Parse a given file specified by file path and return the electronic and photonic components.

    Args:
        file_path (str): the file path of the simulation file

    Returns:
        tuple[str, str]: A tuple containing the electronic and photonic sections.
    '''

    p_start, e_start, content, time = -1, -1, [], None
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if line.startswith('.electronic'):
                if e_start == -1:
                    e_start = i
                else:
                    raise RuntimeError(f'.electronic syntax occurs at least twice at line {e_start} and {i}')
            if line.lower().startswith('.tran'):
                initial_strings, strings, kv_pair = extract(line, convert_numeric=True)
                # real_dtype, not torch's float32 default: this grid is the interpolation abscissa
                # for the SPICE results and becomes the 'time' attribute of every device model.
                time = torch.linspace(float(strings[0]), float(strings[1]), int(strings[2]),
                                      dtype=config['real_dtype']).to(config['device'])
                line = ''
            if line.startswith('.photonic'):
                if p_start == -1:
                    p_start = i
                else:
                    raise RuntimeError(f'.photonic syntax occurs at least twice at line {p_start} and {i}')
            content.append(line.split('#')[0] + '\n')


    if p_start != -1 and e_start != -1:
        if p_start < e_start:
            return content[e_start + 1:], content[p_start + 1:e_start], time
        else:
            return content[e_start + 1:p_start], content[p_start + 1:], time
    else:
        raise RuntimeError(f'.photonic and .electronic syntax must both be used to define circuits.')


class FixedPointError(RuntimeError):
    """The electronic/photonic fixed-point iteration did not produce a usable answer.

    The whole point of this exception is that SPIPE never returns the last iterate of a loop
    that did not converge.  A circuit with feedback (paper, Definition 1: a photocurrent that
    reaches a modulator drive) can oscillate or diverge under the plain Picard iteration, and
    the last iterate of such a run is not an approximation of anything.

    Attributes
    ----------
    residuals : list of float
        The full residual history ``||x_k - step(x_k)|| / sqrt(numel)``, one entry per
        iteration actually performed.
    iters : int
        Number of iterations performed (== ``len(residuals)``).
    info : dict
        The same ``info`` dictionary :func:`solve_fixed_point` would have returned, with
        ``info['converged'] is False``.
    x : torch.Tensor
        The last iterate.  Provided for post-mortem inspection only -- it is *not* a solution.
    """

    def __init__(self, message: str, info: Dict):
        super().__init__(message)
        self.info = info
        self.residuals = info.get('residuals', [])
        self.iters = info.get('iters', len(self.residuals))
        self.x = info.get('x', None)


class FixedPointDivergence(FixedPointError):
    """The residual blew up (grew without bound, or became NaN/Inf)."""


class FixedPointNotConverged(FixedPointError):
    """``max_iter`` iterations were spent without meeting the convergence criterion."""


class CouplingJacobianError(RuntimeError):
    """``(I - g'f')`` is singular or ill conditioned at the converged fixed point.

    The implicit function theorem turns the converged coupled simulation into

    .. math::  \\frac{dV}{d\\theta} = (I - g'f')^{-1}\\,\\frac{\\partial g}{\\partial\\theta}

    and that inverse exists only when the electronic/photonic loop gain is bounded away from
    one.  When it is not, the *forward* problem is the one that is ill posed: the circuit sits
    on (or beyond) the edge of an instability, and an infinitesimal change of the design
    variable moves the operating point by an unbounded amount.  Returning a number there
    would be worse than useless, so SPIPE raises.

    Attributes
    ----------
    condition : float or None
        2-norm condition number of ``I - (g'f')^T``, when it was formed densely.
    spectral_radius : float or None
        Estimated spectral radius of ``g'f'``, when the matrix-free path was used.  A value
        at or above 1 means the Picard/Neumann series for the inverse does not converge.
    size : int
        Number of coupling unknowns, ``len(time) * number of modulators``.
    """

    def __init__(self, message: str, condition: Optional[float] = None,
                 spectral_radius: Optional[float] = None, size: int = 0):
        super().__init__(message)
        self.condition = condition
        self.spectral_radius = spectral_radius
        self.size = size


def _residual_report(residuals: List[float], head: int = 6, tail: int = 6) -> str:
    """A compact, deterministic rendering of a residual history for an error message."""
    if len(residuals) <= head + tail:
        shown = ', '.join(f'{r:.6e}' for r in residuals)
    else:
        shown = (', '.join(f'{r:.6e}' for r in residuals[:head]) + ', ... , ' +
                 ', '.join(f'{r:.6e}' for r in residuals[-tail:]))
    return f'[{shown}]'


def _anderson_gamma(delta_f: torch.Tensor, f: torch.Tensor, reg: float) -> Optional[torch.Tensor]:
    """Least-squares coefficients ``gamma = argmin ||delta_f @ gamma - f||``, Tikhonov damped.

    ``delta_f`` has shape ``(n, m)`` with ``m`` the number of history vectors.  The normal
    equations are used rather than :func:`torch.linalg.lstsq` because ``delta_f`` becomes
    numerically rank deficient exactly when the iteration is about to converge, and a plain
    least-squares driver then returns a huge, meaningless ``gamma``; the ridge term
    ``reg * trace(A)/m`` keeps the step bounded and degrades gracefully to plain (damped)
    Picard.  Returns ``None`` if the system cannot be solved or the answer is not finite,
    in which case the caller falls back to Picard.
    """
    a = delta_f.conj().transpose(0, 1) @ delta_f                      # (m, m)
    b = delta_f.conj().transpose(0, 1) @ f                            # (m,)
    m = a.shape[0]
    trace = torch.diagonal(a).abs().sum()
    if not torch.isfinite(trace) or float(trace) <= 0.0:
        return None
    ridge = reg * trace / m
    a = a + ridge * torch.eye(m, dtype=a.dtype, device=a.device)
    try:
        gamma = torch.linalg.solve(a, b.unsqueeze(-1)).squeeze(-1)
    except Exception:                                    # pragma: no cover - LAPACK dependent
        return None
    if not bool(torch.isfinite(gamma).all()):
        return None
    return gamma


def solve_fixed_point(step: Callable[[torch.Tensor], torch.Tensor],
                      x0: torch.Tensor,
                      config: Mapping,
                      max_iter: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
    """Iterate ``x <- step(x)`` to a fixed point.

    This is the whole electronic/photonic coupling of SPIPE, expressed without any reference
    to SPICE or to the photonic solver, so that its convergence behaviour can be tested with a
    three-line pure-Python ``step``.  :meth:`Circuit.gradient_free_simulate` is a thin wrapper
    over it and the two therefore cannot drift apart.

    :param step: one full round trip, i.e. the composition of the photonic and the electronic
        solve.  Maps a tensor to a tensor of the same shape.  Called **once per iteration**;
        Anderson acceleration adds no extra evaluations.
    :param x0: the initial guess.  Anything :func:`torch.as_tensor` accepts.
    :param config: a mapping supplying the solver settings, normally :data:`spipe.config`:

        ``rtol``, ``atol``
            The convergence criterion, see below.  Required.
        ``max_iter``
            Iteration budget, used when the ``max_iter`` argument is ``None``.
        ``anderson_depth`` (default 5)
            Number of history vectors kept by Anderson acceleration.  ``0`` reduces the solver
            to damped Picard.
        ``anderson_reg`` (default 1e-10)
            Relative Tikhonov regularisation of the Anderson least-squares problem.
        ``damping`` (default 1.0)
            Initial (and maximum) mixing factor ``beta``.  ``1.0`` is undamped.
        ``min_damping`` (default 0.05)
            Floor of the adaptive under-relaxation.
        ``backoff_factor`` (default 10)
            How far above the *best* residual so far an iteration has to land, twice running,
            before the solver halves ``beta`` and drops the Anderson history.
        ``divergence_factor`` (default 1e4)
            Growth of the residual over the best residual seen so far that is treated as
            divergence, see below.
    :param max_iter: overrides ``config['max_iter']`` for this call.
    :returns: ``(x_star, info)``.  ``info`` carries at least ``'iters'`` (int),
        ``'residuals'`` (list of float) and ``'converged'`` (bool, always ``True`` on return),
        plus ``'reference'``, ``'tolerance'``, ``'damping'`` and ``'x'``.
    :raises FixedPointNotConverged: the iteration budget ran out.
    :raises FixedPointDivergence: the residual became non-finite or grew without bound.

    Convergence criterion
    ---------------------
    Unchanged in meaning from the original Picard loop::

        residual  = ||x - step(x)|| / sqrt(numel)
        reference = ||x||          / sqrt(numel)
        converged = residual <= rtol * reference + atol

    Both norms are divided by ``sqrt(numel)``, which makes them RMS-per-entry quantities: the
    criterion means the same thing whatever the number of time samples or the number of
    modulators, which is the sense in which it is scale free.  It is the mixed relative /
    absolute test of the original code, not a pure relative one.

    Why not plain Picard
    --------------------
    Picard converges only for a contraction, ``||dg/dx|| < 1``.  Every example shipped with
    SPIPE is *feedback free*: no photocurrent reaches a modulator drive, so ``step`` is a
    constant map, ``dg/dx = 0``, and the iteration terminates after exactly two evaluations --
    one to produce the constant, one to confirm it.  That property is preserved here exactly,
    because the first update is taken with an empty Anderson history and ``beta = 1``, i.e. it
    *is* a Picard step.  As soon as a circuit has real feedback the contraction factor is no
    longer zero and can exceed one, and then

    * **Anderson acceleration** (the workhorse) builds the next iterate from a least-squares
      combination of the last ``anderson_depth`` residuals.  On a linear coupling it is a
      secant/Newton method and converges even when ``|dg/dx| > 1``;
    * **adaptive under-relaxation** halves ``beta`` and drops the Anderson history when two
      iterations in a row land more than ``backoff_factor`` above the best residual seen, and
      lets ``beta`` relax back towards ``damping`` while the residual keeps falling.  The
      trigger is that ratio and not "the residual went up", because on a circuit that
      oscillates the residual rises for many iterations while the iteration is still
      discovering the limit cycle, and backing off there would throw away the Anderson history
      exactly when it is the only thing that works.

    Divergence
    ----------
    A non-finite residual, or a residual more than ``divergence_factor`` times the best
    residual seen so far (checked only after the solver has had a few iterations to settle),
    raises :class:`FixedPointDivergence`.  Exhausting the budget raises
    :class:`FixedPointNotConverged`.  Both carry the full residual history on the exception,
    because silently returning the last iterate of a non-convergent loop is the worst possible
    outcome.

    Determinism
    -----------
    The solver draws no random numbers and takes no data-dependent branch other than on the
    residual values themselves, so two runs with the same ``step`` and the same ``x0`` produce
    bit-identical results.
    """
    x = x0 if isinstance(x0, torch.Tensor) else torch.as_tensor(x0)

    # `.get` with the spipe defaults, so that a caller can hand this function a three-entry
    # dict rather than the whole of spipe.config and still get the documented behaviour.
    budget = config.get('max_iter', 100) if max_iter is None else max_iter
    budget = max(1, int(budget))

    rtol, atol = float(config.get('rtol', 1e-3)), float(config.get('atol', 1e-3))
    depth = int(config.get('anderson_depth', 5))
    reg = float(config.get('anderson_reg', 1e-10))
    beta_max = float(config.get('damping', 1.0))
    beta_min = float(config.get('min_damping', 0.05))
    grow_factor = float(config.get('divergence_factor', 1e4))
    backoff_factor = float(config.get('backoff_factor', 10.0))

    beta = beta_max
    bad_steps = 0
    scale = max(1, x.numel()) ** 0.5

    residuals: List[float] = []
    dx_hist: List[torch.Tensor] = []
    df_hist: List[torch.Tensor] = []
    prev_x = prev_f = None

    def _info(converged: bool, reference: float, tolerance: float) -> Dict:
        return {'iters': len(residuals),
                'residuals': list(residuals),
                'converged': converged,
                'reference': reference,
                'tolerance': tolerance,
                'damping': beta,
                'anderson_depth': depth,
                'x': x}

    reference = tolerance = float('nan')

    for iteration in range(budget):
        g = step(x)
        if not isinstance(g, torch.Tensor):
            g = torch.as_tensor(g, dtype=x.dtype, device=x.device)
        elif g.dtype != x.dtype or g.device != x.device:
            # The SPICE back ends return float32 whatever the iterate is; keeping the whole
            # iteration in one dtype is what stops the Anderson history from becoming a mix of
            # float32 and float64 tensors.  The iterate itself is created at
            # config['real_dtype'] (see Circuit.gradient_free_simulate), so a float32 SPICE
            # result is widened here, never the float64 iterate narrowed.
            g = g.to(dtype=x.dtype, device=x.device)

        f = g - x                                            # the fixed-point residual vector
        residual = float(torch.linalg.vector_norm(f.reshape(-1)) / scale)
        reference = float(torch.linalg.vector_norm(x.reshape(-1)) / scale)
        tolerance = rtol * reference + atol
        residuals.append(residual)

        logger.debug("fixed point iter %d: rms residual=%.6e, rms |x|=%.6e, threshold=%.6e, "
                     "beta=%.3f, anderson history=%d",
                     iteration, residual, reference, tolerance, beta, len(df_hist))

        if not math.isfinite(residual):
            raise FixedPointDivergence(
                f"The electronic/photonic fixed-point iteration diverged: the residual became "
                f"non-finite ({residual}) at iteration {iteration}. Residual history: "
                f"{_residual_report(residuals)}. This circuit has feedback whose loop gain the "
                f"solver could not tame; reduce the loop gain, or lower "
                f"spipe.config['damping'] (currently {beta_max}).",
                _info(False, reference, tolerance))

        if residual <= tolerance:
            logger.debug("fixed point converged after %d iteration(s), residual=%.6e <= %.6e",
                         len(residuals), residual, tolerance)
            return x, _info(True, reference, tolerance)

        best = min(residuals)
        if iteration >= 3 and residual > grow_factor * max(best, atol):
            raise FixedPointDivergence(
                f"The electronic/photonic fixed-point iteration diverged: after {iteration + 1} "
                f"iterations the residual is {residual:.6e}, more than {grow_factor:g} times the "
                f"best residual seen ({best:.6e}). Residual history: "
                f"{_residual_report(residuals)}. This circuit has feedback whose loop gain the "
                f"solver could not tame; reduce the loop gain, or lower "
                f"spipe.config['damping'] (currently {beta_max}).",
                _info(False, reference, tolerance))

        # ---------------------------------------------------------------- adaptive damping
        #
        # The trigger is deliberately *not* "the residual went up".  A residual that rises for
        # many iterations in a row is the normal behaviour of waveform relaxation on a circuit
        # that oscillates: the iteration is still discovering the limit cycle, and every extra
        # cycle of it makes the residual bigger before it makes it smaller.  Backing off there
        # would throw away the Anderson history exactly when it is the only thing that works.
        # What does mean trouble is a residual that leaves the best one seen far behind, twice
        # running -- that is an Anderson step that overshot.
        if len(residuals) >= 2:
            if residual > backoff_factor * min(residuals):
                bad_steps += 1
            else:
                bad_steps = 0
                if beta < beta_max:
                    beta = min(beta_max, 1.25 * beta)
            if bad_steps >= 2:
                beta = max(beta_min, 0.5 * beta)
                bad_steps = 0
                dx_hist.clear()
                df_hist.clear()
                prev_x = prev_f = None

        # ---------------------------------------------------------------- Anderson history
        if depth > 0:
            if prev_x is not None and prev_f is not None:
                dx_hist.append((x - prev_x).reshape(-1))
                df_hist.append((f - prev_f).reshape(-1))
                while len(df_hist) > depth:
                    dx_hist.pop(0)
                    df_hist.pop(0)
            prev_x, prev_f = x, f

        # ---------------------------------------------------------------- the update itself
        gamma = None
        if df_hist:
            gamma = _anderson_gamma(torch.stack(df_hist, dim=1), f.reshape(-1), reg)

        if gamma is None:
            # Empty history (this is the first update, and it is exactly a Picard step when
            # beta == 1), or a least-squares problem we could not trust.  The undamped case is
            # spelled ``g`` rather than ``x + 1.0 * f`` so that it is the *same* floating point
            # number the original ``param_p = new_param_p`` produced, bit for bit.
            x_next = g if beta == 1.0 else x + beta * f
        else:
            delta_x = torch.stack(dx_hist, dim=1)
            delta_f = torch.stack(df_hist, dim=1)
            correction = (delta_x + beta * delta_f) @ gamma
            x_next = (x + beta * f).reshape(-1) - correction
            x_next = x_next.reshape(x.shape)

        if not bool(torch.isfinite(x_next).all()):
            raise FixedPointDivergence(
                f"The electronic/photonic fixed-point iteration diverged: iterate {iteration + 1} "
                f"contains non-finite entries. Residual history: {_residual_report(residuals)}.",
                _info(False, reference, tolerance))

        x = x_next

    raise FixedPointNotConverged(
        f"The electronic/photonic fixed-point iteration did not converge in {budget} "
        f"iteration(s) (spipe.config['max_iter']): the final residual is {residuals[-1]:.6e}, "
        f"the threshold is {tolerance:.6e}. Residual history: {_residual_report(residuals)}. "
        f"Raise spipe.config['max_iter'], or -- if the residual is not falling -- reduce the "
        f"feedback loop gain or lower spipe.config['damping'].",
        _info(False, reference, tolerance))


def solve_coupling_system(vjp: Callable[[torch.Tensor], torch.Tensor],
                          b: torch.Tensor,
                          config: Mapping,
                          info: Optional[Dict] = None) -> torch.Tensor:
    """Solve ``(I - A^T) lam = b`` where ``A = d(g o f)/dV`` at the converged fixed point.

    This is the whole of E2.3.  The gradient of a coupled electronic/photonic simulation is
    **not** obtained by back-propagating through the fixed-point loop: unrolling costs memory
    proportional to the iteration count, and -- worse -- makes the answer depend on where the
    iteration happened to stop, so two runs that converged to the same waveform to 1e-12 can
    report visibly different gradients.  The implicit function theorem gives the exact
    gradient of the *converged* solution from one linear solve whose size is the number of
    coupling unknowns and whose cost is independent of how many iterations ran.

    :param vjp: ``v -> A^T v``, i.e. one vector-Jacobian product through one full round trip
        (photonic solve then electronic solve) evaluated at the fixed point.  Must accept and
        return tensors shaped like *b*.
    :param b: the incoming cotangent ``dJ/dV``.
    :param config: solver settings, normally :data:`spipe.config`:

        ``coupling_zero_rtol`` (default 1e-10)
            ``||A^T b|| <= coupling_zero_rtol * ||b||`` is treated as no coupling at all.
        ``coupling_dense_limit`` (default 512)
            Largest system built densely.  Above it the matrix-free path runs.
        ``coupling_cond_limit`` (default 1e8)
            Condition number above which :class:`CouplingJacobianError` is raised.
        ``coupling_max_iter`` (default 200), ``coupling_tol`` (default 1e-12)
            Budget and tolerance of the matrix-free path.
        ``coupling_amplification_warn`` (default 1e4)
            ``||(I - A^T)^{-1}||`` above which a warning says the design sits close to the
            stability boundary.  Not an error: a uniformly small ``I - A`` is perfectly
            conditioned and the gradient it gives is correct, just very large.
    :param info: optional dict, filled in with ``'mode'``, ``'size'``, ``'vjp_calls'`` and
        whichever of ``'condition'`` / ``'spectral_radius'`` / ``'amplification'`` the chosen
        path measured.
    :raises CouplingJacobianError: ``I - A^T`` singular, ill conditioned, or (matrix-free) the
        iteration did not converge -- all of which mean the same thing physically.

    Three paths, in order of cost:

    1. **Decoupled.**  One ``vjp`` tells us whether the photocurrent reaches a modulator
       drive at all.  If ``A^T b = 0`` then ``(I - A^T)^{-1} b = b`` *exactly* -- it is an
       identity, not an approximation -- so one product finishes the job.  Every feedback-free
       circuit (which is every example SPIPE ships) lands here, and pays one extra backward
       pass for the whole gradient.
    2. **Dense.**  ``n`` products build ``A^T`` column by column; one SVD and one eigenvalue
       decomposition then give the condition number, the amplification ``1/sigma_min`` and the
       loop gain ``rho(A)`` alongside the answer.  The coupling Jacobian is ``len(time) *
       n_mod`` on a side, which for a circuit small enough to have interesting feedback is a
       few tens to a few hundred -- a dense solve is nothing next to the round trips that
       built it.
    3. **Matrix free.**  Above ``coupling_dense_limit`` the Neumann iteration
       ``lam <- b + A^T lam`` runs instead, converging exactly when the Picard iteration that
       produced the fixed point converged, and its convergence rate *is* an estimate of the
       loop's spectral radius.  Failure to converge is reported as such rather than truncated.
    """
    info = {} if info is None else info
    shape, n = b.shape, b.numel()
    calls = 0

    # The two sides of the loop do not have to agree on a dtype: the photonic solver works in
    # complex128 and hands back float64, while the Xyce driver interpolates its sensitivity
    # block through ``interp1d_warp``, which produces float32.  The cotangent arriving here is
    # therefore float32 on some back ends and float64 on others, and a vector-Jacobian product
    # through the round trip can come back as either.  Everything below is done in the wider
    # of the two -- a coupling matrix is worth solving in double even when the waveforms are
    # not -- and the answer is cast back to the cotangent's own dtype, which is what autograd
    # requires of a Function's backward.
    out_dtype = b.dtype
    solve_dtype = torch.promote_types(out_dtype, torch.float64)

    def product(vector: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        out = vjp(vector)
        if out is None:
            return torch.zeros(shape, dtype=solve_dtype, device=b.device)
        return out.reshape(shape).to(dtype=solve_dtype, device=b.device)

    b = b.to(solve_dtype)
    reference = float(torch.linalg.vector_norm(b.reshape(-1)))
    first = product(b)
    residual = float(torch.linalg.vector_norm(first.reshape(-1)))

    zero_rtol = float(config.get('coupling_zero_rtol', 1e-10))
    if not math.isfinite(residual):
        raise CouplingJacobianError(
            "The electronic/photonic coupling Jacobian evaluated to a non-finite value at the "
            "converged fixed point, so d(output)/d(parameter) cannot be formed.", size=n)
    if residual <= zero_rtol * max(reference, 1e-300):
        info.update(mode='decoupled', size=n, vjp_calls=calls, condition=1.0,
                    spectral_radius=0.0, coupling=residual / max(reference, 1e-300))
        logger.debug("coupling Jacobian: decoupled (||A^T b|| / ||b|| = %.3e)",
                     residual / max(reference, 1e-300))
        return b.to(out_dtype)

    dense_limit = int(config.get('coupling_dense_limit', 512))
    cond_limit = float(config.get('coupling_cond_limit', 1e8))

    if n <= dense_limit:
        basis = torch.zeros(shape, dtype=solve_dtype, device=b.device)
        flat = basis.reshape(-1)
        columns = []
        for index in range(n):
            flat.zero_()
            flat[index] = 1.0
            columns.append(product(basis).reshape(-1).clone())
        transposed = torch.stack(columns, dim=1)              # A^T, column k = A^T e_k
        matrix = torch.eye(n, dtype=solve_dtype, device=b.device) - transposed
        try:
            singular = torch.linalg.svdvals(matrix)
            largest, smallest = float(singular[0]), float(singular[-1])
            condition = largest / smallest if smallest > 0 else float('inf')
            # ||(I - A^T)^{-1}||_2: how much the loop multiplies the open-loop sensitivity.
            # A well-conditioned but uniformly small (I - A) -- every eigenvalue of A near 1 --
            # is not ill posed, but it *is* a loop that amplifies by 1/smallest, and that is
            # worth reporting even when nothing is raised.
            amplification = (1.0 / smallest) if smallest > 0 else float('inf')
        except Exception:                                     # pragma: no cover - LAPACK
            condition = amplification = float('inf')
        radius = None
        if n <= 256:
            try:
                radius = float(torch.linalg.eigvals(transposed).abs().max())
            except Exception:                                 # pragma: no cover - LAPACK
                radius = None
        if not math.isfinite(condition) or condition > cond_limit:
            raise CouplingJacobianError(
                f"The electronic/photonic coupling matrix (I - g'f') has condition number "
                f"{condition:.3e} at the converged fixed point ({n} unknowns), above the "
                f"limit spipe.config['coupling_cond_limit'] = {cond_limit:g}. The loop gain "
                f"of this circuit is at or beyond 1, so the fixed point is only marginally "
                f"stable and d(output)/d(parameter) is unbounded there -- the number SPIPE "
                f"would return is meaningless. Reduce the feedback loop gain (detector "
                f"responsivity, trans-impedance, modulator sensitivity) or study the circuit "
                f"as a dynamical system rather than as a fixed point.",
                condition=condition, spectral_radius=radius, size=n)
        solution = torch.linalg.solve(matrix, b.reshape(-1)).reshape(shape)
        if not bool(torch.isfinite(solution).all()):
            raise CouplingJacobianError(
                f"Solving the {n}x{n} coupling system (I - g'f') produced non-finite values; "
                f"the fixed point is not differentiable there.", condition=condition,
                spectral_radius=radius, size=n)
        if amplification > float(config.get('coupling_amplification_warn', 1e4)):
            warnings.warn(
                f"The electronic/photonic feedback loop amplifies the open-loop sensitivity by "
                f"{amplification:.3e} (||(I - g'f')^-1||, loop gain "
                f"{radius if radius is not None else float('nan'):.6f}). The gradient is "
                f"well defined but the design sits close to the stability boundary, so it is "
                f"very sensitive to anything that moves the loop gain.",
                RuntimeWarning, stacklevel=3)
        info.update(mode='dense', size=n, vjp_calls=calls, condition=condition,
                    spectral_radius=radius, amplification=amplification)
        logger.debug("coupling Jacobian: dense %dx%d, cond=%.3e, rho=%s", n, n, condition,
                     radius)
        return solution.to(out_dtype)

    # Matrix free: lam <- b + A^T lam.
    budget = int(config.get('coupling_max_iter', 200))
    tol = float(config.get('coupling_tol', 1e-12))
    lam = b + first
    previous_delta = residual
    ratios: List[float] = []
    for iteration in range(budget):
        nxt = b + product(lam)
        delta = float(torch.linalg.vector_norm((nxt - lam).reshape(-1)))
        lam = nxt
        if previous_delta > 0:
            ratios.append(delta / previous_delta)
        previous_delta = delta
        if not math.isfinite(delta):
            break
        if delta <= tol * max(float(torch.linalg.vector_norm(lam.reshape(-1))), 1e-300):
            radius = ratios[-1] if ratios else 0.0
            info.update(mode='neumann', size=n, vjp_calls=calls, condition=None,
                        spectral_radius=radius, iters=iteration + 1)
            logger.debug("coupling Jacobian: Neumann converged in %d products, rho~%.3e",
                         calls, radius)
            return lam.to(out_dtype)
    radius = max(ratios[-3:]) if ratios else float('inf')
    raise CouplingJacobianError(
        f"The coupling system (I - g'f') could not be solved: the Neumann iteration did not "
        f"converge in {budget} products on {n} unknowns, and the observed contraction factor "
        f"is {radius:.4f}. A factor at or above 1 is a feedback loop with gain >= 1, i.e. a "
        f"fixed point that is not stable, and the derivative of the coupled solution does not "
        f"exist there. (Below spipe.config['coupling_dense_limit'] SPIPE forms the matrix and "
        f"reports its condition number instead; this system is larger than that, so reduce "
        f"the number of time samples if you want the dense diagnosis.)",
        spectral_radius=radius, size=n)


class _ImplicitFixedPoint(torch.autograd.Function):
    """Identity forward; backward replaces the cotangent by the IFT-corrected one.

    ``forward`` returns the one-step image ``V_new = g(f(V*))`` unchanged -- numerically it
    *is* the fixed point.  ``backward`` receives ``dJ/dV`` and hands back
    ``(I - A^T)^{-1} dJ/dV``, which autograd then propagates into ``V_new``'s own graph.
    Because that graph was built from a **detached** ``V*``, the only route left open is the
    one through the electronic device parameters, so what accumulates on them is

        dJ/dtheta = lam^T (partial g / partial theta)

    -- the implicit function theorem, and no loop was unrolled to get it.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, solver: Callable):
        ctx.solver = solver
        return value.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.solver(grad_output), None


class Circuit(object):
    def __init__(self,
                 file_path: str,
                 spice_exe: str,
                 power_node: Optional[List] = None,
                 spice_wrk_dir: Optional[str] = './tmp',
                 spice_file_name: Optional[str] = 'tmp.sp',
                 keep_spice: Optional[bool] = True,
                 need_grads: Optional[bool] = False,
                 use_adjoint: Optional[bool] = False
                 ) -> None:
        self.file_path = file_path
        self.spice_wrk_dir = spice_wrk_dir
        self.spice_file_name = spice_file_name

        self.e_content, self.p_content, self.time = _parse_file(file_path)
        self.e_circuit = Electronic(self.e_content,
                                    self.p_content,
                                    spice_exe,
                                    len(self.time),
                                    power_node,
                                    use_adjoint=use_adjoint,
                                    spice_wrk_dir=spice_wrk_dir,
                                    spice_file_name=spice_file_name,
                                    keep_spice=keep_spice)
        self.p_circuit = Photonic(self.p_content, need_grads = need_grads)

        #: What :meth:`differentiable_simulate` measured about the electronic/photonic
        #: coupling Jacobian.  Filled in when the backward pass runs, because that is when the
        #: system is actually solved; see :func:`solve_coupling_system` for the keys.
        self.coupling_info: Dict[str, Any] = {}

        # X8: the steady-state photonic solve is only valid while the optical network settles
        # much faster than one transient time step.  Warn (never raise) if that is not the case.
        if self.time is not None and len(self.time) > 1:
            self.p_circuit.check_quasistatic(float(self.time[1] - self.time[0]))

    def simulate(self,
                 seed: Optional[int] = None,
                 x0: Optional[torch.Tensor] = None,
                 mode: str = 'quasistatic',
                 differentiable: Optional[bool] = None):
        """Run the coupled electronic/photonic fixed-point simulation.

        **This is the only simulation entry point you need.**  It follows the usual PyTorch
        convention: gradients cost nothing unless you ask for them.  If some ``.sensparam``
        device parameter has ``requires_grad=True``, the returned tensors carry a usable
        autograd graph all the way back to it; otherwise the cheaper gradient-free path runs
        and the result is identical.

        ::

            ckt = Circuit('link.sp', spice_exe='native')

            _, _, photocurrent, drive, _ = ckt.simulate()      # plain run, no graph

            w = ckt.param('mn1', 'W')                          # requires_grad already True
            _, _, photocurrent, drive, _ = ckt.simulate()      # now differentiable
            (photocurrent[:, 0] ** 2).sum().backward()
            w.grad                                             # d(optical) / dW

        :param seed: overrides ``config['seed']`` for this run.  The seed is applied to a private
            ``torch.Generator``, so the caller's global RNG state is left untouched, and the same
            seed always reproduces the same result bit for bit.
        :param x0: an explicit initial guess for the modulator drive voltages, shape
            ``(len(time), number of active photonic devices)``; a scalar or a 1-D row is
            broadcast.  ``None`` (the default) keeps the historical random start.  A circuit
            with *multiple* stable fixed points -- an optical latch, say -- has genuinely more
            than one answer, and which one the solver reaches is a property of the initial
            guess, not of the circuit; this is how you choose.
        :param mode: photonic solver mode, ``'quasistatic'`` (default) or ``'envelope'``.
            Only consulted on the differentiable path; ``'envelope'`` does not carry gradients.
        :param differentiable: force the choice instead of detecting it.  ``True`` always
            builds the graph (and warns if nothing can receive a gradient), ``False`` never
            does.  ``None``, the default, decides from whether any declared parameter
            requires grad and whether grad is enabled at all.
        """
        if differentiable is None:
            differentiable = (
                torch.is_grad_enabled()
                and bool(self.e_circuit.sens_declared)
                and any(self.e_circuit.sens_values[(d.lower(), n)].requires_grad
                        for d, n in self.e_circuit.sens_declared)
            )
        if differentiable:
            return self.differentiable_simulate(self.time, seed=seed, x0=x0, mode=mode)
        return self.gradient_free_simulate(self.time, seed=seed, x0=x0)

    def param(self, device: str, name: str) -> torch.Tensor:
        """The float64 leaf tensor holding one ``.sensparam`` electronic device parameter.

        Shorthand for ``circuit.e_circuit.param(device, name)``; see
        :meth:`spipe.electronic.electronic.Electronic.param`.
        """
        return self.e_circuit.param(device, name)

    def gradient_free_simulate(self, t: torch.Tensor,
                               seed: Optional[int] = None,
                               x0: Optional[torch.Tensor] = None
                               ) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor, Dict]:
        shape = (len(t), len(self.p_circuit.mod_element.keys()))

        if x0 is None:
            # X3: seed a *local* generator so the initial guess is reproducible without
            # clobbering the caller's global torch RNG state.
            seed = config['seed'] if seed is None else seed
            generator = torch.Generator(device=config['device'])
            generator.manual_seed(int(seed))
            param_p = torch.randn(*shape, generator=generator, device=config['device'],
                                  dtype=config['real_dtype'])
        else:
            param_p = torch.as_tensor(x0, device=config['device'],
                                      dtype=config['real_dtype']).expand(shape).clone()

        # Both branches pin the dtype. solve_fixed_point() casts every iterate to the dtype of
        # the initial guess, and torch.randn / torch.as_tensor(float) default to float32 -- so
        # without this the whole co-simulation fixed point ran in single precision even with
        # real_dtype = float64. The converged drive came back float32, and the differentiable
        # and plain paths disagreed by 9.3e-08 at ANY convergence tolerance (float32 rounding,
        # not residual). With it they agree to 5.6e-16.

        # X2: scale-free convergence test, and everything else about the iteration, now lives in
        # solve_fixed_point() -- a SPICE-free, independently testable function.  This method is a
        # thin wrapper over it so the two cannot drift apart: `step` is one full round trip
        # (photonic solve, then electronic solve) and `state` keeps whatever that round trip
        # produced besides the fixed-point variable itself.
        state: Dict[str, Any] = {}

        def step(current_param_p: torch.Tensor) -> torch.Tensor:
            s1 = time.time()
            param_e, result_middle_p, power_p = self.p_circuit.simulate(t, current_param_p)
            s2 = time.time()
            new_param_p, result_middle_e, power_e = self.e_circuit.simulate(t, param_e)
            s3 = time.time()
            logger.debug("photonic simulate=%f s, electronic simulate=%f s", s2 - s1, s3 - s2)
            state.update(param_e=param_e,
                         result_middle_p=result_middle_p, power_p=power_p,
                         result_middle_e=result_middle_e, power_e=power_e)
            return new_param_p

        # solve_fixed_point returns the iterate that *satisfied* the criterion, i.e. the very
        # `current_param_p` the last `step` call was given, so `state` describes exactly the
        # returned `param_p` -- as it did in the original loop.
        param_p, self.fixed_point_info = solve_fixed_point(step, param_p, config)

        power_e, power_p = state['power_e'], state['power_p']
        self.power_report = {**power_e, **power_p}
        return (state['result_middle_e'], state['result_middle_p'], state['param_e'], param_p,
                {**power_e, 'photonic': power_p['photonic']})


    def get_power(self) -> Dict:
        """The power report from the most recent :meth:`simulate` call.

        Same dict :meth:`simulate` returns as its fifth element -- the electronic power
        nodes plus ``'photonic'``, the laser wall-plug power. Kept as a method because it
        is convenient after a call whose return value you did not keep.

        :raises RuntimeError: no simulation has run yet, so there is nothing to report.
            (This used to be a stub that returned ``None`` unconditionally, which looked
            like "this circuit draws no power".)
        """
        if getattr(self, 'power_report', None) is None:
            raise RuntimeError(
                "Circuit.get_power() has nothing to report: call Circuit.simulate() first. "
                "The power report is built during the simulation, not from the netlist.")
        return self.power_report

    # ----------------------------------------------------------------- differentiable path
    def differentiable_simulate(self,
                                t: Optional[torch.Tensor] = None,
                                seed: Optional[int] = None,
                                x0: Optional[torch.Tensor] = None,
                                mode: str = 'quasistatic'
                                ) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor, Dict]:
        """:meth:`simulate`, but every returned tensor carries a usable autograd graph.

        This is the end-to-end link SPIPE was missing: the returned photocurrent (and hence
        any optical figure of merit built from it) is differentiable with respect to the
        **electronic device parameters** declared with ``.sensparam``, through the electronic
        solve, the photonic solve *and* the fixed point that couples them::

            circuit = Circuit('link.sp', spice_exe='native')
            w = circuit.param('mn1', 'W')                  # a float64 leaf, requires_grad
            _, _, photocurrent, drive, _ = circuit.differentiable_simulate()
            (photocurrent[:, 0].sum()).backward()
            w.grad                                         # d(optical output) / dW  [A/m]

        Returns the same ``(result_middle_e, result_middle_p, param_e, param_p, power)``
        five-tuple :meth:`gradient_free_simulate` returns, and the *values* are the same to
        solver tolerance -- only the graph is added.

        How it works
        ------------
        1. The fixed point is solved exactly as before, gradient-free, giving ``V*``.
        2. One round trip is rebuilt at ``V*`` from a **detached** copy, which exposes both
           the coupling Jacobian ``A = d(g o f)/dV`` and the partial ``dg/dtheta``.
        3. :class:`_ImplicitFixedPoint` splices the implicit function theorem into the
           backward pass, so a cotangent arriving at the drive voltages is replaced by
           ``(I - A^T)^{-1}`` times itself before it reaches the device parameters.  The loop
           is *not* unrolled: the cost and the answer are both independent of how many
           iterations the forward solve took.
        4. A second *photonic* solve, from the IFT-corrected drive, gives the photocurrent
           ``param_e`` and the photonic probes ``result_middle_p`` a correct **total**
           derivative, ``d(optical)/dtheta = f'(V*) (I - A)^{-1} dg/dtheta``.

        The extra cost over :meth:`simulate` is one round trip, one further photonic solve and
        the ``vjp`` calls :func:`solve_coupling_system` needs (one, for a feedback-free
        circuit).  ``circuit.coupling_info`` reports which path it took and what it measured.

        What is *not* differentiable here
        ---------------------------------
        ``result_middle_e`` and the electrical entries of ``power`` -- the ``.print tran``
        probes of the electronic deck -- are returned with the partial derivative
        ``partial(probe)/partial(theta)`` **at frozen photocurrent**, not the total one.
        Closing that last loop would need a second *electronic* solve driven by the corrected
        photocurrent, and the device parameters would then be reachable by two different
        routes through one autograd graph, which double counts.  The optical outputs, which
        are what this method is for, are total derivatives.  On a feedback-free circuit --
        every example SPIPE ships -- the distinction only affects probes downstream of a
        detector, never the modulator drive.

        :param mode: photonic propagation mode.  ``'envelope'`` is rejected: that path
            integrates an impulse response with no backward, so a gradient taken through it
            would silently be the quasi-static one.
        :raises CouplingJacobianError: the coupled fixed point is not differentiable, i.e. the
            feedback loop gain is at or above one.  See that exception.
        :raises NotImplementedError: the electronic back end cannot supply the gradients this
            needs (HSPICE cannot differentiate with respect to the photocurrent).
        """
        if mode != 'quasistatic':
            raise ValueError(
                f"differentiable_simulate(mode={mode!r}) is not available: the envelope "
                f"propagation mode (spipe.photonic.envelope) has no backward pass, so a "
                f"gradient taken through it would silently be the quasi-static gradient of a "
                f"different simulation. Use mode='quasistatic'.")

        t = self.time if t is None else t
        if not self.e_circuit.sens_declared:
            warnings.warn(
                "differentiable_simulate() was called on a netlist that declares no "
                "electronic device parameters, so nothing will receive a gradient. Add a "
                "'.sensparam DEVICE:PARAM' card to the .electronic section, e.g. "
                "'.sensparam M1:W'.", RuntimeWarning, stacklevel=2)

        # 1. the ordinary, gradient-free fixed point.
        self.gradient_free_simulate(t, seed=seed, x0=x0)
        v_star = self.fixed_point_info['x'].detach()

        electronic, photonic = self.e_circuit, self.p_circuit
        coupling_info: Dict[str, Any] = {}

        # The photonic solver only keeps the factorisation its backward needs when asked to,
        # and this method is exactly the place that asks.  The gradient-free fixed point above
        # already ran without it, so the iterations that do not need it do not pay for it.
        previous_need_grads = photonic.need_grads
        photonic.need_grads = True
        try:
            with electronic.enable_gradients():
                # 2. one round trip from a detached V*, which is the fixed point itself.
                v_leaf = v_star.clone().requires_grad_(True)
                probe_current, _, _ = photonic.simulate(t, v_leaf, mode=mode)

                # A back end that cannot differentiate the drive with respect to the
                # photocurrent (HSPICE) can still be used -- but only once it has been
                # *measured* that the photocurrent does not reach a modulator, which makes the
                # coupling Jacobian exactly zero and the IFT solve the identity.  The
                # photocurrent is then detached, so nothing later asks for the derivative the
                # back end does not have.
                measurable = electronic.supports_photocurrent_gradient
                current_for_electronic = probe_current if measurable else probe_current.detach()
                v_next, result_middle_e, power_e = electronic.simulate(t, current_for_electronic)
                captured = electronic.capture_state()

                if not measurable:
                    decoupled, change = electronic.probe_decoupling(
                        t, current_for_electronic, v_next)
                    if not decoupled:
                        raise NotImplementedError(
                            f"This circuit has optical-to-electrical feedback (the modulator "
                            f"drive responded to a photocurrent perturbation, by a relative "
                            f"{change:.3e}, and the response scaled with the perturbation), "
                            f"so its coupled gradient needs "
                            f"d(drive)/d(photocurrent) -- and the "
                            f"{electronic.backend!r} back end cannot supply it: HSPICE has no "
                            f"transient parameter sensitivity analysis. Use "
                            f"spice_exe='native' (SPIPE's own differentiable engine, exact "
                            f"gradients in one adjoint sweep) or a Xyce command line (.SENS, "
                            f"direct method). Device-parameter gradients alone, without the "
                            f"feedback loop, are available on HSPICE by finite differences. "
                            f"If this circuit *is* feedback free, what the probe measured is "
                            f"the back end's own reproducibility: pin the transient grid and "
                            f"the print precision ('.option numdgt=7 delmax=... relv=... "
                            f"absv=...'), because an adaptive time step makes the answer move "
                            f"when the photocurrent does. "
                            f"spipe.config['coupling_probe_rtol'] (currently "
                            f"{float(config.get('coupling_probe_rtol', 1e-9)):g}) is the "
                            f"threshold.")
                    coupling_info.update(mode='decoupled-probe', size=v_next.numel(),
                                         vjp_calls=0, condition=1.0, spectral_radius=0.0,
                                         coupling=change)

                def vjp(vector: torch.Tensor) -> torch.Tensor:
                    # The photonic solve below overwrites the back end's per-run state, so the
                    # state this graph was built with is put back for the product.
                    with electronic.restored_state(captured):
                        gradient, = torch.autograd.grad(
                            v_next, v_leaf,
                            grad_outputs=vector.to(dtype=v_next.dtype, device=v_next.device),
                            retain_graph=True, allow_unused=True)
                    return gradient

                def solver(cotangent: torch.Tensor) -> torch.Tensor:
                    if not measurable:
                        return cotangent            # measured above: A is exactly zero
                    return solve_coupling_system(vjp, cotangent, config, coupling_info)

                # 3. splice in the implicit function theorem.
                v_fixed = _ImplicitFixedPoint.apply(v_next, solver)

                # 4. one more *photonic* solve, so that the optical output -- the thing this
                #    whole method exists for -- carries the total derivative
                #    d(optical)/dtheta = f'(V*) (I - A)^{-1} dg/dtheta and not a partial one.
                param_e, result_middle_p, power_p = photonic.simulate(t, v_fixed, mode=mode)
        finally:
            photonic.need_grads = previous_need_grads

        self.coupling_info = coupling_info
        self.power_report = {**power_e, **power_p}
        return (result_middle_e, result_middle_p, param_e, v_fixed,
                {**power_e, 'photonic': power_p['photonic']})

    def gradient_based_simulate(self,
                                t: Optional[torch.Tensor] = None,
                                seed: Optional[int] = None,
                                x0: Optional[torch.Tensor] = None,
                                lr: Optional[float] = None,
                                max_iter: Optional[int] = None
                                ) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor, Dict]:
        """Solve the coupled system by *minimising the round-trip residual* (paper Eq. 13).

        An alternative to :meth:`gradient_free_simulate`'s Picard/Anderson iteration for the
        same equation ``V = g(f(V))``: instead of iterating the map, descend its residual

            L(V) = || V - g(f(V)) ||^2 / N

        with Adam.  Where the Picard map is a contraction the fixed-point solver is far
        cheaper and should be preferred; this one exists for the case the paper's Eq. (13)
        is about, a loop gain near or above one, where iterating the map does not converge but
        the residual still has a minimum.  It needs a back end that can differentiate the
        electronic solve with respect to the photocurrent, i.e. ``'native'`` or Xyce -- HSPICE
        raises :class:`NotImplementedError` here, as it does everywhere it cannot deliver.

        :param lr: Adam learning rate; defaults to ``config['lr']``.
        :param max_iter: iteration budget; defaults to ``config['max_iter']``.
        :returns: the same five-tuple as :meth:`gradient_free_simulate`.
        :raises FixedPointNotConverged: the residual never met the same mixed
            relative/absolute criterion :func:`solve_fixed_point` uses, carrying the full
            residual history.  As everywhere else in SPIPE, the last iterate of a run that did
            not converge is *not* returned.
        """
        t = self.time if t is None else t
        shape = (len(t), len(self.p_circuit.mod_element.keys()))
        if x0 is None:
            seed = config['seed'] if seed is None else seed
            generator = torch.Generator(device=config['device'])
            generator.manual_seed(int(seed))
            start = torch.randn(*shape, generator=generator, device=config['device'])
        else:
            start = torch.as_tensor(x0, device=config['device']).expand(shape).clone()

        variable = start.detach().clone().to(config['real_dtype']).requires_grad_(True)
        optimizer = torch.optim.Adam([variable], lr=config['lr'] if lr is None else lr)
        budget = max(1, int(config['max_iter'] if max_iter is None else max_iter))

        rtol, atol = float(config['rtol']), float(config['atol'])
        scale = max(1, variable.numel()) ** 0.5
        residuals: List[float] = []
        state: Dict[str, Any] = {}

        previous_need_grads = self.p_circuit.need_grads
        self.p_circuit.need_grads = True
        try:
            return self._descend_residual(t, variable, optimizer, budget, rtol, atol, scale,
                                          residuals, state)
        finally:
            self.p_circuit.need_grads = previous_need_grads

    def _descend_residual(self, t, variable, optimizer, budget, rtol, atol, scale,
                          residuals, state):
        with self.e_circuit.enable_gradients():
            for iteration in range(budget):
                optimizer.zero_grad()
                param_e, result_middle_p, power_p = self.p_circuit.simulate(t, variable)
                image, result_middle_e, power_e = self.e_circuit.simulate(t, param_e)
                difference = image - variable
                residual = float(torch.linalg.vector_norm(difference.detach().reshape(-1))) / scale
                reference = float(torch.linalg.vector_norm(variable.detach().reshape(-1))) / scale
                residuals.append(residual)
                state.update(param_e=param_e, result_middle_p=result_middle_p, power_p=power_p,
                             result_middle_e=result_middle_e, power_e=power_e)
                logger.debug("gradient based iter %d: rms residual=%.6e", iteration, residual)
                if residual <= rtol * reference + atol:
                    self.fixed_point_info = {'iters': len(residuals),
                                             'residuals': list(residuals),
                                             'converged': True, 'reference': reference,
                                             'tolerance': rtol * reference + atol,
                                             'method': 'gradient', 'x': variable.detach()}
                    power_e, power_p = state['power_e'], state['power_p']
                    self.power_report = {**power_e, **power_p}
                    return (state['result_middle_e'], state['result_middle_p'],
                            state['param_e'], variable.detach(),
                            {**power_e, 'photonic': power_p['photonic']})
                loss = (difference ** 2).mean()
                loss.backward()
                optimizer.step()

        info = {'iters': len(residuals), 'residuals': list(residuals), 'converged': False,
                'method': 'gradient', 'x': variable.detach()}
        raise FixedPointNotConverged(
            f"gradient_based_simulate() did not converge in {budget} iteration(s): the final "
            f"round-trip residual is {residuals[-1]:.6e}. Residual history: "
            f"{_residual_report(residuals)}. Raise spipe.config['max_iter'], adjust "
            f"spipe.config['lr'] (currently {config['lr']:g}), or -- if the circuit is "
            f"feedback free -- use gradient_free_simulate(), which solves it in two "
            f"iterations.", info)

