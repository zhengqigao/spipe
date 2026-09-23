"""TB-07 : the electronic-photonic fixed-point solver (SPEC-L.1 / L.1a).

Every check drives `solve_fixed_point` with a pure-Python `step`, so convergence behaviour
is tested with NO simulator involved and the contraction factor is known exactly.
That makes each property checkable in closed form.
Written from SPEC_L.md only.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch


def build():
    tb = TB('TB-07  fixed-point solver (convergence, feedback, divergence)')
    sp = fresh_spipe()

    try:
        from spipe.core.core import solve_fixed_point
    except ImportError as e:
        tb.ok('L1a.solve_fixed_point_exported', False,
              f"spipe.core.core.solve_fixed_point is required by SPEC-L.1a: {e}")
        return tb
    tb.ok('L1a.solve_fixed_point_exported', True, '')

    cfg = dict(sp.config)
    cfg['atol'], cfg['rtol'], cfg['max_iter'] = 1e-10, 1e-10, 200

    # ---------- 1. feedback-free: converges in EXACTLY 2 iterations --------
    # a circuit with no feedback has step(x) = const, independent of x
    target = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    calls = {'n': 0}

    def const_step(x):
        calls['n'] += 1
        return target.clone()

    ok, got = tb.no_raise('L1.nofeedback_runs',
                          lambda: solve_fixed_point(const_step, torch.zeros(3, dtype=torch.float64), cfg))
    if ok and got is not None:
        x, info = got
        tb.lt('L1.nofeedback_accurate', float((x - target).abs().max()), 1e-12,
              f"converged to {x.tolist()}")
        tb.ok('L1.nofeedback_two_iters', info.get('iters') == 2,
              f"iters={info.get('iters')} -- a feedback-free circuit must take exactly 2 "
              f"(this is what makes the shipped examples cheap)")
        tb.ok('L1.info_has_residuals', isinstance(info.get('residuals'), (list, tuple)),
              f"info keys = {sorted(info)}")
        tb.ok('L1.info_converged_flag', info.get('converged') is True, '')

    # ---------- 2. contraction factors, including near unity --------------
    # step(x) = k*x + b  has the unique fixed point b/(1-k) for |k| < 1
    for k in (0.3, 0.8, 0.95, 0.99, -0.9):
        b = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
        exact = b / (1.0 - k)

        def lin(x, k=k, b=b):
            return k * x + b

        ok, got = tb.no_raise(f'L1.contract_k{k}_runs',
                              lambda lin=lin: solve_fixed_point(
                                  lin, torch.zeros(3, dtype=torch.float64), cfg))
        if ok and got is not None:
            x, info = got
            err = float((x - exact).abs().max() / exact.abs().max())
            tb.lt(f'L1.contract_k{k}', err, 1e-9,
                  f"k={k}: exact={exact.tolist()}, iters={info.get('iters')}")

    # ---------- 3. scale-freeness: same k, 1e6x bigger signal -------------
    k = 0.9
    for scale in (1.0, 1e6):
        b = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64) * scale
        exact = b / (1.0 - k)
        x, info = solve_fixed_point(lambda x, k=k, b=b: k * x + b,
                                    torch.zeros(3, dtype=torch.float64), cfg)
        err = float((x - exact).abs().max() / exact.abs().max())
        tb.lt(f'L1.scale_free_{scale:g}', err, 1e-9,
              f"iters={info.get('iters')} at signal scale {scale:g}")

    # ---------- 4. DIVERGENCE MUST RAISE, never return quietly ------------
    # NB: step(x) = 1.5x + 1 is NOT a valid divergence test -- it has a perfectly
    # good fixed point at x = -2 (1.5*-2+1 = -2), merely a REPELLING one. A solver
    # that root-finds (Anderson/Newton) rather than iterating Picard will find it,
    # and should. Use a map with genuinely NO fixed point instead.
    def diverging(x):
        return x + 1.0            # x = x + 1 has no solution

    raised = {'v': False, 'msg': ''}
    try:
        x, info = solve_fixed_point(diverging, torch.ones(3, dtype=torch.float64), cfg)
        # returning is only acceptable if it clearly says it did NOT converge
        quiet = bool(info.get('converged', True))
        tb.ok('L1.divergence_detected', not quiet,
              f"returned without raising; info={ {k: v for k, v in info.items() if k != 'residuals'} }")
    except Exception as e:
        raised['v'], raised['msg'] = True, str(e)
        tb.ok('L1.divergence_detected', True, f"raised {type(e).__name__}: {str(e)[:120]}")
    if raised['v']:
        tb.ok('L1.divergence_msg_actionable',
              any(w in raised['msg'].lower() for w in
                  ('converge', 'diverge', 'residual', 'iter')),
              f"message must name the failure and ideally the residual history: "
              f"{raised['msg'][:160]!r}")

    # second no-fixed-point case: x = x^2 + 2 has discriminant 1 - 8 < 0
    def nofix2(x):
        return x ** 2 + 2.0
    try:
        x, info = solve_fixed_point(nofix2, torch.zeros(2, dtype=torch.float64), cfg)
        tb.ok('L1.divergence_detected_nonlinear', not bool(info.get('converged', True)),
              f"x^2+2 has no real fixed point; returned converged={info.get('converged')} "
              f"x={x.tolist()}")
    except Exception as e:
        tb.ok('L1.divergence_detected_nonlinear', True,
              f"raised {type(e).__name__}: {str(e)[:110]}")

    # a REPELLING but real fixed point must still be found (this is a feature)
    x, info = solve_fixed_point(lambda x: 1.5 * x + 1.0,
                                torch.zeros(3, dtype=torch.float64), cfg)
    tb.close('L1.repelling_fixed_point_found', float(x[0]), -2.0, 1e-9,
             detail='step(x)=1.5x+1 has a repelling fixed point at -2; a root-finding '
                    'solver should converge to it even though Picard diverges')

    # ---------- 5. determinism -------------------------------------------
    def noisyish(x):
        return 0.5 * x + torch.tensor([0.3, 0.1, -0.2], dtype=torch.float64)
    a, _ = solve_fixed_point(noisyish, torch.zeros(3, dtype=torch.float64), cfg)
    b2, _ = solve_fixed_point(noisyish, torch.zeros(3, dtype=torch.float64), cfg)
    tb.ok('L1.deterministic', bool(torch.equal(a, b2)),
          'two identical calls must give bitwise identical results')

    # ---------- 6. BISTABILITY: both states must be reachable -------------
    # step(x) = tanh(3x) has three fixed points: -a, 0, +a, with +/-a stable.
    A = 0.9949015284526292      # solves a = tanh(3a); verified with brentq
    outs = {}
    for x0 in (-1.0, -0.2, 0.2, 1.0):
        try:
            x, info = solve_fixed_point(lambda x: torch.tanh(3.0 * x),
                                        torch.tensor([x0], dtype=torch.float64), cfg)
            outs[x0] = float(x[0])
        except Exception as e:
            outs[x0] = float('nan')
    pos = [v for v in outs.values() if v > 0.5]
    neg = [v for v in outs.values() if v < -0.5]
    tb.ok('L1.bistable_both_states_reachable', len(pos) > 0 and len(neg) > 0,
          f"x0 -> x*: { {k: round(v, 6) for k, v in outs.items()} }; "
          f"both +{A:.4f} and -{A:.4f} must be reachable")
    for x0, v in outs.items():
        if not math.isnan(v) and abs(v) > 0.5:
            tb.close(f'L1.bistable_value_x0_{x0}', abs(v), A, 1e-6,
                     detail='converged state must be an actual fixed point of tanh(3x)')

    # ---------- 7. max_iter is honoured ------------------------------------
    cfg2 = dict(cfg); cfg2['max_iter'] = 3
    hits = {'n': 0}

    def slow(x):
        hits['n'] += 1
        return 0.999 * x + 1.0
    try:
        x, info = solve_fixed_point(slow, torch.zeros(1, dtype=torch.float64), cfg2)
        tb.ok('L1.max_iter_honoured', hits['n'] <= 4,
              f"step called {hits['n']} times with max_iter=3")
    except Exception as e:
        tb.ok('L1.max_iter_honoured', hits['n'] <= 4,
              f"step called {hits['n']} times with max_iter=3, then raised "
              f"{type(e).__name__} (acceptable)")

    return tb


if __name__ == '__main__':
    main(build)
