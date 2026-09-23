"""Minimal test harness for the SPIPE acceptance suite.

Deliberately dependency-free (no pytest) so the supervisor can run it anywhere.
Every check records a PASS/FAIL with the measured number, so a failure report is
actionable without re-running.
"""
import sys, os, time, traceback, json

# Repo root: the parent of test/. Override with SPIPE_REPO to test another checkout.
REPO = os.environ.get('SPIPE_REPO',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_SRC = os.path.join(REPO, 'src')
if os.path.isdir(os.path.join(_SRC, 'spipe')) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class TB:
    def __init__(self, name):
        self.name = name
        self.checks = []          # (id, ok, detail)
        self._t0 = time.time()

    # ---- assertions -------------------------------------------------------
    def ok(self, cid, cond, detail=''):
        self.checks.append((cid, bool(cond), str(detail)))
        return bool(cond)

    def close(self, cid, got, want, tol, rel=True, detail=''):
        """Numeric closeness check; records the actual error either way."""
        try:
            g, w = float(got), float(want)
            err = abs(g - w) / max(abs(w), 1e-300) if rel else abs(g - w)
            kind = 'rel' if rel else 'abs'
            return self.ok(cid, err <= tol,
                           f"got={g:.10g} want={w:.10g} {kind}err={err:.3e} tol={tol:.3e} {detail}")
        except Exception as e:
            return self.ok(cid, False, f"comparison failed: {e!r} {detail}")

    def lt(self, cid, got, bound, detail=''):
        try:
            g = float(got)
            return self.ok(cid, g < float(bound), f"got={g:.6g} bound={float(bound):.6g} {detail}")
        except Exception as e:
            return self.ok(cid, False, f"comparison failed: {e!r} {detail}")

    def raises(self, cid, fn, exc=Exception, detail=''):
        try:
            fn()
        except exc as e:
            return self.ok(cid, True, f"raised {type(e).__name__}: {str(e)[:120]} {detail}")
        except Exception as e:
            return self.ok(cid, False, f"raised WRONG type {type(e).__name__}: {str(e)[:120]} {detail}")
        return self.ok(cid, False, f"did not raise {getattr(exc,'__name__',exc)} {detail}")

    def no_raise(self, cid, fn, detail=''):
        try:
            r = fn()
            return self.ok(cid, True, f"ok {detail}"), r
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()[-1]
            return self.ok(cid, False, f"raised {tb} {detail}"), None

    def skip(self, cid, why):
        self.checks.append((cid, None, f"SKIPPED: {why}"))

    # ---- reporting --------------------------------------------------------
    def report(self):
        dt = time.time() - self._t0
        npass = sum(1 for _, o, _ in self.checks if o is True)
        nfail = sum(1 for _, o, _ in self.checks if o is False)
        nskip = sum(1 for _, o, _ in self.checks if o is None)
        print(f"\n{'='*78}\n{self.name}   {npass} pass / {nfail} FAIL / {nskip} skip   ({dt:.1f}s)\n{'='*78}")
        for cid, o, d in self.checks:
            tag = 'PASS' if o is True else ('FAIL' if o is False else 'SKIP')
            print(f"  [{tag}] {cid:<44} {d}")
        return {'name': self.name, 'pass': npass, 'fail': nfail, 'skip': nskip,
                'checks': [{'id': c, 'ok': o, 'detail': d} for c, o, d in self.checks]}


def fresh_spipe(**cfg):
    """Import spipe fresh (clearing cached modules) with optional config overrides.

    Needed because spipe caches model.json and dtype config at import time.
    """
    for m in [k for k in list(sys.modules) if k == 'spipe' or k.startswith('spipe.')]:
        del sys.modules[m]
    _src = os.path.join(REPO, 'src')
    _root = _src if os.path.isdir(os.path.join(_src, 'spipe')) else REPO
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import spipe
    for k, v in cfg.items():
        spipe.config[k] = v
    return spipe


def main(tb_fn):
    tb = tb_fn()
    res = tb.report()
    out = os.environ.get('TB_JSON')
    if out:
        with open(out, 'a') as f:
            f.write(json.dumps(res) + '\n')
    sys.exit(1 if res['fail'] else 0)
