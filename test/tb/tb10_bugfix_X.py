"""TB-10 : acceptance suite for SPEC-X (correctness fixes and hygiene).

One check group per spec item. Every one of these FAILS against the pre-fix code,
which is what makes them meaningful.
"""
import sys, os, math, warnings, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO

import torch



def _pkg(*parts):
    """Path to a file inside the spipe package, in either layout (src/ or flat)."""
    for base in (os.path.join(REPO, 'src'), REPO):
        cand = os.path.join(base, 'spipe', *parts)
        if os.path.exists(cand) or os.path.isdir(os.path.dirname(cand)):
            return cand
    return os.path.join(REPO, 'src', 'spipe', *parts)


def _repo_file(*parts):
    """Path to a non-package file, tolerating the examples/ relocation."""
    alts = [os.path.join(REPO, *parts),
            os.path.join(REPO, 'examples', 'paper', 'mesh_lumerical', parts[-1])]
    for a in alts:
        if os.path.exists(a):
            return a
    return alts[0]


def NL(mode=".mode neff=2.35 ng=4.0 wl=1550e-9", extra=()):
    base = [mode,
            ".freq 193.1e12 193.1e12 1",
            ".source 1.0@a1 0.0@a2",
            "mzi0 a1 a2 b1 b2 theta=0.3",
            "pd1 b1 v1 level1 r0=1.0",
            "pd2 b2 v2 level1 r0=1.0"]
    return [l + '\n' for l in list(extra) + base]


def build():
    tb = TB('TB-10  SPEC-X correctness fixes')

    # ---------------- X1 : complex dtype honoured ------------------------
    sp = fresh_spipe()
    tb.ok('X1.default_is_complex128',
          sp.config['complex_dtype'] == torch.complex128,
          f"config['complex_dtype']={sp.config['complex_dtype']}")

    for dt in (torch.complex64, torch.complex128):
        sp = fresh_spipe(complex_dtype=dt)
        import spipe.photonic.model as M
        om = 2 * math.pi * torch.tensor([193.1e12], dtype=torch.float64)
        t = torch.tensor([0.0])
        common = dict(neff=2.35, ng=4.0, wl=1550e-9, omega=om, time=t)
        devs = {
            'WaveGuide': (M.WaveGuide, dict(l=1e-4, alpha=1.0)),
            'MZI':       (M.MZI, dict(theta=0.3)),
            'PBUm':      (M.PBUm, dict(theta=0.4, phi=1.1, l=0.0)),
            'PS':        (M.PS, dict(ps=0.7)),
            'ModM':      (M.ModM, dict(act=torch.tensor([0.3]), an='l1', coeff1=0.5, act_l=2e-6)),
            'ModP':      (M.ModP, dict(act=torch.tensor([0.3]), an='l1', coeff1=0.5, act_l=2e-6)),
        }
        for nm, (cls, kw) in devs.items():
            tag = 'c64' if dt == torch.complex64 else 'c128'
            try:
                S = cls(**{**kw, **common}).transfer(None)
                tb.ok(f'X1.{nm}.{tag}', S.dtype == dt, f"S.dtype={S.dtype} expected={dt}")
            except Exception as e:
                tb.ok(f'X1.{nm}.{tag}', False, f"RAISED {type(e).__name__}: {str(e)[:100]}")

        # assembled solve keeps the dtype, and the source vector too
        from spipe.photonic.photonic import Photonic
        try:
            p = Photonic(NL())
            src = list(p.srce_node.values())[0]
            tb.ok(f'X1.source_dtype.{tag}', src.dtype == dt, f"srce dtype={src.dtype} expected={dt}")
        except Exception as e:
            tb.ok(f'X1.source_dtype.{tag}', False, f"RAISED {type(e).__name__}: {str(e)[:100]}")

    # precision actually improves in c128: energy error on a loop circuit
    loop = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
        ".source 1.0@a1 0.0@a2",
        "mzi0 a1 a2 c1 c2 theta=0.35", "wg1 c2 d2 l=53e-6 alpha=1.0",
        "mzi2 c1 d2 e1 e2 theta=0.42",
        "pd3 e1 v1 level1 r0=1.0", "pd4 e2 v2 level1 r0=1.0"]]
    sp = fresh_spipe(complex_dtype=torch.complex128)
    from spipe.photonic.photonic import Photonic
    res, _, _ = Photonic(loop).simulate()
    tb.lt('X1.energy_err_c128', abs(res.sum().item() - 1.0), 1e-12,
          'lossless loop circuit must conserve energy to double precision')

    # ---------------- X2 : rtol branch actually fires --------------------
    sp = fresh_spipe()
    import spipe.core.core as core, inspect, re as _re
    src = inspect.getsource(core)
    _code = '\n'.join(l.split('#')[0] for l in src.splitlines())   # strip comments
    _flat = _re.sub(r'\s+', '', _code)
    tb.ok('X2.no_precedence_bug',
          'param_p-new_param_p/param_p' not in _flat,
          'the buggy expression must be gone from CODE (comments describing it are fine)')
    # Behavioural half. This used to import a `_converged()` helper that the solver has
    # never exported, so it skipped every run and the criterion was only ever grepped for.
    # The criterion lives inside solve_fixed_point(), so drive it from there instead: no
    # private API, and a real pass/fail.
    from spipe.core.core import solve_fixed_point, FixedPointNotConverged

    _cfg = dict(sp.config)
    _cfg['rtol'], _cfg['atol'], _cfg['max_iter'] = 1e-6, 1e-12, 50

    def _iters_for(scale, k=0.5):
        """Contraction x -> k*x + (1-k)*target, started one unit away, at a given scale."""
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64) * scale
        x, info = solve_fixed_point(lambda v: k * v + (1 - k) * target,
                                    torch.zeros_like(target), _cfg)
        return x, info, target

    x1, i1, t1 = _iters_for(1.0)
    tb.ok('X2.converged_true', bool(i1.get('converged')),
          f"a contraction must converge: iters={i1.get('iters')}, "
          f"residual={i1.get('residuals', [float('nan')])[-1]:.3e}, "
          f"tolerance={i1.get('tolerance'):.3e}")
    tb.lt('X2.converged_to_the_right_point',
          float((x1 - t1).abs().max() / t1.abs().max()), 1e-5,
          'and to the actual fixed point, not merely to a stopped iterate')

    # Scale-free: the SAME relative problem at 1e6x the signal must take the same number of
    # iterations. A criterion written as an absolute difference would stop later (or never).
    x2, i2, t2 = _iters_for(1e6)
    tb.ok('X2.scale_free', i2.get('iters') == i1.get('iters') and bool(i2.get('converged')),
          f"iters at scale 1 = {i1.get('iters')}, at scale 1e6 = {i2.get('iters')} "
          f"-- the criterion must not depend on absolute signal scale")

    # And it must not declare victory on iterates that are still far apart: a map whose
    # 'fixed point' is never approached has to run out of budget and raise.
    tb.raises('X2.diverged_false',
              lambda: solve_fixed_point(lambda v: v + 1.0,
                                        torch.zeros(3, dtype=torch.float64), _cfg),
              Exception,
              'x -> x + 1 has no fixed point; the criterion must never be met')

    # ---------------- X3 : determinism ------------------------------------
    sp = fresh_spipe()
    tb.ok('X3.seed_in_config', 'seed' in sp.config, f"config keys={sorted(sp.config)}")
    src = inspect.getsource(core)
    tb.ok('X3.no_global_manual_seed', 'torch.manual_seed' not in src,
          'must use a local torch.Generator, not the global RNG')
    tb.ok('X3.generator_used', 'Generator' in src or 'generator' in src,
          'expected a local torch.Generator')

    # ---------------- X4 : photonic_register works ------------------------
    sp = fresh_spipe()
    from spipe.photonic.model.base import Device

    class MyDev(Device):
        _name = 'mydev'
        _required_attr = ['time', 'omega', 'neff']
        _optional_attr = {'ng': None, 'wl': None}
        _num_port = [1, 1]
        _active_port = 0

        def forward(self, vari_set=None):
            import torch as _t
            from spipe import config as _c
            S = _t.zeros((len(self.params['time']), len(self.params['omega']), 2, 2),
                         dtype=_c['complex_dtype'])
            S[..., 0, 1] = S[..., 1, 0] = 1.0
            return S, {}

    okreg, _ = tb.no_raise('X4.register_runs', lambda: sp.photonic_register(MyDev))
    # NB: must re-import after fresh_spipe(), otherwise `Photonic` still refers to a
    # class from a purged module incarnation whose globals registration never touched.
    from spipe.photonic.photonic import Photonic
    if okreg:
        nl = [l + '\n' for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
            ".source 1.0@a1", "mydev0 a1 b1", "pd1 b1 v1 level1 r0=1.0"]]
        okuse, out = tb.no_raise('X4.registered_model_usable',
                                 lambda: Photonic(nl).simulate())
        if okuse and out is not None:
            tb.close('X4.registered_model_value', out[0].sum().item(), 1.0, 1e-10,
                     detail='unit-transmission custom device')
    tb.no_raise('X4.reset_runs', lambda: sp.photonic_reset())
    # must reject for the RIGHT reason: a name collision, not an unrelated import crash
    try:
        sp.photonic_register(type('Dup', (MyDev,), {'_name': 'wg'}))
        tb.ok('X4.collision_rejected', False, 'colliding name "wg" was accepted')
    except Exception as e:
        m = str(e).lower()
        tb.ok('X4.collision_rejected',
              ('already' in m or 'collid' in m or 'in use' in m or 'used' in m) and 'wg' in m,
              f"message must name the collision; got {str(e)[:140]!r}")

    # ---------------- X5 : power model ------------------------------------
    sp = fresh_spipe()
    from spipe.photonic.photonic import Photonic as Ph

    def pw(scale, power):
        # NOTE: deliberately LOSSY (alpha=0.7). With a lossless circuit the buggy
        # `optical_input_power * (input_unit - output_unit)` term vanishes and the
        # units defect is invisible, so the check would pass for the wrong reason.
        nl = [l + '\n' for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
            f".source {scale}@a1 0.0@a2 power={power} eff=0.2",
            "mzi0 a1 a2 c1 c2 theta=0.3",
            "wg1 c1 b1 l=100e-6 alpha=0.7",
            "wg2 c2 b2 l=100e-6 alpha=0.7",
            "pd1 b1 v1 level1 r0=1.0", "pd2 b2 v2 level1 r0=1.0"]]
        _, _, d = Ph(nl).simulate()
        return d

    okp, d1 = tb.no_raise('X5.power_runs', lambda: pw(1.0, 0.1))
    if okp and d1 is not None:
        keys = set(d1.keys()) if isinstance(d1, dict) else set()
        tb.ok('X5.reports_named_quantities',
              {'photonic'} <= keys or {'optical_input_w', 'optical_output_w'} <= keys,
              f"power dict keys={sorted(keys)}")

        def scalar(d):
            v = d.get('photonic')
            if isinstance(v, dict):
                return v
            return d

        # invariance: doubling source amplitude (4x power) and quartering `power`
        # must leave consumed electrical power unchanged
        d2 = pw(2.0, 0.025)

        def consumed(d):
            dd = scalar(d)
            for k in ('consumed_w', 'laser_drive_w', 'electrical_w', 'photonic'):
                if k in dd and dd[k] is not None:
                    x = dd[k]
                    return float(x if not hasattr(x, 'numel') else x.reshape(-1)[0])
            return float('nan')

        c1, c2 = consumed(d1), consumed(d2)
        tb.close('X5.scale_invariant_consumed', c2, c1, 1e-9,
                 detail='4x source power with 1/4x `power` must give identical consumed W')

        # optical output must never exceed optical input
        dd = scalar(d1)
        # the spec did not pin key names, so accept either spelling
        def pick(dd, *names):
            for n in names:
                if n in dd and dd[n] is not None:
                    return float(torch.as_tensor(dd[n]).reshape(-1)[0])
            return None
        oi = pick(dd, 'optical_input_w', 'optical_input_power')
        oo = pick(dd, 'optical_output_w', 'optical_output_power')
        ol = pick(dd, 'optical_loss_w', 'optical_loss')
        if oi is not None and oo is not None:
            tb.ok('X5.output_le_input', oo <= oi * (1 + 1e-12),
                  f"optical_out={oo:.6g} W  optical_in={oi:.6g} W")
            if ol is not None:
                tb.close('X5.loss_closes_budget', oo + ol, oi, 1e-12,
                         detail='output + loss must exactly equal input')
        else:
            tb.ok('X5.output_le_input', False,
                  f"no optical input/output power reported; keys={sorted(dd)}")

    # ---------------- X6 : WDM validation ---------------------------------
    sp = fresh_spipe()
    from spipe.photonic.photonic import Photonic as Ph2
    bad = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9",
        ".freq 193.1e12 193.1e12 1",                      # 1 freq, but wdm1to2 needs 2
        ".source 1.0@a1", "wdm1to2_0 a1 b1 b2",
        "pd1 b1 v1 level1 r0=1.0", "pd2 b2 v2 level1 r0=1.0"]]
    tb.raises('X6.freq_port_mismatch_raises', lambda: Ph2(bad).simulate(),
              Exception, 'len(omega)=1 with a 1->2 WDM must be a clear error')
    try:
        Ph2(bad).simulate()
    except Exception as e:
        msg = str(e).lower()
        tb.ok('X6.error_is_actionable',
              ('wdm' in msg or 'channel' in msg) and ('freq' in msg or 'omega' in msg),
              f"message={str(e)[:140]!r}")

    good = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9",
        ".freq 193.1e12 193.4e12 2", ".source 1.0@a1",
        "wdm1to2_0 a1 b1 b2",
        "pd1 b1 v1 level1 r0=1.0", "pd2 b2 v2 level1 r0=1.0"]]
    tb.no_raise('X6.matched_case_runs', lambda: Ph2(good).simulate())

    # ---------------- X7 : .mode positional --------------------------------
    sp = fresh_spipe()
    from spipe.photonic.photonic import Photonic as Ph3
    okpos, rpos = tb.no_raise('X7.positional_mode_runs',
                              lambda: Ph3(NL(mode=".mode 2.35 4.0 wl=1550e-9")).simulate())
    okkw, rkw = tb.no_raise('X7.keyword_mode_runs',
                            lambda: Ph3(NL(mode=".mode neff=2.35 ng=4.0 wl=1550e-9")).simulate())
    if okpos and okkw and rpos is not None and rkw is not None:
        tb.close('X7.positional_equals_keyword',
                 rpos[0].sum().item(), rkw[0].sum().item(), 1e-12,
                 detail='the two spellings must give identical results')

    # ---------------- X8 : quasi-static guard ------------------------------
    sp = fresh_spipe()
    tb.ok('X8.config_flag', 'quasistatic_check' in sp.config,
          f"config keys={sorted(sp.config)}")
    from spipe.photonic.photonic import Photonic as Ph4

    def run_with_time(wg_len, dt_total):
        """A modulated circuit so a real transient time grid exists."""
        nl = [l + '\n' for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
            ".source 1.0@a1 0.0@a2",
            "modm0 a1 a2 c1 c2 v0 level1 coeff1=0.05 act_l=2e-6 alpha=1.0",
            f"wg1 c1 b1 l={wg_len} alpha=1.0",
            "pd1 b1 v1 level1 r0=1.0", "pd2 c2 v2 level1 r0=1.0"]]
        t = torch.linspace(0.0, dt_total, 11)
        act = torch.zeros(11, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            try:
                Ph4(nl).simulate(t, act)
            except Exception:
                pass
            return ' | '.join(str(x.message).lower() for x in w)

    # 5 cm of ng=4 waveguide => ~0.67 ns group delay, vs dt = 1 ps  -> must warn
    msgs = run_with_time('5e-2', 1e-11)
    tb.ok('X8.warns_when_stretched',
          'quasi' in msgs or 'group delay' in msgs,
          f"warnings seen: {msgs[:220]!r}")
    # 100 um of waveguide (~1.3 ps) vs dt = 1 us -> must NOT warn
    msgs2 = run_with_time('100e-6', 1e-5)
    tb.ok('X8.no_false_positive',
          not ('quasi' in msgs2 or 'group delay' in msgs2),
          f"warnings seen on a short circuit: {msgs2[:220]!r}")

    # ---------------- X9 / X10 / X12 : hygiene ----------------------------
    tb.ok('X9.dead_module_deleted',
          not os.path.exists(_pkg('electronic', 'interp1d.py')),
          'spipe/electronic/interp1d.py must be gone')

    core_src = open(_pkg('core', 'core.py')).read()
    bad_prints = [l for l in core_src.splitlines()
                  if l.strip().startswith('print(')]
    tb.ok('X10.no_prints_in_core', not bad_prints, f"found: {bad_prints[:3]}")

    gi = os.path.join(REPO, '.gitignore')
    gitign = open(gi).read() if os.path.exists(gi) else ''
    tb.ok('X12.gitignore_pycache',
          '__pycache__' in gitign and '.pyc' in gitign, f"contents={gitign[:120]!r}")
    r = subprocess.run(['git', '-C', REPO, 'ls-files'], capture_output=True, text=True)
    tracked_pyc = [f for f in r.stdout.splitlines() if f.endswith('.pyc')]
    tb.ok('X12.pyc_untracked', not tracked_pyc,
          f"{len(tracked_pyc)} .pyc still tracked (e.g. {tracked_pyc[:2]})")

    # ---------------- X11 : main3_diff units -------------------------------
    # Numeric check, not a string check: evaluate the script's own freq variables and
    # beta expression and require a physically sane propagation constant.
    # _repo_file() resolves 'test1/' (pre-release) or examples/paper/mesh_lumerical/ (released)
    m3 = open(_repo_file('test1', 'main3_diff.py')).read()
    import re
    env = {'pi': math.pi, 'FreeLightSpeed': 299792458.0}
    # pull in simple module-level scalar constants (e.g. THZ = 1e12) so a fix that
    # introduces a named unit conversion evaluates correctly
    for cm in re.finditer(r'^\s*([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*(?:#.*)?$', m3, re.M):
        try:
            env[cm.group(1)] = float(cm.group(2))
        except ValueError:
            pass
    mfreq = re.search(r'^\s*freq_start\s*,\s*freq_end\s*,\s*freq_num\s*=\s*(.+?)\s*$',
                      m3, re.M)
    mbeta = re.search(r'^\s*beta\s*=\s*(.+?)\s*$', m3, re.M)
    mneff = re.search(r'^\s*ng\s*,\s*neff\s*,\s*wl\s*=\s*(.+?)\s*$', m3, re.M)
    tb.ok('X11.beta_line_found', mbeta is not None and mfreq is not None,
          f"freq line={bool(mfreq)} beta line={bool(mbeta)}")
    if mbeta and mfreq:
        try:
            env['freq_start'], env['freq_end'], env['freq_num'] = eval(mfreq.group(1), dict(env))
            if mneff:
                env['ng'], env['neff'], env['wl'] = eval(mneff.group(1), dict(env))
            else:
                env['neff'] = 2.35
            beta = eval(mbeta.group(1), dict(env))
            # physical value at ~193 THz, neff 2.35:  2*pi*193.1e12*2.35/c = 9.51e6 rad/m
            expect = 2 * math.pi * 193.1e12 * env.get('neff', 2.35) / 299792458.0
            tb.close('X11.beta_physical', beta, expect, 0.05,
                     detail='beta must be ~9.5e6 rad/m; the THz/Hz bug makes it 1e12 too small')
        except Exception as e:
            tb.ok('X11.beta_physical', False, f"could not evaluate: {e!r}")

    # ---------------- X13 : the native BJT refuses what it cannot model -----
    # The Gummel-Poon builder routed every .model parameter through an alias table and
    # silently dropped the rest -- LEVEL included. A VBIC HBT card (LEVEL=12, as in the IHP
    # SG13G2 PDK) therefore ran as Gummel-Poon and returned 187 uA where VBIC gives 593 uA:
    # 3.2x wrong, no warning. No bench exercised a BJT at all, which is how it survived.
    import warnings as _w
    from spipe.electronic.native import Netlist as _NL
    from spipe.electronic.native.units import SpiceSyntaxError as _SSE

    _bjt = ("* bjt\nVcc c 0 1.5\nVbe b 0 0.85\nRc c cc 1k\nQ1 cc b 0 nq\n"
            ".model nq npn {p}\n.op\n.end\n")

    def _bjt_op(params):
        ck = _NL.from_string(_bjt.format(p=params))
        return ck.op() if hasattr(ck, 'op') else ck.dc()

    for lvl in ('9', '12', '99'):
        tb.raises(f'X13.bjt_rejects_level_{lvl}',
                  lambda lvl=lvl: _bjt_op(f'level={lvl} is=1e-18 bf=800'), _SSE,
                  f'LEVEL={lvl} is not Gummel-Poon and must be refused, not silently solved')

    ok1, op1 = tb.no_raise('X13.bjt_level1_accepted',
                           lambda: _bjt_op('level=1 is=1e-18 bf=800'),
                           'LEVEL=1 is Gummel-Poon and must still run')
    ok0, op0 = tb.no_raise('X13.bjt_no_level_accepted',
                           lambda: _bjt_op('is=1e-18 bf=800'),
                           'no LEVEL means Gummel-Poon and must still run')
    if ok1 and ok0 and op1 is not None and op0 is not None:
        tb.close('X13.bjt_level1_same_as_default', float(op1.v('cc')), float(op0.v('cc')),
                 0.0, rel=False, detail='LEVEL=1 and no LEVEL are the same model')

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter('always')
        try:
            _bjt_op('is=1e-18 bf=800 rb=100 ikf=1e-3')
        except Exception:
            pass
    msgs = ' | '.join(str(c.message) for c in caught)
    tb.ok('X13.bjt_warns_unmodelled_params', 'RB' in msgs and 'IKF' in msgs,
          f"RB and IKF change the answer on a real deck; ignoring them must be visible: "
          f"{msgs[:110]!r}")

    # Closed form. With the node voltages solved, the collector current is fixed by the
    # Gummel-Poon constitutive law alone, and KCL says it all flows through Rc:
    #   Ic = IS*(exp(Vbe/Vt) - exp(Vbc/Vt)) - (IS/BR)*(exp(Vbc/Vt) - 1)
    # at NF = NR = 1, VAF = VAR = 0. This is the first check of the BJT's physics anywhere.
    if ok0 and op0 is not None:
        _k, _q = 1.380649e-23, 1.602176634e-19
        vt = _k * (273.15 + 27.0) / _q
        IS, BR = 1e-18, 1.0
        vc = float(op0.v('cc'))
        vbe, vbc = 0.85, 0.85 - vc
        ic_law = (IS * (math.exp(vbe / vt) - math.exp(vbc / vt))
                  - (IS / BR) * (math.exp(vbc / vt) - 1.0))
        ic_kcl = (1.5 - vc) / 1e3
        tb.close('X13.bjt_gummel_poon_closed_form', ic_kcl, ic_law, 1e-6,
                 detail=f"Ic through Rc = {ic_kcl * 1e6:.4f} uA vs the Gummel-Poon law "
                        f"at the solved Vbc = {vbc:.4f} V (forward active)")

    return tb


if __name__ == '__main__':
    main(build)
