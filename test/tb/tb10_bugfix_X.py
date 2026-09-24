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

    # ---------------- X14 : gradients on a GPU ------------------------------
    # Simulate.forward built its detector index on the CPU; backward then index_add_-ed a
    # CUDA tensor with it and raised "Expected all tensors to be on the same device". The
    # forward pass ran fine, so only gradients on a GPU were broken -- all of them.
    if not torch.cuda.is_available():
        tb.skip('X14.gpu_gradient_matches_cpu',
                'no CUDA device here; this guard runs wherever one exists')
    else:
        from spipe.photonic.photonic import Photonic as _Ph14
        _net14 = [l + "\n" for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.3e12 4",
            ".source 1.0@a1 0.0@a2",
            "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
            "pd1 b1 vo1 level1 r0=1.0", "pd2 b2 vo2 level1 r0=1.0"]]

        def _grad14(dev):
            sp.config['device'] = dev
            t = torch.linspace(0, 1e-9, 8, dtype=torch.float64, device=dev)
            v = torch.linspace(0, 2, 8, dtype=torch.float64,
                               device=dev).reshape(-1, 1).clone().requires_grad_(True)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out, _, _ = _Ph14(_net14, need_grads=True).simulate(t, v)
            (out[:, 0] ** 2).sum().backward()
            return v.grad.detach().cpu()

        try:
            g_cpu = _grad14(torch.device('cpu'))
            ok14, g_gpu = tb.no_raise('X14.gpu_gradient_runs',
                                      lambda: _grad14(torch.device('cuda:0')),
                                      'backward() on cuda:0 must not raise a device mismatch')
            if ok14 and g_gpu is not None:
                tb.lt('X14.gpu_gradient_matches_cpu', float((g_gpu - g_cpu).abs().max()), 1e-10,
                      'the same gradient on GPU and CPU (complex128)')
        finally:
            sp.config['device'] = torch.device('cpu')

    # ---------------- X15 : an ideal source on a capacitor node, under UIC ---------
    # UIC pins every grounded capacitor's node to its IC (0 V). Both modulator load models put
    # a capacitor on the drive node, so an ideal source there fixed that node twice and the
    # native engine raised "singular MNA matrix" on the simplest possible drive circuit, while
    # HSPICE and Xyce ran it. In SPICE a source wins over an initial condition.
    from spipe.electronic.native import Netlist as _N15
    for _label, _src in (("DC 1 V", "1.0"), ("PWL ramp", "PWL(0 0 10n 4)")):
        _deck = (f"* ideal source on a capacitor node\nVd n 0 {_src}\nC1 n 0 1p\n"
                 f"R1 n 0 1k\n.tran 1n 10n\n.end\n")
        _ok, _res = tb.no_raise(f'X15.uic_source_beats_cap_{_label.split()[0]}',
                                lambda d=_deck: _N15.from_string(d).tran(1e-9, 1e-8, uic=True),
                                f'{_label} on a grounded-capacitor node must simulate under UIC')
        if _ok and _res is not None:
            _v = [float(x) for x in _res.v('n')]
            _want = 1.0 if _label.startswith("DC") else 4.0
            tb.close(f'X15.uic_source_value_{_label.split()[0]}', _v[-1], _want, 1e-9,
                     detail=f'the source sets the node, not the 0 V capacitor pin ({_label})')

    # ---------------- X16 : a misspelled MOSFET parameter is not silently ignored -----
    # "WW=20u" used to fall back to the 100 um default width with no message, and an
    # unimplemented model parameter (RD=1k) was dropped silently.
    import warnings as _w16
    def _mos(card, model="LEVEL=1 VTO=0.7 KP=120u"):
        return _N15.from_string(f"* t\n.model nch NMOS ({model})\nVdd vdd 0 3.0\nVg g 0 1.2\n"
                                f"R1 vdd d 20k\n{card}\n.end\n")
    tb.raises('X16.mos_instance_typo_W', lambda: _mos("M1 d g 0 0 nch WW=20u L=1u"),
              Exception, 'WW=20u must be an error, not the 100 um default width')
    tb.raises('X16.mos_instance_typo_L', lambda: _mos("M1 d g 0 0 nch W=20u LL=1u"),
              Exception, 'LL=1u must be an error, not the 100 um default length')
    for _cid, _card, _model in (
            ('X16.mos_unused_instance_warns', "M1 d g 0 0 nch W=20u L=1u PD=10u", None),
            ('X16.mos_unused_model_param_warns', "M1 d g 0 0 nch W=20u L=1u",
             "LEVEL=1 VTO=0.7 KP=120u RD=1k")):
        with _w16.catch_warnings(record=True) as _caught:
            _w16.simplefilter("always")
            _ok16, _ = tb.no_raise(_cid + '_builds',
                                   lambda c=_card, m=_model: _mos(c) if m is None else _mos(c, m))
        tb.ok(_cid, any("does not implement" in str(x.message) for x in _caught),
              'an accepted-but-unused parameter is dropped with a warning, not silently')
    _ok16, _n16 = tb.no_raise('X16.mos_clean_card_builds',
                              lambda: _mos("M1 d g 0 0 nch W=20u L=1u"))
    if _ok16:
        tb.close('X16.mos_clean_card_width', float(_n16.param('M1', 'W')), 20e-6, 1e-12,
                 detail='a correctly spelled W is used as given')

    # ---------------- X17 : scratch files and relative .include ------------------------
    # Circuit used to write its deck to ./tmp in the caller's working directory, shared by every
    # run started there, and the built-in engine resolved a relative .include against that
    # ./tmp. Now each Circuit gets a private scratch directory and relative includes resolve
    # against the netlist's own directory.
    import tempfile as _tf17, shutil as _sh17, gc as _gc17
    _src17 = open(_repo_file('examples', 'link_driver_mzm.sp')).read()
    _models = ''.join(l for l in _src17.splitlines(True) if l.lower().startswith('.model'))
    _deck17 = ''.join(l if not l.lower().startswith('.model') else ''
                      for l in _src17.splitlines(True)).replace('.tran', '.include lib/models.lib\n.tran', 1)
    _d17 = _tf17.mkdtemp(prefix='tb10_x17_')
    _cwd17 = os.getcwd()
    try:
        os.makedirs(os.path.join(_d17, 'deck', 'lib'))
        os.makedirs(os.path.join(_d17, 'elsewhere'))
        open(os.path.join(_d17, 'deck', 'lib', 'models.lib'), 'w').write(_models)
        open(os.path.join(_d17, 'deck', 'link.sp'), 'w').write(_deck17)
        os.chdir(os.path.join(_d17, 'elsewhere'))
        _ok17, _c17 = tb.no_raise('X17.relative_include_resolves',
                                  lambda: sp.Circuit(os.path.join('..', 'deck', 'link.sp'), 'native'),
                                  'models only in a relative .include are found from another cwd')
        _ok17b, _c17b = tb.no_raise('X17.second_circuit_builds',
                                    lambda: sp.Circuit(os.path.join('..', 'deck', 'link.sp'), 'native'))
        if _ok17 and _ok17b:
            _w1, _w2 = _c17.e_circuit.spice_wrk_dir, _c17b.e_circuit.spice_wrk_dir
            tb.ok('X17.private_scratch_dirs', _w1 != _w2, f'{_w1} vs {_w2}: two runs must not share a deck')
            tb.ok('X17.nothing_written_to_cwd', os.listdir('.') == [], f'cwd contains {os.listdir(".")}')
            del _c17, _c17b; _gc17.collect()
            tb.ok('X17.scratch_removed', not os.path.exists(_w1) and not os.path.exists(_w2),
                  'the private scratch directory is removed with the Circuit')
    finally:
        os.chdir(_cwd17)
        _sh17.rmtree(_d17, ignore_errors=True)

    # ---------------- X18 : a detector output with no DC path to ground --------------
    # Without a load resistor the photocurrent charges the detector's 2 fF forever; one deck
    # failed Newton, another ran and returned 170 kV at the detector with no message.
    _d18 = _tf17.mkdtemp(prefix='tb10_x18_')
    try:
        _no_load = ''.join(l for l in _src17.splitlines(True) if not l.lower().startswith('rload'))
        open(os.path.join(_d18, 'no_load.sp'), 'w').write(_no_load)
        open(os.path.join(_d18, 'load.sp'), 'w').write(_src17)
        tb.raises('X18.floating_detector_refused',
                  lambda: sp.Circuit(os.path.join(_d18, 'no_load.sp'), 'native'), ValueError,
                  'a detector output with no DC path to ground is refused, naming the node')
        tb.no_raise('X18.loaded_detector_builds',
                    lambda: sp.Circuit(os.path.join(_d18, 'load.sp'), 'native'))
    finally:
        _sh17.rmtree(_d18, ignore_errors=True)

    # ---------------- X19 : UIC starting state, two more singular cases ---------------
    # (a) a capacitor on a node set by a source that hangs off another source (Eamp nrf nofs
    # ... with Vofs nofs 0); (b) a node between an inductor and a floating capacitor, which
    # nothing sets at t = 0. Both were "singular MNA matrix" on the built-in engine while HSPICE
    # ran them; they broke two shipped examples (bistable_latch, oeo_electronic) on it.
    _chain = ("* source chain\nVofs nofs 0 0.1\nVin npd 0 0.5\nEamp nrf nofs npd 0 2\n"
              "C1 nrf 0 1p\nR1 nrf 0 1k\n.tran 1n 10n\n.end\n")
    _ok19, _r19 = tb.no_raise('X19.uic_capacitor_on_source_chain',
                              lambda: _N15.from_string(_chain).tran(1e-9, 1e-8, uic=True))
    if _ok19:
        tb.close('X19.uic_source_chain_value', float(_r19.v('nrf')[0]), 1.1, 1e-9,
                 detail='v(nrf) = 0.1 + 2 * 0.5 from t = 0: the sources win over the capacitor pin')
    _rlc = ("* series RLC, floating capacitor\nVs a 0 PULSE(0 1 1n 0.1n 0.1n 5n 20n)\n"
            "L1 a b 10n\nC1 b c 1p\nR1 c 0 50\n.tran 0.1n 10n\n.end\n")
    _ok19b, _r19b = tb.no_raise('X19.uic_inductor_floating_capacitor',
                                lambda: _N15.from_string(_rlc).tran(1e-10, 1e-8, uic=True))
    if _ok19b:
        tb.lt('X19.uic_undetermined_node_starts_at_zero', abs(float(_r19b.v('b')[0])), 1e-9,
              'SPICE UIC: a node nothing sets starts at 0 V')

    # ---------------- X20 : photonic gradients a mesh-training user needs --------------
    # .prob fields came out of the solve as a dict, which autograd never tracked: a loss using
    # them got a silently zero (or missing) gradient. Passive parameters (a coupler angle, a
    # phase shift, a length) had no gradient at all, while the docs said quasistatic mode had.
    _P20 = sp.Photonic
    _base20 = [".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.0e12 193.2e12 3", ".source 1.0@a1 0.6j@a2"]

    def _mesh20(ps=0.9, theta=0.3, l=12.3e-6):
        return [x + "\n" for x in _base20 + [
            f"pbum0 a1 a2 x1 x2 theta={theta!r} phi=0.7 l=5e-6", f"ps0 x1 y1 ps={ps!r}",
            f"wg0 y1 z1 l={l!r} alpha=0.9", "mzi1 z1 x2 b1 b2 theta=0.4",
            "pd1 b1 v1 level1 r0=1", "pd2 b2 v2 level1 r0=1", ".prob y1"]]

    def _loss20(ph):
        I, pr, _ = ph.simulate()
        return (I * torch.tensor([1.0, 2.0], dtype=torch.float64)).sum() + 3 * pr['y1'][..., 1].real.sum()

    for _dev, _name, _kw, _v0 in (('pbum0', 'theta', 'theta', 0.3), ('ps0', 'ps', 'ps', 0.9),
                                  ('wg0', 'l', 'l', 12.3e-6)):
        try:
            _ph = _P20(_mesh20())
            _p = _ph.param(_dev, _name)
            _loss20(_ph).backward()
            _h = 1e-6 * abs(_v0)
            _fd = (float(_loss20(_P20(_mesh20(**{_kw: _v0 + _h})))) -
                   float(_loss20(_P20(_mesh20(**{_kw: _v0 - _h}))))) / (2 * _h)
            tb.lt(f'X20.passive_grad_{_dev}_{_name}', abs(float(_p.grad) - _fd) / abs(_fd), 1e-6,
                  f'Photonic.param({_dev!r}, {_name!r}): adjoint {float(_p.grad):.10g} vs FD {_fd:.10g} '
                  f'(the loss also uses a .prob field)')
        except Exception as _e:
            tb.ok(f'X20.passive_grad_{_dev}_{_name}', False, repr(_e))

    try:
        _mod20 = [x + "\n" for x in _base20[:2] + [".source 1.0@a1 0.0@a2",
                  "modp0 a1 x vd level1 coeff1=1e-3 act_l=100e-6", "mzi0 x a2 b1 b2 theta=0.25pi",
                  "pd1 b1 v1 level1 r0=1", "pd2 b2 v2 level1 r0=1", ".prob b1"]]
        _ph = _P20(_mod20); _t = torch.zeros(1, dtype=torch.float64)
        _f = lambda v: _ph.simulate(_t, v)[1]['b1'][..., 1].real.sum()
        _v = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        _f(_v).backward()
        _fd = (float(_f(torch.tensor([[0.3 + 1e-6]], dtype=torch.float64))) -
               float(_f(torch.tensor([[0.3 - 1e-6]], dtype=torch.float64)))) / 2e-6
        tb.lt('X20.probe_field_grad_vs_fd', abs(float(_v.grad) - _fd) / abs(_fd), 1e-6,
              f'd Re(probe)/d drive: adjoint {float(_v.grad):.10g} vs FD {_fd:.10g} (was 0 / no graph)')
    except Exception as _e:
        tb.ok('X20.probe_field_grad_vs_fd', False, repr(_e))

    # .prob direction: relative to the first device listed on the node, even a modulator
    try:
        def _dir20(first_mod):
            lines = ["modp0 a1 x vd level1 coeff1=1e-3 act_l=10e-6", "wg0 x b1 l=10e-6 alpha=0.5"]
            if not first_mod:
                lines.reverse()
            ph = _P20([x + "\n" for x in _base20[:2] + [".source 1.0@a1"] + lines +
                       ["pd1 b1 v1 level1 r0=1", ".prob x"]])
            f = ph.simulate(torch.zeros(1, dtype=torch.float64),
                            torch.zeros(1, 1, dtype=torch.float64))[1]['x'][0, 1]
            return [round(float(abs(f[0])), 6), round(float(abs(f[1])), 6)]
        tb.ok('X20.prob_direction_follows_netlist_order', _dir20(True) == [0.0, 1.0]
              and _dir20(False) == [1.0, 0.0],
              f'modulator listed first -> {_dir20(True)} (want [0, 1]); wg first -> {_dir20(False)}')
    except Exception as _e:
        tb.ok('X20.prob_direction_follows_netlist_order', False, repr(_e))

    def _build20(line, src=".source 1.0@a1 0.0@a2"):
        return _P20([x + "\n" for x in _base20[:2] + [src, line,
                     "pd1 b1 v1 level1 r0=1", "pd2 b2 v2 level1 r0=1"]])
    with warnings.catch_warnings(record=True) as _w20:
        warnings.simplefilter("always")
        tb.no_raise('X20.mzi_alpha_builds', lambda: _build20("mzi0 a1 a2 b1 b2 theta=0.25pi alpha=0.9").simulate())
    tb.ok('X20.mzi_alpha_at_l0_warns', any('has no effect' in str(x.message) for x in _w20),
          'mzi alpha with the default l=0 is ignored, and now says so (wg and pbum already did)')
    tb.raises('X20.negative_alpha_refused',
              lambda: _build20("mzi0 a1 a2 b1 b2 theta=0.25pi l=10e-6 alpha=-0.5").simulate(), ValueError)
    tb.raises('X20.unknown_key_at_construction',
              lambda: _build20("mzi0 a1 a2 b1 b2 theta=0.3 cp_left=0.1"), TypeError,
              'refused when the netlist is read, not at the first simulate()')
    tb.raises('X20.source_amplitude_named', lambda: _build20("mzi0 a1 a2 b1 b2 theta=0.3",
                                                             ".source 31.6m@a1"), ValueError)

    # ---------------- X21 : non-positive MOSFET W or L -------------------------------------
    # HSPICE and Xyce stop on these; the square law returned 1.25e-25 V for L=0 and a 9 V node
    # on a 3 V supply for W<0.
    tb.raises('X21.mos_zero_length_refused', lambda: _mos("M1 d g 0 0 nch W=20u L=0"), Exception)
    tb.raises('X21.mos_negative_width_refused', lambda: _mos("M1 d g 0 0 nch W=-20u L=1u"), Exception)

    return tb


if __name__ == '__main__':
    main(build)
