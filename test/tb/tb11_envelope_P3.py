"""TB-11 : envelope propagation / optical memory (SPEC-P3).

The decisive check is the ring resonator: driven by a step, a real cavity fills
EXPONENTIALLY with the photon lifetime, while the quasi-static solver jumps instantly
because it has no memory at all. That difference is structural, so it cannot be faked
by tuning a fit.
Written from SPEC_P3.md only.
"""
import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch

C0 = 299792458.0


def fit_tau(t, y):
    """Fit y(t) = y_inf + (y0 - y_inf) exp(-t/tau); returns tau or None."""
    y = np.asarray(y, dtype=float)
    yinf = float(np.mean(y[-max(3, len(y) // 10):]))
    d = np.abs(y - yinf)
    m = (d > d.max() * 1e-3) & (d > 0)
    if m.sum() < 5:
        return None
    A = np.polyfit(t[m], np.log(d[m]), 1)
    return None if A[0] >= 0 else -1.0 / A[0]


def build():
    tb = TB('TB-11  envelope propagation (optical memory)')
    sp = fresh_spipe(complex_dtype=torch.complex128)
    from spipe.photonic.photonic import Photonic

    # does the mode= argument exist at all?
    probe = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
        ".source 1.0@a1 0.0@a2",
        "modm0 a1 a2 b1 b2 v0 level1 coeff1=0.05 act_l=2e-6 alpha=1.0",
        "pd1 b1 vo1 level1 r0=1.0", "pd2 b2 vo2 level1 r0=1.0"]]
    t = torch.linspace(0, 2e-9, 32)
    act = torch.zeros(32, 1, dtype=torch.float64)
    ok, _ = tb.no_raise('P3.mode_arg_exists',
                        lambda: Photonic(probe).simulate(t, act, mode='envelope'))
    if not ok:
        tb.ok('P3.available', False, "simulate(..., mode='envelope') not supported")
        return tb

    # ---------- 1. adiabatic limit: envelope == quasistatic ---------------
    # short waveguides -> group delay far below dt -> the two must agree
    nl = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.0e12 193.2e12 65",
        ".source 1.0@a1 0.0@a2",
        "modm0 a1 a2 b1 b2 v0 level1 coeff1=0.05 act_l=2e-6 alpha=1.0",
        "wg1 b1 c1 l=10e-6 alpha=1.0",
        "pd2 c1 vo1 level1 r0=1.0", "pd3 b2 vo2 level1 r0=1.0"]]
    NT = 64
    t = torch.linspace(0, 1e-6, NT)                   # dt ~ 16 ns >> 1.3e-13 s delay
    act = torch.linspace(0.0, 1.0, NT).reshape(NT, 1).to(torch.float64)
    p = Photonic(nl)
    rq, _, _ = p.simulate(t, act, mode='quasistatic')
    re, _, _ = p.simulate(t, act, mode='envelope')
    err = float((rq - re).abs().max() / rq.abs().max())
    tb.lt('P3.adiabatic_limit', err, 1e-9,
          'with group delay << dt, envelope must reproduce quasistatic')

    # ---------- 2. lossless energy conservation in envelope mode ----------
    nl2 = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.0e12 193.2e12 65",
        ".source 1.0@a1 0.0@a2", "mzi0 a1 a2 c1 c2 theta=0.35",
        "wg1 c2 d2 l=53e-6 alpha=1.0", "mzi2 c1 d2 e1 e2 theta=0.42",
        "pd3 e1 v1 level1 r0=1.0", "pd4 e2 v2 level1 r0=1.0"]]
    try:
        p2 = Photonic(nl2)
        r2, _, _ = p2.simulate(mode='envelope')
        nf = len(p2.omega)
        tot = r2.sum().item() / nf
        tb.close('P3.energy_envelope', tot, 1.0, 1e-9,
                 detail='lossless passive network, envelope mode')
    except Exception as e:
        tb.ok('P3.energy_envelope', False, f"raised {e!r}")

    # ---------- 3. THE test: ring resonator must ring up ------------------
    # all-pass ring: coupler (MZI, self-coupling r = cos theta) with the feedback
    # waveguide closing port4 -> port2.
    theta = 0.30
    r_amp = math.cos(theta)
    a_amp = 0.92
    ng = 4.0
    Lrt = 300e-6
    T_rt = ng * Lrt / C0                              # round-trip time
    tau_amp = -T_rt / math.log(r_amp * a_amp)         # amplitude photon lifetime
    tau_pow = tau_amp / 2.0                           # power decays twice as fast

    NT = 512
    tstop = 40 * tau_amp
    t = torch.linspace(0, tstop, NT)
    # modulator acts as the shutter: drive steps at t = tstop/4
    step = torch.where(t > tstop / 4, torch.ones_like(t), torch.zeros_like(t))
    act = step.reshape(NT, 1).to(torch.float64)

    # Fit ONE carrier, not the detector sum. The band spans many free spectral
    # ranges (FSR = 1/T_rt), so the detector sums carriers at wildly different
    # detunings -- on-resonance ones ring up slowly, off-resonance ones almost
    # instantly -- and that superposition has no single time constant. tau_p
    # depends only on |r*a| and T_rt, so any one full-window carrier gives it.
    NT = 256
    dt = tau_amp / 20.0
    tstop = (NT - 1) * dt
    t = torch.linspace(0, tstop, NT)
    istep = NT // 8
    step = torch.zeros(NT, dtype=torch.float64)
    step[istep:] = 200.0
    act = step.reshape(NT, 1)

    band = 5.0 / dt                     # band*dt = 5  -> edge fraction ~20%
    df = 1.0 / (20.0 * tau_amp)         # unaliased window +/- 10 tau_p
    nf = int(round(band / df)) + 1
    f0, f1 = 193.1e12 - band / 2, 193.1e12 + band / 2

    ring = [l + '\n' for l in [
        f".mode neff={ng} ng={ng} wl=1550e-9",
        f".freq {f0:.10e} {f1:.10e} {nf}",
        ".source 1.0@pin",
        "modp0 pin pgate v0 coeff1=1.0 act_l=1e-9",
        "mzi1 pgate pring_b pthru pring_a theta=%.10f" % theta,
        f"wg2 pring_a pring_b l={Lrt} alpha={a_amp}",
        "pd3 pthru vo1 level1 r0=1.0",
        ".prob pthru",
    ]]
    tb.ok('P3.ring_analytic', True,
          f"T_rt={T_rt:.4e}s  tau_amp={tau_amp:.4e}s  r*a={r_amp*a_amp:.4f}  "
          f"nfreq={nf}  band*dt={band*dt:.1f}  window/tau={1/(2*df)/tau_amp:.0f}")

    def carrier_fit(mode):
        p = Photonic(ring)
        _, mid, _ = p.simulate(t, act, mode=mode)
        E = mid['pthru'][..., 1].detach()          # outward field, (T, F)
        tr = (E - E[-1:]).abs().max(dim=0).values  # transient size per carrier
        M = int(round(1.0 / (df * dt)))
        lo, hi = M // 2, E.shape[1] - M // 2
        if hi <= lo:
            return None, None
        k = int(torch.argmax(tr[lo:hi]).item()) + lo
        y = (E[:, k] - E[-1, k]).abs().numpy()
        tn = t.numpy()
        i0, i1 = istep + 4, istep + 4 + int(4 * tau_amp / dt)
        i1 = min(i1, NT - 2)
        seg = y[i0:i1]
        if seg.max() <= 0 or np.count_nonzero(seg) < 6:
            return None, k
        sl = np.polyfit(tn[i0:i1] - tn[i0], np.log(np.maximum(seg, 1e-300)), 1)[0]
        return (None if sl >= 0 else -1.0 / sl), k

    okr, got = tb.no_raise('P3.ring_runs', lambda: (carrier_fit('quasistatic'),
                                                    carrier_fit('envelope')))
    if okr and got is not None:
        (tq, kq), (te, ke) = got
        tb.ok('P3.quasistatic_jumps_instantly', tq is None,
              f"quasi-static must show NO exponential approach (fitted tau={tq}), "
              f"because it has no optical memory")
        if te is None:
            tb.ok('P3.envelope_rings_up', False,
                  f"envelope showed no exponential approach on carrier {ke}")
        else:
            tb.lt('P3.envelope_rings_up', abs(te - tau_amp) / tau_amp, 0.05,
                  f"carrier {ke}: fitted tau = {te:.5e} s vs analytic "
                  f"tau_p = -T_rt/ln(r*a) = {tau_amp:.5e} s")

    # ---------- 4. bandwidth honesty --------------------------------------
    # too few frequency points to resolve a long delay -> must warn, not alias silently
    thin = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9",
        ".freq 193.0e12 193.2e12 4",                # very coarse
        ".source 1.0@a1 0.0@a2",
        "modm0 a1 a2 b1 b2 v0 level1 coeff1=0.05 act_l=2e-6 alpha=1.0",
        "wg1 b1 c1 l=5e-2 alpha=1.0",               # ~0.67 ns delay
        "pd2 c1 vo1 level1 r0=1.0", "pd3 b2 vo2 level1 r0=0.0"]]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            tt = torch.linspace(0, 1e-9, 32)
            # a CONSTANT drive takes a documented fast path, so the
            # bandwidth check is never reached -- vary it
            aa = torch.linspace(0.0, 1.0, 32).reshape(32, 1).double()
            Photonic(thin).simulate(tt, aa, mode='envelope')
        except Exception:
            pass
        msgs = ' | '.join(str(x.message).lower() for x in w)
    tb.ok('P3.warns_on_insufficient_band',
          any(k in msgs for k in ('band', 'resolution', 'alias', 'wrap', 'delay')),
          f"warnings seen: {msgs[:220]!r}")

    # ---------- 5. default behaviour unchanged ----------------------------
    p3 = Photonic(nl)
    tt5 = torch.linspace(0, 1e-6, 64)
    aa5 = torch.linspace(0, 1, 64).reshape(64, 1).to(torch.float64)
    d1, _, _ = p3.simulate(tt5, aa5)                       # no mode= at all
    d2, _, _ = p3.simulate(tt5, aa5, mode='quasistatic')
    tb.lt('P3.default_is_quasistatic', float((d1 - d2).abs().max()), 1e-15,
          'omitting mode= must be exactly the historical behaviour')

    # ---------------- gradients through mode='envelope' --------------------------------
    # Device._param_wrap stores every attribute as a torch.nn.Parameter, which builds a NEW
    # leaf and discards the drive's history. Envelope mode is plain autograd, so its loss had
    # a real grad_fn, backward() completed without error, and drive.grad was silently None.
    # These checks drive a genuinely non-adiabatic circuit (a 4 ps delay sampled at 2 ps) so
    # envelope mode is really doing something the quasi-static path cannot.
    import warnings as _w
    from spipe.photonic.photonic import Photonic as _Ph
    sp.config['quasistatic_check'] = False
    _net = [l + "\n" for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9",
        ".freq 192.6e12 193.6e12 65",
        ".source 1.0@a1 0.0@a2",
        "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
        "wg0 b1 c1 l=300e-6",
        "pd1 c1 vo1 level1 r0=1.0",
        "pd2 b2 vo2 level1 r0=1.0"]]
    _T = 48
    _t = torch.arange(_T, dtype=torch.float64) * 2e-12
    _base = (1.0 + 0.8 * torch.sin(2 * math.pi * _t / (_T * 2e-12) * 3)).reshape(-1, 1)
    _wts = torch.linspace(0.5, 1.5, _T, dtype=torch.float64)

    def _env_loss(drive):
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out, _, _ = _Ph(_net).simulate(_t, drive, mode='envelope')
        return (_wts * out[:, 0]).sum()

    try:
        _d = _base.clone().requires_grad_(True)
        _loss = _env_loss(_d)
        _loss.backward()
        tb.ok('ENV.grad_reaches_drive', _d.grad is not None,
              'backward() through mode=envelope must populate drive.grad; it used to complete '
              'without error and leave it None')
        if _d.grad is not None:
            _worst = 0.0
            for _k in (0, 15, 31, 47):
                _h = 1e-4
                _p = _base.clone(); _p[_k, 0] += _h
                _m = _base.clone(); _m[_k, 0] -= _h
                with torch.no_grad():
                    _fd = (float(_env_loss(_p)) - float(_env_loss(_m))) / (2 * _h)
                _worst = max(_worst, abs(float(_d.grad[_k, 0]) - _fd) / max(abs(_fd), 1e-30))
            tb.lt('ENV.grad_matches_fd', _worst, 1e-6,
                  f'analytic d(loss)/d(drive) vs central finite differences at 4 instants '
                  f'(worst relative error)')

        # The forward answer must not depend on whether a graph was requested.
        with torch.no_grad():
            _plain = _Ph(_net).simulate(_t, _base, mode='envelope')[0]
        _graph = _Ph(_net).simulate(_t, _base.clone().requires_grad_(True), mode='envelope')[0]
        tb.ok('ENV.forward_unchanged_by_grad', torch.equal(_plain, _graph.detach()),
              f'max |with graph - without| = {float((_plain - _graph.detach()).abs().max()):.3e}')

        # The envelope model cache is keyed by VALUE. Two equal drives on different graphs
        # must not share a cached model, or the second backward attaches to the first graph.
        _ph = _Ph(_net)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            _ph.simulate(_t, _base.clone(), mode='envelope')          # populate the cache
            _d2 = _base.clone().requires_grad_(True)
            (_wts * _ph.simulate(_t, _d2, mode='envelope')[0][:, 0]).sum().backward()
        tb.ok('ENV.cache_does_not_steal_graph', _d2.grad is not None,
              'a value-equal cached drive must not swallow the gradient of a new one')
    except Exception as e:
        tb.ok('ENV.grad_reaches_drive', False, f"{e!r}")

    # ---------------- envelope mode must not return runaway or unsettled results ------------
    # A modulator inside a ring, sampled at a step shorter than the round trip, drove the
    # envelope recursion unstable: 464x the launched power at the end of the record, returned
    # with no error. (A resonator CAN briefly emit more than it receives -- stored energy --
    # so the test is runaway growth, not instantaneous passivity.)
    import warnings as _w2
    from spipe.photonic.photonic import Photonic as _Ph2
    _c = 299792458.0
    _L = 2 * math.pi * 20e-6
    _fres = 193.10292e12

    def _ring(t_amp, a_amp, nfreq, band, coeff):
        return [l + "\n" for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9",
            f".freq {_fres - band / 2} {_fres + band / 2} {nfreq}", ".source 1.0@a1",
            f"mzi0 a1 a2 b1 b2 theta={math.acos(t_amp)}", f"wg0 b2 x l={_L} alpha={a_amp}",
            f"modp0 x a2 vdrv level1 coeff1={coeff} act_l=10e-6", "pd1 b1 vo level1 r0=1.0"]]

    _T = 301
    _tt = torch.arange(_T, dtype=torch.float64) * 1e-12
    _drv = torch.cat([torch.ones(_T // 6, 1), torch.zeros(_T - _T // 6, 1)]).double()
    tb.raises('ENV.runaway_is_an_error',
              lambda: _Ph2(_ring(0.95, 0.99, 1001, 2e12, 1e-2)).simulate(_tt, _drv, mode='envelope'),
              RuntimeError,
              'modulator inside a ring at dt < round trip: the unstable recursion must raise')

    # A high-Q ring that settles to the wrong level must warn: the envelope result at the end,
    # with the drive long constant, must match the quasi-static steady state.
    _T2 = 1001
    _tt2 = torch.linspace(0, 10e-9, _T2)
    _drv2 = torch.cat([torch.ones(_T2 // 10, 1), torch.zeros(_T2 - _T2 // 10, 1)]).double()
    with _w2.catch_warnings(record=True) as _caught:
        _w2.simplefilter("always")
        try:
            _Ph2(_ring(0.998, 0.999, 4001, 400e9, 1e-4)).simulate(_tt2, _drv2, mode='envelope')
            _raised = None
        except Exception as e:
            _raised = e
    _msgs = [str(x.message) for x in _caught if 'did not settle' in str(x.message)]
    tb.ok('ENV.unsettled_steady_state_warns', bool(_msgs) or isinstance(_raised, RuntimeError),
          f"a high-Q ring whose envelope settles 0.9% off the steady state must be flagged: "
          f"{(_msgs[0][:90] if _msgs else repr(_raised)[:90])!r}")

    # ---- a constant drive still has a delayed gradient -------------------------------
    # A constant drive used to be folded into the static system and the run handed to the
    # quasi-static adjoint: d pd[50]/d drive[50] = 315 where the true sensitivity sits on
    # drive[40], 100 ps (the delay line) earlier. Values were right; the gradient was not causal.
    try:
        from spipe.photonic.photonic import Photonic as _PhC
        _c = 299792458.0; _dt = 10e-12; _N = 81
        _net = [x + "\n" for x in [".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 192.9e12 193.3e12 401",
                ".source 1.0@a1 0.0@a2", "mzm0 a1 a2 b1 b2 vdrv level1 vpi=2.0 act_l=1e-9",
                f"wg0 b1 c1 l={100e-12 * _c / 4.0!r}", "pd1 c1 vo1 level1 r0=1.0"]]
        _t = torch.arange(_N, dtype=torch.float64) * _dt
        _ph = _PhC(_net)
        _d0 = torch.full((_N, 1), 1.0, dtype=torch.float64)
        _d = _d0.clone().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ph.simulate(_t, _d, mode='envelope')[0][50, 0].backward()

            def _fd(n, h=1e-5):
                dp, dm = _d0.clone(), _d0.clone(); dp[n, 0] += h; dm[n, 0] -= h
                with torch.no_grad():
                    return float(_ph.simulate(_t, dp, mode='envelope')[0][50, 0]
                                 - _ph.simulate(_t, dm, mode='envelope')[0][50, 0]) / (2 * h)
            _fd40, _fd50 = _fd(40), _fd(50)
        tb.close('P3.const_drive_grad_is_delayed', float(_d.grad[40, 0]), _fd40, 1e-6,
                 detail=f'd pd[50]/d drive[40] (100 ps earlier) vs FD {_fd40:.6g}')
        tb.lt('P3.const_drive_grad_not_instant', abs(float(_d.grad[50, 0])), 1e-6 * abs(_fd40),
              f'd pd[50]/d drive[50] = {float(_d.grad[50, 0]):.3g}; FD {_fd50:.3g} (light is still in flight)')
    except Exception as _e:
        tb.ok('P3.const_drive_grad_is_delayed', False, repr(_e))

    # ---- the quasi-static guard sees a ring that 0 V hides --------------------------------
    # With the modulator steering all light to the other output at zero drive, the ring path
    # carried 1e-30 of the power and was skipped: no warning at dt = 200 ps against a ~380 ps
    # group delay. The guard now also measures at the drive the run actually uses.
    try:
        _c = 299792458.0
        _neff, _ng, _f0 = 2.35, 4.0, 193.1e12
        _lam = _c / _f0; _m = round(10e-12 * _c / _ng / (_lam / _neff)); _L = _m * _lam / _neff
        _ring = [x + "\n" for x in [f".mode neff={_neff} ng={_ng} wl=1550e-9",
                 f".freq {_f0 - 50e9} {_f0 + 50e9} 201", ".source 1.0@a1 0.0@a2",
                 "mzm0 a1 a2 b1 x2 vdrv level1 vpi=2.0 act_l=1e-9",
                 f"mzi0 b1 r2 o1 r1 theta={math.acos(0.95)!r}", f"wg0 r1 r2 l={_L!r} alpha=0.99",
                 "pd1 o1 vo1 level1 r0=1.0", "pd2 x2 vo2 level1 r0=1.0"]]
        from spipe.photonic.photonic import Photonic as _PhR
        sp.config['quasistatic_check'] = True            # an earlier section switched it off
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            _PhR(_ring).simulate(torch.arange(10, dtype=torch.float64) * 200e-12,
                                 torch.full((10, 1), 2.0, dtype=torch.float64))
        tb.ok('P3.quasistatic_guard_sees_hidden_ring',
              any('Quasi-static' in str(x.message) for x in _w),
              'ring behind a modulator that is off at 0 V, run at 2 V with dt = 200 ps')
    except Exception as _e:
        tb.ok('P3.quasistatic_guard_sees_hidden_ring', False, repr(_e))

    return tb


if __name__ == '__main__':
    main(build)
