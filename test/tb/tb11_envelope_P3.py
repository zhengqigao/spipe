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

    return tb


if __name__ == '__main__':
    main(build)
