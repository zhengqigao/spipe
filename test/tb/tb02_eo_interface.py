"""TB-02 : the new `mzm` electro-optic model (SPEC-P1.1).

Ground truth is the textbook push-pull Mach-Zehnder modulator, so these checks are
analytic. The headline property is that V_pi must literally be the voltage that moves
the output from full-on to full-off -- which is exactly what `modm` gets wrong.
Written from SPEC_P1.md only.
"""
import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import torch

OM = None


def mzm_dev(sp, **kw):
    import spipe.photonic.model as M
    global OM
    OM = 2 * math.pi * torch.tensor([193.1e12], dtype=torch.float64)
    p = dict(neff=2.35, ng=4.0, wl=1550e-9, omega=OM, an='level1')
    p.update(kw)
    return M.MZM(**p) if hasattr(M, 'MZM') else None


def herm(S):
    S = S.detach()
    I = torch.eye(S.shape[-1], dtype=S.dtype)
    return (S.conj().transpose(-1, -2) @ S - I).abs().max().item()


def build():
    tb = TB('TB-02  electro-optic interface (new `mzm` model)')
    sp = fresh_spipe(complex_dtype=torch.complex128)

    import spipe.photonic.model as M
    has = hasattr(M, 'MZM')
    tb.ok('P1.mzm_class_exists', has, 'spipe.photonic.model.MZM')
    if not has:
        return tb

    VPI, VBIAS = 2.0, 0.0
    N = 41
    v = torch.linspace(-2 * VPI, 2 * VPI, N, dtype=torch.float64)
    t = torch.linspace(0, 1e-9, N)

    # ---- ideal, lossless, chirp-free ---------------------------------
    ok, d = tb.no_raise('P1.ideal_builds',
                        lambda: mzm_dev(sp, act=v, time=t, vpi=VPI, vbias=VBIAS,
                                        il=0.0, chirp=0.0))
    if not ok or d is None:
        return tb
    ok, S = tb.no_raise('P1.ideal_transfer', lambda: d.transfer(None))
    if not ok or S is None:
        return tb

    tb.lt('P1.unitary', herm(S), 1e-10,
          'lossless chirp-free MZM must be unitary at every drive voltage')
    tb.lt('P1.reciprocal', (S - S.transpose(-1, -2)).detach().abs().max().item(), 1e-12,
          'S must equal its transpose')

    # forward block: left ports -> right ports
    fwd = S[..., 2:, :2]                       # (T, F, 2, 2)
    bar = (fwd[:, 0, 0, 0].detach().abs() ** 2).numpy()
    cross = (fwd[:, 0, 1, 0].detach().abs() ** 2).numpy()

    dphi = (math.pi * (v - VBIAS) / VPI).numpy()
    ana_a = (torch.cos(torch.tensor(dphi) / 2) ** 2).numpy()
    ana_b = (torch.sin(torch.tensor(dphi) / 2) ** 2).numpy()
    import numpy as np
    # either assignment of bar/cross is acceptable; take the better match
    e1 = max(np.max(np.abs(bar - ana_a)), np.max(np.abs(cross - ana_b)))
    e2 = max(np.max(np.abs(bar - ana_b)), np.max(np.abs(cross - ana_a)))
    tb.lt('P1.cos2_sin2_law', min(e1, e2), 1e-9,
          'bar/cross power must follow cos^2(dphi/2) / sin^2(dphi/2)')
    tb.lt('P1.power_conserved', float(np.max(np.abs(bar + cross - 1.0))), 1e-10,
          'bar + cross = 1 for a lossless MZM')

    # ---- THE headline property: V_pi really is V_pi ---------------------
    out = bar if min(e1, e2) == e1 else cross          # the arm that peaks at v=vbias
    i_on = int(np.argmin(np.abs(v.numpy() - VBIAS)))
    i_off = int(np.argmin(np.abs(v.numpy() - (VBIAS + VPI))))
    tb.close('P1.on_at_vbias', float(out[i_on]), 1.0, 1e-9,
             detail='full transmission at v = vbias')
    tb.lt('P1.off_at_vbias_plus_vpi', float(out[i_off]), 1e-9,
          f"v = vbias + vpi = {VBIAS+VPI} must extinguish the output")
    # and the transition must be monotone in between (no half-period error)
    seg = out[min(i_on, i_off):max(i_on, i_off) + 1]
    tb.ok('P1.monotone_on_to_off', bool(np.all(np.diff(seg) <= 1e-12)) or
          bool(np.all(np.diff(seg) >= -1e-12)),
          'transmission must move monotonically from on to off over one V_pi')

    # ---- finite extinction ratio ---------------------------------------
    for er_db in (20.0, 30.0):
        ok, d2 = tb.no_raise(f'P1.er{int(er_db)}_builds',
                             lambda er_db=er_db: mzm_dev(sp, act=v, time=t, vpi=VPI,
                                                         vbias=VBIAS, er=er_db, il=0.0,
                                                         chirp=0.0))
        if not ok or d2 is None:
            continue
        S2 = d2.transfer(None)[..., 2:, :2]
        p_on = float(np.max((S2[:, 0, 0, 0].detach().abs() ** 2).numpy()))
        p_off = float(np.min((S2[:, 0, 0, 0].detach().abs() ** 2).numpy()))
        if p_off <= 0:
            tb.ok(f'P1.er{int(er_db)}_finite', False, 'off-state is exactly zero: ER is infinite')
            continue
        meas = 10 * math.log10(p_on / p_off)
        tb.close(f'P1.er{int(er_db)}_matches', meas, er_db, 0.01,
                 detail=f"measured ER = {meas:.3f} dB, requested {er_db} dB")

    # ---- insertion loss -------------------------------------------------
    for il_db in (1.0, 3.0):
        ok, d3 = tb.no_raise(f'P1.il{int(il_db)}_builds',
                             lambda il_db=il_db: mzm_dev(sp, act=v, time=t, vpi=VPI,
                                                         vbias=VBIAS, il=il_db, chirp=0.0))
        if not ok or d3 is None:
            continue
        S3 = d3.transfer(None)[..., 2:, :2]
        tot = (S3[:, 0, :, 0].abs() ** 2).sum(dim=-1)
        tb.close(f'P1.il{int(il_db)}_matches', float(tot.max()), 10 ** (-il_db / 10), 1e-6,
                 detail=f"peak power transmission vs 10^(-IL/10)")

    # ---- chirp moves common-mode phase only -----------------------------
    ok, dc = tb.no_raise('P1.chirp_builds',
                         lambda: mzm_dev(sp, act=v, time=t, vpi=VPI, vbias=VBIAS,
                                         il=0.0, chirp=1.0))
    if ok and dc is not None:
        Sc = dc.transfer(None)[..., 2:, :2]
        barc = (Sc[:, 0, 0, 0].detach().abs() ** 2).numpy()
        crossc = (Sc[:, 0, 1, 0].detach().abs() ** 2).numpy()
        tb.lt('P1.chirp_preserves_amplitude',
              float(np.max(np.abs((barc + crossc) - 1.0))), 1e-10,
              'chirp must not change power split, only common-mode phase')
        ph0 = torch.angle(S[:, 0, 2, 0].detach())
        ph1 = torch.angle(Sc[:, 0, 0, 0].detach())
        tb.ok('P1.chirp_changes_phase',
              float((ph1 - ph0).abs().max()) > 1e-3,
              'chirp=1 must produce a different output phase than chirp=0')

    # ---- modulator response time ----------------------------------------
    step = torch.where(t > t[len(t) // 2], torch.ones_like(t), torch.zeros_like(t)) * VPI
    ok, d0 = tb.no_raise('P1.tau0_builds',
                         lambda: mzm_dev(sp, act=step, time=t, vpi=VPI, vbias=VBIAS,
                                         il=0.0, chirp=0.0, tau=0.0))
    ok2, dt_ = tb.no_raise('P1.tau_builds',
                           lambda: mzm_dev(sp, act=step, time=t, vpi=VPI, vbias=VBIAS,
                                           il=0.0, chirp=0.0, tau=1e-10))
    if ok and ok2 and d0 is not None and dt_ is not None:
        S0 = d0.transfer(None)[..., 2:, :2]
        St = dt_.transfer(None)[..., 2:, :2]
        a0 = (S0[:, 0, 0, 0].detach().abs() ** 2).numpy()
        at = (St[:, 0, 0, 0].detach().abs() ** 2).numpy()
        # tau=0 must equal the instantaneous evaluation
        okref, dref = tb.no_raise('P1.tau_ref',
                                  lambda: mzm_dev(sp, act=step, time=t, vpi=VPI,
                                                  vbias=VBIAS, il=0.0, chirp=0.0))
        if okref and dref is not None:
            aref = (dref.transfer(None)[..., 2:, :2][:, 0, 0, 0].detach().abs() ** 2).numpy()
            tb.lt('P1.tau0_equals_instant', float(np.max(np.abs(a0 - aref))), 1e-12,
                  'tau=0 must reproduce the instantaneous model exactly')
        tb.ok('P1.tau_smooths_step', float(np.max(np.abs(at - a0))) > 1e-3,
              'a non-zero tau must visibly slow the response to a step')

    # ---- regression: modm must be untouched -----------------------------
    sp2 = fresh_spipe(complex_dtype=torch.complex128)
    import spipe.photonic.model as M2
    ok, dm = tb.no_raise('P1.modm_still_works',
                         lambda: M2.ModM(act=torch.tensor([0.3]), an='level1', coeff1=0.5,
                                         act_l=2e-6, neff=2.35, ng=4.0, wl=1550e-9,
                                         omega=OM, time=torch.tensor([0.0])))
    if ok and dm is not None:
        Sm = dm.transfer(None)
        tb.lt('P1.modm_unitary', herm(Sm), 1e-10, 'modm must be unchanged and still unitary')
        # modm is the coupler form: S11 == S22 (this is what distinguishes it from a real MZM)
        tb.lt('P1.modm_is_still_coupler_form',
              (Sm[0, 0, 2, 0] - Sm[0, 0, 3, 1]).detach().abs().item(), 1e-12,
              'modm must remain the coupler-form abstraction (S11 == S22)')

    # ---------- passivity with a finite extinction ratio -------------------
    # The ER imbalance was once written as arm amplitudes base*(1 +/- eps), which puts more than
    # `base` on one arm: at il=0, er=10 dB the device emitted 1.1 W for 1 W in. Nothing above
    # tests a finite er for passivity (the IL checks use er=inf), which is how it survived.
    from spipe.photonic.photonic import Photonic as _Ph
    _t = torch.linspace(0, 1e-8, 81)
    _v = torch.linspace(-4.0, 4.0, 81, dtype=torch.float64).reshape(-1, 1)
    for _er in (10.0, 20.0, 30.0):
        for _il in (0.0, 3.0):
            _net = [l + "\n" for l in [
                ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
                ".source 1.0@a1 0.0@a2",
                f"mzm0 a1 a2 b1 b2 vdrv level1 vpi=2.0 vbias=0.0 il={_il} er={_er}",
                "pd1 b1 vo1 level1 r0=1.0", "pd2 b2 vo2 level1 r0=1.0"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _pc = _Ph(_net).simulate(_t, _v)[0]
            _limit = 10 ** (-_il / 10)
            tb.ok(f'P1.passive_er{_er:g}_il{_il:g}', float(_pc.sum(1).max()) <= _limit + 1e-12,
                  f"max total output {float(_pc.sum(1).max()):.6f} W must not exceed "
                  f"10^(-il/10) = {_limit:.6f} W for 1 W in")
            _ratio = float(_pc[:, 0].max() / _pc[:, 0].min())
            tb.close(f'P1.er_exact_er{_er:g}_il{_il:g}', 10 * math.log10(_ratio), _er, 1e-9,
                     detail='the extinction ratio must still be exactly the requested one')

    return tb


if __name__ == '__main__':
    main(build)
