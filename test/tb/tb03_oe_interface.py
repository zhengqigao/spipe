"""TB-03 : photodetector / opto-electronic interface (SPEC-P1.2).

The decisive check is statistical: shot noise must scale as sqrt(I), thermal noise must
not scale with I at all.  The shipped model (`clean * (1 + std*randn)`) has a constant
RELATIVE sigma, i.e. sigma ~ I^1.0, so it fails this by a wide margin.
Written from SPEC_P1.md only.
"""
import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch

Q = 1.602176634e-19
KB = 1.380649e-23


def build():
    tb = TB('TB-03  opto-electronic interface (photodetector)')
    sp = fresh_spipe(complex_dtype=torch.complex128)
    from spipe.photonic.model.pd_array import PDArray

    om = 2 * math.pi * torch.tensor([193.1e12], dtype=torch.float64)

    # ---------- 1. responsivity is linear in optical power ---------------
    R0, NS = 0.8, 64
    pd = PDArray([dict(r0=R0)], om)
    amp = torch.logspace(-3, 0, NS, dtype=torch.float64).reshape(NS, 1, 1)
    x = amp.to(torch.complex128)
    ok, i = tb.no_raise('OE.clean_runs', lambda: pd(x))
    if ok and i is not None:
        got = i.reshape(-1).detach().numpy()
        want = R0 * (amp.reshape(-1).numpy() ** 2)
        tb.lt('OE.linear_in_power', float(np.max(np.abs(got - want) / want)), 1e-12,
              'I = R * |E|^2 with no noise configured')

    # ---------- 2. shot noise must scale as sqrt(I) ----------------------
    BW = 1e9
    levels = [1e-6, 1e-5, 1e-4, 1e-3]          # target photocurrents, 3 decades
    NSAMP = 20000
    made, sigmas, means = True, [], []
    for I0 in levels:
        a = math.sqrt(I0 / R0)
        try:
            pdn = PDArray([dict(r0=R0, bw=BW, inoise=0.0)], om)   # thermal off: isolate shot
        except Exception:
            try:
                pdn = PDArray([dict(r0=R0, bw=BW, inoise=0.0)], om)
            except Exception as e:
                tb.ok('OE.shot_noise_configurable', False,
                      f"cannot build a PD with bw=/shot= parameters: {e!r}")
                made = False
                break
        xs = torch.full((NSAMP, 1, 1), a, dtype=torch.complex128)
        try:
            out = pdn(xs).reshape(-1).detach().numpy()
        except Exception as e:
            tb.ok('OE.shot_noise_configurable', False, f"PD with bw= raised {e!r}")
            made = False
            break
        sigmas.append(float(np.std(out)))
        means.append(float(np.mean(out)))
    if made and len(sigmas) == len(levels):
        tb.ok('OE.shot_noise_configurable', True, f"sigmas={['%.3e' % s for s in sigmas]}")
        # slope of log sigma vs log I
        slope = float(np.polyfit(np.log(levels), np.log(np.maximum(sigmas, 1e-300)), 1)[0])
        tb.close('OE.shot_sigma_sqrtI', slope, 0.5, 0.15,
                 detail=f"d(log sigma)/d(log I) = {slope:.4f}; shot noise = 0.5, "
                        f"the old multiplicative model = 1.0")
        # absolute magnitude at the top level
        exp_sig = math.sqrt(2 * Q * levels[-1] * BW)
        tb.close('OE.shot_sigma_magnitude', sigmas[-1], exp_sig, 0.25,
                 detail=f"sigma at I={levels[-1]:.1e} A, B={BW:.1e} Hz; "
                        f"expected sqrt(2qIB)={exp_sig:.3e}")
        tb.lt('OE.mean_unbiased', abs(means[-1] - levels[-1]) / levels[-1], 0.02,
              'noise must be zero-mean: it is added in current, not multiplied')

    # ---------- 3. thermal noise must NOT scale with I --------------------
    try:
        tsig = []
        for I0 in (1e-12, 1e-9):   # shot negligible vs thermal here
            a = math.sqrt(I0 / R0)
            pdt = PDArray([dict(r0=R0, bw=BW, inoise=1e-12)], om)
            xs = torch.full((NSAMP, 1, 1), a, dtype=torch.complex128)
            tsig.append(float(np.std(pdt(xs).reshape(-1).detach().numpy())))
        ratio = tsig[1] / max(tsig[0], 1e-300)
        tb.lt('OE.thermal_independent_of_I', abs(ratio - 1.0), 0.25,
              f"sigma ratio over a 1000x current change = {ratio:.4f} (must be ~1)")
        exp_t = 1e-12 * math.sqrt(BW)
        tb.close('OE.thermal_magnitude', tsig[0], exp_t, 0.25,
                 detail=f"inoise*sqrt(B) = {exp_t:.3e}")
    except Exception as e:
        tb.ok('OE.thermal_independent_of_I', False, f"inoise= not supported: {e!r}")

    # ---------- 4. dark current ------------------------------------------
    try:
        pdd = PDArray([dict(r0=R0, idark=1e-9)], om)
        zero = torch.zeros((4, 1, 1), dtype=torch.complex128)
        out = pdd(zero).reshape(-1).detach().numpy()
        tb.close('OE.dark_current', float(np.mean(out)), 1e-9, 0.05,
                 detail='with no light, output must be the dark current')
    except Exception as e:
        tb.ok('OE.dark_current', False, f"idark= not supported: {e!r}")

    # ---------- 5. reproducibility: local generator, not global RNG -------
    try:
        torch.manual_seed(1234)
        pdA = PDArray([dict(r0=R0, bw=BW, inoise=0.0)], om)
        xs = torch.full((256, 1, 1), math.sqrt(1e-4 / R0), dtype=torch.complex128)
        a1 = pdA(xs).reshape(-1).detach().numpy().copy()
        torch.manual_seed(9999)                      # perturb the GLOBAL rng
        pdB = PDArray([dict(r0=R0, bw=BW, inoise=0.0)], om)
        a2 = pdB(xs).reshape(-1).detach().numpy().copy()
        tb.ok('OE.independent_of_global_rng',
              bool(np.allclose(a1, a2)),
              'two PDs built with the same config must draw identically regardless of '
              'the global torch seed (i.e. use a local Generator seeded from config)')
    except Exception as e:
        tb.ok('OE.independent_of_global_rng', False, f"raised {e!r}")

    # ---------- 6. bandwidth low-pass ------------------------------------
    try:
        NT = 4096
        tt = torch.linspace(0, 1e-7, NT, dtype=torch.float64)
        for f_sig, expect_pass in ((1e7, True), (1e10, False)):
            env = 1.0 + 0.5 * torch.sin(2 * math.pi * f_sig * tt)
            xs = torch.sqrt(env).reshape(NT, 1, 1).to(torch.complex128)
            pdb = PDArray([dict(r0=R0, bw=1e9, inoise=0.0, dt=float(tt[1]-tt[0]))], om)
            o = pdb(xs).reshape(-1).detach().numpy()
            ac = float(np.std(o[NT // 4:]))
            dc = float(np.mean(o[NT // 4:]))
            ratio = ac / max(dc, 1e-30)
            if expect_pass:
                tb.ok('OE.bw_passes_inband', ratio > 0.1,
                      f"10 MHz modulation through a 1 GHz PD: ac/dc = {ratio:.4f}")
            else:
                tb.ok('OE.bw_rejects_outofband', ratio < 0.05,
                      f"10 GHz modulation through a 1 GHz PD: ac/dc = {ratio:.4f} "
                      f"(must be strongly attenuated)")
    except Exception as e:
        tb.ok('OE.bw_passes_inband', False, f"bw= low-pass not supported: {e!r}")

    # ---------- 7. backward compatibility ---------------------------------
    pd_old = PDArray([dict(r0=1e-3, std=0.0)], om)
    xs = torch.full((8, 1, 1), 2.0, dtype=torch.complex128)
    out = pd_old(xs).reshape(-1).detach().numpy()
    tb.lt('OE.legacy_std0_unchanged', float(np.max(np.abs(out - 1e-3 * 4.0))), 1e-15,
          'a legacy pd line with std=0 must give exactly r0*|E|^2')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        PDArray([dict(r0=1e-3, std=0.05)], om)(xs)
        msgs = ' | '.join(str(m.message).lower() for m in w)
    tb.ok('OE.legacy_std_warns', 'deprecat' in msgs or 'std' in msgs,
          f"legacy std= should warn once; saw: {msgs[:160]!r}")

    return tb


if __name__ == '__main__':
    main(build)
