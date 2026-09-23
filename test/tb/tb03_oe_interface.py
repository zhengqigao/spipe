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

    # ---------- 8. the detector options must survive the Photonic wiring ---
    # Everything above builds a PDArray by hand and hands it a time axis. Photonic is the
    # only thing users actually go through, and it used to construct the array as
    # PDArray(pd_args, omega) and call it as pd_array(res) -- no time axis -- so BOTH the
    # bw= low-pass and coherent=1 silently degraded behind a warning. These checks go
    # through a netlist, which is the path that was broken.
    from spipe.photonic.photonic import Photonic

    sp.config['quasistatic_check'] = False
    NT, DT, SPACING, BW = 4096, 2.5e-12, 1e11, 2e8
    tgrid = torch.arange(NT, dtype=torch.float64) * DT
    TAU = 1.0 / (2 * math.pi * BW)

    def through_photonic(pd_line, modulated=False):
        """Run a netlist and return (photocurrent of pd1, warnings raised).

        ``modulated`` drives the MZM at 5 GHz, i.e. 25x above the 0.2 GHz detector pole,
        so a working low-pass must visibly flatten the photocurrent. A constant drive
        gives a flat photocurrent, which the filter cannot change -- that would make the
        check vacuous.
        """
        deck = [".mode neff=2.35 ng=4.0 wl=1550e-9",
                f".freq 193.1e12 {193.1e12 + 2 * SPACING:.6e} 3",
                ".source 1.0@a1 0.0@a2",
                "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
                pd_line,
                "pd2 b2 vo2 level1 r0=1.0"]
        if modulated:
            drive = (1.0 + torch.sin(2 * math.pi * 5e9 * tgrid)).reshape(-1, 1)
        else:
            drive = torch.full((NT, 1), 1.0, dtype=torch.float64)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = Photonic([l + '\n' for l in deck]).simulate(tgrid, drive)[0][:, 0]
        return out, ' | '.join(str(m.message) for m in w)

    try:
        plain, _ = through_photonic("pd1 b1 vo1 level1 r0=1.0", modulated=True)
        limited, wbw = through_photonic(f"pd1 b1 vo1 level1 r0=1.0 bw={BW:g}", modulated=True)
        # The coherent pair is run on a CONSTANT drive, so the only thing that can differ
        # between them is the treatment of the three WDM carriers -- not the modulation.
        cw_incoherent, _ = through_photonic(f"pd1 b1 vo1 level1 r0=1.0 bw={BW:g}")
        coherent, wco = through_photonic(f"pd1 b1 vo1 level1 r0=1.0 bw={BW:g} coherent=1")

        tb.ok('OE.photonic_gives_pd_the_time_axis',
              'time axis is not available' not in wbw
              and 'time axis is not available' not in wco,
              'Photonic must hand PDArray the transient grid; the fallback warning means '
              f'bw=/coherent= were silently ignored. saw: {(wbw + wco)[:150]!r}')

        # The low-pass must actually do something: a 5 GHz modulation is 25x above the
        # 0.2 GHz pole, so a working filter has to flatten it by roughly that factor. An
        # unchanged peak-to-peak means the low-pass was skipped -- which is what the
        # missing time axis used to cause.
        settle = tgrid > 5 * TAU
        p2p_plain = float(plain[settle].max() - plain[settle].min())
        p2p_limited = float(limited[settle].max() - limited[settle].min())
        tb.ok('OE.photonic_bw_lowpass_applied',
              p2p_plain > 0.1 and p2p_limited < 0.2 * p2p_plain,
              f'5 GHz modulation, peak-to-peak {p2p_plain:.6f} (no bw=) -> '
              f'{p2p_limited:.6f} (bw={BW:g}); the drive must be visibly modulated in the '
              f'first place, and the pole must reject it')

        # coherent=1 must change the answer at all -- it used to return the incoherent sum.
        tb.ok('OE.photonic_coherent_applied',
              float((coherent - cw_incoherent).abs().max()) > 1e-6,
              f'coherent=1 must differ from the incoherent sum on the same drive: max dev '
              f'{float((coherent - cw_incoherent).abs().max()):.4e}')

        # ...and it must reduce to the incoherent sum once the beat notes (here 1e11 Hz,
        # 500x the detector bandwidth) are filtered out and the pole has settled. The
        # filter starts from the fully-constructive t=0 sample, so this is only true after
        # several time constants -- which is the physics, not an approximation of it.
        settled = tgrid > 5 * TAU
        tb.ok('OE.coherent_settles_to_incoherent', bool(settled.any()), 
              f'record {NT * DT * 1e9:.1f} ns must span > 5 tau = {5 * TAU * 1e9:.1f} ns')
        tb.lt('OE.coherent_reduces_to_incoherent',
              float((coherent[settled] - cw_incoherent[settled]).abs().max()), 0.05,
              f'beat notes at {SPACING:g} Hz are {SPACING / BW:.0f}x above bw and must be '
              f'rejected, leaving the incoherent sum')
    except Exception as e:
        tb.ok('OE.photonic_gives_pd_the_time_axis', False, f"{e!r}")

    return tb


if __name__ == '__main__':
    main(build)
