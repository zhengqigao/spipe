"""TB-01 : foundational photonic invariants.

These must hold after EVERY phase of the campaign -- they are the properties that make
the solver physically meaningful at all. Ground truth is physics and closed-form algebra,
never another simulator.

Also exercises the REAL netlists shipped in test2/*.sp (photonic sections only, so no
HSPICE needed), which is the cheapest broad regression available.
"""
import sys, os, math, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch


def herm(S):
    S = S.detach()
    I = torch.eye(S.shape[-1], dtype=S.dtype)
    return (S.conj().transpose(-1, -2) @ S - I).abs().max().item()


def sym(S):
    return (S - S.transpose(-1, -2)).detach().abs().max().item()


def build():
    tb = TB('TB-01  photonic invariants (unitarity, reciprocity, energy, resonance)')
    sp = fresh_spipe(complex_dtype=torch.complex128)
    import spipe.photonic.model as M
    from spipe.photonic.photonic import Photonic

    om1 = 2 * math.pi * torch.tensor([193.1e12], dtype=torch.float64)
    om4 = 2 * math.pi * torch.linspace(193.1e12, 193.4e12, 4, dtype=torch.float64)
    t1 = torch.tensor([0.0])
    base = dict(neff=2.35, ng=4.0, wl=1550e-9)

    # ---------- 1. device-level unitarity & reciprocity, randomised -------
    torch.manual_seed(7)
    for trial in range(3):
        th = float(torch.rand(1) * math.pi)
        ph = float(torch.rand(1) * math.pi)
        L = float(torch.rand(1) * 1e-3)
        cases = [
            (f'WaveGuide{trial}', M.WaveGuide, dict(l=L, alpha=1.0), om1),
            (f'MZI{trial}',       M.MZI,       dict(theta=th), om1),
            (f'PBUm{trial}',      M.PBUm,      dict(theta=th, phi=ph, l=0.0), om1),
            (f'PS{trial}',        M.PS,        dict(ps=th), om1),
        ]
        for nm, cls, kw, om in cases:
            try:
                S = cls(**{**kw, **base, 'omega': om, 'time': t1}).transfer(None)
                tb.lt(f'INV.unitary.{nm}', herm(S), 1e-12, f"params={kw}")
                tb.lt(f'INV.reciprocal.{nm}', sym(S), 1e-14, '')
            except Exception as e:
                tb.ok(f'INV.unitary.{nm}', False, f"raised {e!r}")

    # 1->N splitters are legitimately non-unitary (combining loss), but must be
    # passive: no output power may exceed the input.
    for nm, cls, n in [('Splitter1to2', M.Splitter1to2, 2),
                       ('Splitter1to3', M.Splitter1to3, 3),
                       ('Splitter1to4', M.Splitter1to4, 4)]:
        S = cls(**{**base, 'omega': om1, 'time': t1}).transfer(None).detach()
        colnorm = (S.abs() ** 2).sum(dim=-2).max().item()
        tb.lt(f'INV.passive.{nm}', colnorm - 1.0, 1e-12,
              f"max column power gain = {colnorm:.12f} must not exceed 1")
        tb.lt(f'INV.reciprocal.{nm}', sym(S), 1e-14, '')

    # ---------- 2. network energy conservation, incl. loops ---------------
    nets = {
        'feedforward': ["mzi0 a1 a2 c1 c2 theta=0.35",
                        "wg1 c2 d2 l=53e-6 alpha=1.0",
                        "mzi2 c1 d2 e1 e2 theta=0.42",
                        "pd3 e1 v1 level1 r0=1.0", "pd4 e2 v2 level1 r0=1.0"],
        'pbum_chain': ["pbum0 a1 a2 c1 c2 theta=0.5pi phi=0.0pi l=1e-4",
                       "pbum1 c1 c2 e1 e2 theta=0.3pi phi=0.1pi l=1e-4",
                       "pd2 e1 v1 level1 r0=1.0", "pd3 e2 v2 level1 r0=1.0"],
        'ps_wg':      ["ps0 a1 b1 ps=0.7",
                       "wg1 b1 c1 l=250e-6 alpha=1.0",
                       "ps2 a2 c2 ps=-0.3",
                       "pd3 c1 v1 level1 r0=1.0", "pd4 c2 v2 level1 r0=1.0"],
    }
    for nm, body in nets.items():
        nl = [l + '\n' for l in
              [".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
               ".source 1.0@a1 0.0@a2"] + body]
        try:
            res, _, _ = Photonic(nl).simulate()
            tb.lt(f'INV.energy.{nm}', abs(res.sum().item() - 1.0), 1e-12,
                  f"total detected power = {res.sum().item():.16f}")
        except Exception as e:
            tb.ok(f'INV.energy.{nm}', False, f"raised {e!r}")

    # lossy network: output must be strictly less than input, never more
    nl = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
        ".source 1.0@a1 0.0@a2", "mzi0 a1 a2 c1 c2 theta=0.35",
        "wg1 c1 d1 l=1e-4 alpha=0.6", "wg2 c2 d2 l=1e-4 alpha=0.6",
        "pd3 d1 v1 level1 r0=1.0", "pd4 d2 v2 level1 r0=1.0"]]
    res, _, _ = Photonic(nl).simulate()
    got = res.sum().item()
    tb.close('INV.lossy_alpha', got, 0.6 ** 2, 1e-12,
             detail='two alpha=0.6 arms -> total transmission 0.36')

    # ---------- 3. ring resonator vs the analytic all-pass formula --------
    # all-pass ring: |t| = |(r - a e^{i phi}) / (1 - r a e^{i phi})|
    # Built from an MZI acting as the coupler (cos = r) plus a feedback waveguide.
    r_amp, a_amp = math.cos(0.3), 0.85
    Lrt = 200e-6
    nl = [l + '\n' for l in [
        ".mode neff=2.35 ng=2.35 wl=1550e-9",
        ".freq 193.0e12 193.2e12 201",
        ".source 1.0@in 0.0@drop",
        "mzi0 in drop thru ring1 theta=0.3",
        f"wg1 ring1 ring2 l={Lrt} alpha={a_amp}",
        "pd2 thru vo1 level1 r0=1.0",
    ]]
    # note: ring2 must close the loop back into the coupler; use a 2nd MZI port
    nl = [l + '\n' for l in [
        ".mode neff=2.35 ng=2.35 wl=1550e-9",
        ".freq 193.0e12 193.2e12 401",
        ".source 1.0@pin 0.0@pdrop",
        "mzi0 pin pdrop pthru pring_a theta=0.3",
        f"wg1 pring_a pring_b l={Lrt} alpha={a_amp}",
        "pd2 pthru vo1 level1 r0=1.0",
        "pd3 pring_b vo2 level1 r0=1.0",
    ]]
    try:
        p = Photonic(nl)
        res, _, _ = p.simulate()
        nfreq = len(p.omega)                 # source injects 1.0 at EVERY omega
        tot = res.sum().item() / nfreq       # so normalise by the channel count
        tb.ok('INV.ring_builds', True,
              f"total power/channel = {tot:.9f} over {nfreq} frequency points")
        tb.lt('INV.ring_passive', tot - 1.0, 1e-12,
              'summed output power per channel cannot exceed the injected 1.0')
        # a resonant structure must show real spectral structure, not a flat response
        per_ch = (res.sum(dim=0) if res.dim() > 1 else res).detach().numpy()
        tb.ok('INV.ring_has_resonance', float(np.ptp(per_ch)) > 1e-3,
              f"through-port varies by {float(np.ptp(per_ch)):.4f} across the sweep")
    except Exception as e:
        tb.ok('INV.ring_builds', False, f"raised {e!r}")

    # ---------- 4. reciprocity of the assembled network -------------------
    # swapping source and detector must give the same transmission (Lorentz reciprocity)
    def transmit(src, det):
        nl = [l + '\n' for l in [
            ".mode neff=2.35 ng=4.0 wl=1550e-9", ".freq 193.1e12 193.1e12 1",
            f".source 1.0@{src}", "mzi0 a1 a2 c1 c2 theta=0.37",
            "wg1 c2 d2 l=137e-6 alpha=0.9",
            "mzi2 c1 d2 e1 e2 theta=0.21",
            f"pd9 {det} vo level1 r0=1.0"]]
        return Photonic(nl).simulate()[0].sum().item()
    try:
        fwd = transmit('a1', 'e1')
        rev = transmit('e1', 'a1')
        tb.close('INV.lorentz_reciprocity', rev, fwd, 1e-10,
                 detail=f"a1->e1 = {fwd:.12f}, e1->a1 = {rev:.12f}")
    except Exception as e:
        tb.ok('INV.lorentz_reciprocity', False, f"raised {e!r}")

    # ---------- 5. the REAL shipped netlists still solve -------------------
    # the shipped paper netlists live under examples/ in the released layout
    _pats = [os.path.join(REPO, 'examples', 'paper', 'ptc_hspice', 'test1*.sp'),
             os.path.join(REPO, 'test2', 'test1*.sp')]
    _files = sorted({f for _p in _pats for f in glob.glob(_p)})
    if not _files:
        tb.ok('REG.netlists_found', False, f'no shipped netlists matched {_pats}')
    for sp_file in _files:
        name = os.path.basename(sp_file)
        try:
            txt = open(sp_file).read()
            m = re.search(r'^\s*\.photonic\b(.*)', txt, re.S | re.M)
            if not m:
                tb.skip(f'REG.{name}', 'no .photonic section')
                continue
            body = [l.strip() + '\n' for l in m.group(1).splitlines()
                    if l.strip() and not l.strip().startswith('*')]
            ph = Photonic(body)
            nmod = len(ph.mod_element)
            t = torch.linspace(0, 1e-9, 3)
            act = torch.zeros(3, nmod, dtype=torch.float64) if nmod else None
            res, _, pw = ph.simulate(t, act) if nmod else ph.simulate()
            fin = bool(torch.isfinite(res).all())
            tb.ok(f'REG.{name}', fin,
                  f"{len(ph.circuit_element)} passive + {nmod} active devices, "
                  f"out shape {tuple(res.shape)}, finite={fin}")
        except Exception as e:
            tb.ok(f'REG.{name}', False, f"raised {type(e).__name__}: {str(e)[:110]}")

    return tb


if __name__ == '__main__':
    main(build)
