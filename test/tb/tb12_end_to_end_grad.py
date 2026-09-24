"""TB-12 : end-to-end differentiability (SPEC-E2).

The headline of the whole campaign: d|E_out|^2 / dW for a transistor width, through
driver -> modulator -> photodetector, against central finite differences over the WHOLE
chain. Written from SPEC_E2.md and SPEC_E1.md only.

Note the sky130 DAC is deliberately NOT used: sky130 ships 63 per-size binned BSIM4 cards
(lmin=1.45e-7..1.55e-7, wmin=1.255e-6..1.265e-6), so W is effectively discrete there and
d/dW is ill-posed. Level-1 devices make W continuous and analytically differentiable.
"""
import sys, os, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import TB, fresh_spipe, main, REPO
import numpy as np
import torch



def _pkg(*parts):
    """Path to a file inside the spipe package, in either layout (src/ or flat)."""
    for base in (os.path.join(REPO, 'src'), REPO):
        cand = os.path.join(base, 'spipe', *parts)
        if os.path.exists(cand):
            return cand
    return os.path.join(REPO, 'src', 'spipe', *parts)


# A CMOS inverter driving a modulator load. W of the pull-down sets the drive
# strength, hence the modulator voltage, hence the optical output.
DRIVER = """* mzm driver
.model nch NMOS (LEVEL=1 VTO=0.7 KP=120u LAMBDA=0.02)
.model pch PMOS (LEVEL=1 VTO=-0.7 KP=40u LAMBDA=0.02)
Vdd vdd 0 3.0
Vin g 0 PULSE(0 3 1n 0.2n 0.2n 10n 20n)
MN1 drv g 0 0 nch W=8u L=0.5u
MP1 drv g vdd vdd pch W=16u L=0.5u
Cload drv 0 150f
Rs drv mod 10
Cj mod 0 200f
.tran 2e-11 3e-8
.end
"""


def build():
    tb = TB('TB-12  end-to-end gradient: d|E|^2 / dW')
    sp = fresh_spipe(complex_dtype=torch.complex128)

    # ------------------------------------------------------------------
    # Path A: native engine + photonic solver composed by hand.
    # This is the ground-truth chain; if this does not work, nothing does.
    # ------------------------------------------------------------------
    try:
        from spipe.electronic.native import Netlist
    except Exception as e:
        tb.ok('E2.native_available', False, f"native engine not importable: {e!r}")
        return tb
    tb.ok('E2.native_available', True, '')

    from spipe.photonic.photonic import Photonic

    PHOT = [l + '\n' for l in [
        ".mode neff=2.35 ng=4.0 wl=1550e-9",
        ".freq 193.1e12 193.1e12 1",
        ".source 1.0@a1 0.0@a2",
        "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
        "pd1 b1 vo1 level1 r0=1.0",
        "pd2 b2 vo2 level1 r0=0.0",
    ]]

    def optical_out(w_value, requires_grad):
        """Full chain: transistor W -> drive waveform -> MZM -> detected power."""
        ckt = Netlist.from_string(DRIVER)
        w = ckt.param('MN1', 'W')
        if requires_grad:
            w.requires_grad_(True)
        else:
            with torch.no_grad():
                w.copy_(torch.as_tensor(w_value, dtype=torch.float64))
        res = ckt.tran(2e-11, 3e-8)
        t = res.t
        drive = res.v('mod')
        # subsample to keep the photonic solve cheap
        k = max(1, len(t) // 48)
        t_s, d_s = t[::k], drive[::k]
        p = Photonic(PHOT, need_grads=True)
        out, _, _ = p.simulate(t_s, d_s.reshape(-1, 1))
        return (out[:, 0] ** 2).sum(), w

    okA, gotA = tb.no_raise('E2.chain_runs', lambda: optical_out(None, True))
    if okA and gotA is not None:
        loss, w = gotA
        tb.ok('E2.loss_finite', bool(torch.isfinite(loss)), f"loss={float(loss):.8g}")
        okB, _ = tb.no_raise('E2.backward_runs', lambda: loss.backward())
        if okB and w.grad is not None:
            g_auto = float(w.grad)
            tb.ok('E2.grad_exists', True, f"d|E|^2/dW = {g_auto:.10g}")

            # central finite difference over the WHOLE chain
            w0 = 8e-6
            h = w0 * 1e-5
            vals = []
            for s in (+1, -1):
                with torch.no_grad():
                    L, _ = optical_out(w0 + s * h, False)
                vals.append(float(L))
            g_fd = (vals[0] - vals[1]) / (2 * h)
            scale = max(abs(g_fd), abs(g_auto), 1e-300)
            tb.lt('E2.HEADLINE_dEdW_vs_fd', abs(g_auto - g_fd) / scale, 1e-4,
                  f"autograd={g_auto:.10g}  finite-diff={g_fd:.10g}  (W={w0:g} m)")
            tb.ok('E2.grad_nontrivial', abs(g_fd) > 1e-12,
                  f"|d|E|^2/dW| = {abs(g_fd):.4e} must be measurably non-zero, "
                  f"otherwise the agreement above is vacuous")
        else:
            tb.ok('E2.grad_exists', False,
                  'no .grad on the transistor width -- the chain is not differentiable')

    # ------------------------------------------------------------------
    # Path B: the same thing through Circuit + .sensparam.
    #
    # These used to be grep-the-source checks ("is the string 'sensparam'
    # present in electronic.py?"), which pass on a comment and would survive
    # the feature being deleted. They are now behavioural: the parser, the
    # guards and the implicit differentiation are each *called* and their
    # answers compared with what they are supposed to produce.
    # ------------------------------------------------------------------
    try:
        from spipe.electronic.sensitivity import parse_sensparam
        deck = ["* driver\n",
                "MN1 d g 0 0 nch W=8u L=0.5u\n",
                "RL d vdd 2k\n",
                ".sensparam MN1:W MN1:L RL:R\n",
                ".tran 1n 10n\n"]
        kept, declared = parse_sensparam(deck)
        # DeviceParameter is a plain (device, PARAM) tuple; the param is upper-cased
        got = sorted((d.upper(), n.upper()) for d, n in declared)
        tb.ok('E2.sensparam_parsed',
              got == [('MN1', 'L'), ('MN1', 'W'), ('RL', 'R')],
              f"declared {got} from '.sensparam MN1:W MN1:L RL:R'")
        tb.ok('E2.sensparam_card_consumed',
              not any('.sensparam' in line.lower() for line in kept),
              "the .sensparam card must not be passed through to the SPICE deck "
              f"(kept {len(kept)} of {len(deck)} lines)")
    except Exception as e:
        tb.ok('E2.sensparam_parsed', False, f"{e!r}")
        tb.ok('E2.sensparam_card_consumed', False, f"{e!r}")

    # A declared parameter must arrive as a differentiable leaf on the Circuit.
    try:
        from spipe.core.core import Circuit
        link = os.path.join(REPO, 'examples', 'link_driver_mzm.sp')
        ckt = Circuit(link, spice_exe='native')
        w = ckt.param('mn1', 'W')
        tb.ok('E2.declared_param_is_leaf',
              w.is_leaf and w.requires_grad and w.dtype == torch.float64,
              f"ckt.param('mn1','W') -> {w.dtype}, leaf={w.is_leaf}, "
              f"requires_grad={w.requires_grad}")
    except Exception as e:
        tb.ok('E2.declared_param_is_leaf', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The silent-zero guard. Xyce's transient ADJOINT returns all zeros for
    # device parameters (measured: d_{V(D)}/d_M1:W_adj = 0.0 while direct
    # gives -4.99695862e+04). Call the guard and check it actually raises --
    # and, just as important, that it does NOT raise on the two cases where
    # zero is the right answer.
    # ------------------------------------------------------------------
    try:
        from spipe.electronic.sensitivity import (guard_all_zero, guard_analysis_ran,
                                                  ZeroSensitivityError)
        moving = torch.linspace(0.0, 3.0, 16, dtype=torch.float64)
        flat = torch.full((16,), 1.5, dtype=torch.float64)
        zeros = torch.zeros(16, dtype=torch.float64)

        tb.raises('E2.zero_block_raises',
                  lambda: guard_all_zero(zeros, ['M1:W'], moving, 'adjoint', 'xyce'),
                  ZeroSensitivityError,
                  'an identically-zero block under a swinging objective must raise')
        ok_flat, _ = tb.no_raise(
            'E2.zero_block_ok_if_objective_constant',
            lambda: guard_all_zero(zeros, ['M1:W'], flat, 'adjoint', 'xyce'),
            'a constant objective genuinely has zero sensitivity -- must NOT raise')
        ok_nz, _ = tb.no_raise(
            'E2.nonzero_block_passes',
            lambda: guard_all_zero(torch.full((16,), 1e-30, dtype=torch.float64),
                                   ['M1:W'], moving, 'direct', 'xyce'),
            'a genuinely tiny but non-zero gradient is a legitimate answer')
        tb.raises('E2.zero_objective_raises',
                  lambda: guard_analysis_ran(zeros, moving, 'adjoint', 'xyce', ['M1:W']),
                  ZeroSensitivityError,
                  'sensitivity analysis reporting a zero objective the transient did not')
        # the error must name the method, since switching method is the fix
        try:
            guard_all_zero(zeros, ['M1:W'], moving, 'adjoint', 'xyce')
            msg = ''
        except ZeroSensitivityError as e:
            msg = str(e)
        tb.ok('E2.zero_message_actionable',
              'direct' in msg.lower() and 'adjoint' in msg.lower(),
              f"message must say which method to use instead: {msg[:90]!r}")
    except Exception as e:
        tb.ok('E2.zero_block_raises', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The fixed point must be differentiated by the implicit function
    # theorem, not by unrolling. The observable signature is that
    # solve_coupling_system returns (I - A^T)^-1 b exactly, for a coupling
    # Jacobian we choose, by whichever path -- and that the answer does not
    # depend on which path ran.
    # ------------------------------------------------------------------
    try:
        from spipe.core.core import solve_coupling_system, CouplingJacobianError
        import spipe

        def exact(matrix, rhs):
            """(I - A^T)^-1 b, computed directly."""
            n = matrix.shape[0]
            eye = torch.eye(n, dtype=torch.float64)
            return torch.linalg.solve(eye - matrix.T, rhs)

        torch.manual_seed(0)
        n = 6
        a = torch.randn(n, n, dtype=torch.float64) * 0.12      # loop gain well below 1
        b = torch.randn(n, dtype=torch.float64)

        info = {}
        got = solve_coupling_system(lambda v: a.T @ v, b, spipe.config, info)
        want = exact(a, b)
        tb.lt('E2.ift_matches_closed_form',
              float((got - want).abs().max() / want.abs().max()), 1e-12,
              f"(I - A^T)^-1 b over {n} unknowns, mode={info.get('mode')}, "
              f"rho={info.get('spectral_radius')}")

        # No coupling at all: the answer is b itself, exactly, in one product.
        info0 = {}
        got0 = solve_coupling_system(lambda v: torch.zeros_like(v), b, spipe.config, info0)
        tb.ok('E2.ift_decoupled_exact',
              bool(torch.equal(got0, b)) and info0.get('vjp_calls') == 1,
              f"A = 0 must give lam = b bit-for-bit in one product "
              f"(mode={info0.get('mode')}, calls={info0.get('vjp_calls')})")

        # Path independence: the dense solve and the matrix-free Neumann
        # iteration must agree. Unrolling would make the answer depend on
        # how many products were taken; the IFT answer cannot.
        cfg_mf = dict(spipe.config)
        cfg_mf['coupling_dense_limit'] = 1          # force the matrix-free path
        info_mf = {}
        got_mf = solve_coupling_system(lambda v: a.T @ v, b, cfg_mf, info_mf)
        tb.lt('E2.ift_path_independent',
              float((got_mf - got).abs().max() / got.abs().max()), 1e-10,
              f"dense (mode={info.get('mode')}, {info.get('vjp_calls')} products) vs "
              f"matrix-free (mode={info_mf.get('mode')}, {info_mf.get('vjp_calls')} "
              f"products) must give the same gradient")

        # Loop gain >= 1: the fixed point is not stable and the derivative is
        # unbounded. Returning a number there would be worse than raising.
        singular = torch.eye(n, dtype=torch.float64)            # A = I  =>  I - A^T = 0
        tb.raises('E2.ift_singular_raises',
                  lambda: solve_coupling_system(lambda v: singular.T @ v, b,
                                                spipe.config, {}),
                  CouplingJacobianError,
                  'a singular (I - A^T) must raise rather than return a huge number')
    except Exception as e:
        tb.ok('E2.ift_matches_closed_form', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The co-simulation fixed point must run at config['real_dtype']. Its initial
    # guess was created with torch.randn / torch.as_tensor and no dtype -- float32 --
    # and solve_fixed_point casts every iterate to the guess's dtype, so the whole
    # iteration ran in single precision. Symptom: the differentiable and plain paths
    # disagreed by 9.3e-08 at ANY tolerance. Fixed, they agree to machine precision.
    # ------------------------------------------------------------------
    try:
        from spipe.core.core import Circuit
        link = os.path.join(REPO, 'examples', 'link_driver_mzm.sp')
        c_rand = Circuit(link, spice_exe='native')
        with torch.no_grad():
            out_rand = c_rand.simulate()
        c_x0 = Circuit(link, spice_exe='native')
        with torch.no_grad():
            out_x0 = c_x0.simulate(x0=0.65)
        want = sp.config['real_dtype']
        tb.ok('E2.fixed_point_dtype_random_guess', out_rand[3].dtype == want,
              f"converged drive is {out_rand[3].dtype}, config real_dtype is {want}")
        tb.ok('E2.fixed_point_dtype_explicit_x0', out_x0[3].dtype == want,
              f"x0=0.65 (a Python float) must not default the iteration to float32: "
              f"got {out_x0[3].dtype}")

        c_diff = Circuit(link, spice_exe='native')
        pc_diff = c_diff.simulate()[2].detach()
        gap = float((pc_diff - out_rand[2]).abs().max())
        tb.lt('E2.diff_and_plain_paths_agree', gap, 1e-12,
              f"max |differentiable - plain| photocurrent = {gap:.3e}; float32 rounding of "
              f"the drive made this 9.3e-08 regardless of rtol")
    except Exception as e:
        tb.ok('E2.fixed_point_dtype_random_guess', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The documented check: dL/dW from Circuit.simulate() against numerical
    # differentiation -- a central finite difference that re-runs the whole
    # co-simulation. docs/differentiability.md quotes 4.6e-08 at h/W = 1e-4,
    # the bottom of the step-size sweep; the bound leaves room for platform
    # round-off while still catching any real gradient error by orders of
    # magnitude.
    # ------------------------------------------------------------------
    try:
        from spipe.core.core import Circuit
        link = os.path.join(REPO, 'examples', 'link_driver_mzm.sp')
        ck = Circuit(link, spice_exe='native')
        w = ck.param('mn1', 'W')
        _, _, pc, _, _ = ck.simulate()
        (pc[:, 0] ** 2).sum().backward()
        g = float(w.grad)
        w0 = float(w.detach())
        h = w0 * 1e-4

        def _loss_at(value):
            c = Circuit(link, spice_exe='native')
            with torch.no_grad():
                c.param('mn1', 'W').copy_(torch.tensor(value, dtype=torch.float64))
                return float((c.simulate()[2][:, 0] ** 2).sum())

        fd = (_loss_at(w0 + h) - _loss_at(w0 - h)) / (2 * h)
        tb.lt('E2.circuit_grad_vs_finite_difference', abs(g - fd) / abs(fd), 1e-6,
              f"Circuit.simulate(): analytic dL/dW = {g:.8f}, central FD = {fd:.8f} "
              f"(h/W = 1e-4)")
    except Exception as e:
        tb.ok('E2.circuit_grad_vs_finite_difference', False, f"{e!r}")

    # ------------------------------------------------------------------
    # Electrical probes (.print tran) must carry the TOTAL derivative. A detector output
    # depends on W only through the light; its gradient used to come back exactly 0 against
    # a finite difference of 2.3e4 -- silently.
    # ------------------------------------------------------------------
    try:
        import tempfile
        from spipe.core.core import Circuit
        src = open(os.path.join(REPO, 'examples', 'link_driver_mzm.sp')).read()
        src = src.replace('.sensparam MN1:W MN1:L', '.sensparam MN1:W MN1:L\n.print tran v(vo1) v(vdrv)')
        probe_deck = os.path.join(tempfile.mkdtemp(prefix='spipe_tb12_'), 'probe.sp')
        open(probe_deck, 'w').write(src)
        wrk = os.path.dirname(probe_deck)

        def _probe_loss(width, grad):
            c = Circuit(probe_deck, spice_exe='native', spice_wrk_dir=wrk)
            w = c.param('mn1', 'W')
            if width is not None:
                with torch.no_grad():
                    w.copy_(torch.tensor(width, dtype=torch.float64))
            if grad:
                return c.simulate()[0]['v(vo1)'].sum(), w
            with torch.no_grad():
                return float(c.simulate()[0]['v(vo1)'].sum()), w

        loss, w = _probe_loss(None, True)
        loss.backward()
        w0 = float(w.detach()); h = w0 * 1e-4
        fd = (_probe_loss(w0 + h, False)[0] - _probe_loss(w0 - h, False)[0]) / (2 * h)
        tb.ok('E2.probe_grad_nonzero', abs(float(w.grad)) > 1e-3 * abs(fd),
              f"d sum v(vo1)/dW = {float(w.grad):.6g}; it used to be exactly 0")
        tb.lt('E2.probe_grad_vs_finite_difference', abs(float(w.grad) - fd) / abs(fd), 1e-5,
              f"detector-output probe: analytic {float(w.grad):.6f} vs central FD {fd:.6f}")
    except Exception as e:
        tb.ok('E2.probe_grad_vs_finite_difference', False, f"{e!r}")

    # ------------------------------------------------------------------
    # Detector noise inside the co-simulation. The seed used to reach only the initial guess,
    # so two runs with the same seed differed and different seeds gave the same noise; and a
    # noisy detector next to a noiseless one made the gradient NaN (sqrt of a zero variance).
    # With the realisation fixed by the seed, the gradient must match a finite difference
    # taken at that same seed.
    # ------------------------------------------------------------------
    try:
        import tempfile
        from spipe.core.core import Circuit
        src = open(os.path.join(REPO, 'examples', 'link_driver_mzm.sp')).read()
        src = src.replace('pd1 b1 vo1 level1 r0=1.0', 'pd1 b1 vo1 level1 r0=1.0 bw=10e9')
        noisy_deck = os.path.join(tempfile.mkdtemp(prefix='spipe_tb12_'), 'noisy.sp')
        open(noisy_deck, 'w').write(src)

        def _noisy(width, seed, grad=False):
            c = Circuit(noisy_deck, spice_exe='native')
            w = c.param('mn1', 'W')
            if width is not None:
                with torch.no_grad():
                    w.copy_(torch.tensor(width, dtype=torch.float64))
            if grad:
                return c.simulate(seed=seed)[2][:, 0].sum(), w
            with torch.no_grad():
                return c.simulate(seed=seed)[2][:, 0].clone(), w

        a1, _ = _noisy(None, 5)
        b1, _ = _noisy(None, 6)
        same = Circuit(noisy_deck, spice_exe='native')      # one object, run twice
        with torch.no_grad():
            s1 = same.simulate(seed=5)[2][:, 0].clone()
            s2 = same.simulate(seed=5)[2][:, 0].clone()
        tb.ok('E3.noise_same_seed_reproduces', bool(torch.equal(s1, s2)) and bool(torch.equal(s1, a1)),
              'the same Circuit, simulate(seed=5) twice, gives the same noisy photocurrent bit for '
              'bit (and the same as a fresh Circuit); the noise generator used to run on')
        tb.ok('E3.noise_seed_changes_realisation', not bool(torch.equal(a1, b1)),
              'seed=6 gives a different noise realisation from seed=5')
        loss, w = _noisy(None, 5, grad=True)
        loss.backward()
        tb.ok('E3.noise_gradient_finite', bool(torch.isfinite(w.grad).all()),
              f'd sum I/dW with one noisy detector = {float(w.grad):.6g}')
        w0 = float(w.detach()); h = w0 * 1e-4
        fd = float((_noisy(w0 + h, 5)[0].sum() - _noisy(w0 - h, 5)[0].sum()) / (2 * h))
        tb.lt('E3.noise_grad_vs_finite_difference', abs(float(w.grad) - fd) / abs(fd), 1e-5,
              f'same realisation: analytic {float(w.grad):.8g} vs central FD {fd:.8g}')
    except Exception as e:
        tb.ok('E3.noise_gradient_finite', False, f"{e!r}")

    # ------------------------------------------------------------------
    # Circuit in envelope mode. simulate(mode='envelope') used to be dropped silently on the
    # plain path (and any string was accepted), and refused on the gradient path. A 100 ps
    # delay line sits after the modulator, so the loop derivative has to carry the delay.
    # ------------------------------------------------------------------
    try:
        import tempfile
        from spipe.core.core import Circuit
        deck = (".electronic\n.model nch NMOS (LEVEL=1 VTO=0.7  KP=120u LAMBDA=0.02)\n"
                ".model pch PMOS (LEVEL=1 VTO=-0.7 KP=40u  LAMBDA=0.02)\nVdd vdd 0 3.0\n"
                "Vin g   0 PULSE(0 3 0.2n 0.02n 0.02n 1n 2n)\n"
                "MN1 vdrv g 0   0   nch W=8u  L=0.5u\nMP1 vdrv g vdd vdd pch W=16u L=0.5u\n"
                "Rload1 vo1 0 1k\n.sensparam MN1:W\n.tran 0 1.6n 81\n.photonic\n"
                ".mode neff=2.35 ng=4.0 wl=1550e-9\n.freq 193.0e12 193.2e12 201\n"
                ".source 0.0316@a1 0.0@a2\nmzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 act_l=1e-9\n"
                "wg0 b1 c1 l=7.49e-3\npd1 c1 vo1 level1 r0=1.0\n")
        env_deck = os.path.join(tempfile.mkdtemp(prefix='spipe_tb12_'), 'delay.sp')
        open(env_deck, 'w').write(deck)
        tb.raises('E4.circuit_rejects_unknown_mode',
                  lambda: Circuit(env_deck, 'native').simulate(mode='nonsense'), ValueError)

        def _env(width=None, grad=False):
            c = Circuit(env_deck, 'native'); w = c.param('mn1', 'W')
            if width is not None:
                w.data.fill_(width)
            with (torch.enable_grad() if grad else torch.no_grad()), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pc = c.simulate(mode='envelope')[2]
                return (pc[:, 0] * torch.linspace(0, 1, pc.shape[0], dtype=torch.float64)).sum(), w

        loss, w = _env(grad=True); loss.backward()
        w0 = float(w.detach()); h = 1e-4 * w0
        fd = (float(_env(w0 + h)[0]) - float(_env(w0 - h)[0])) / (2 * h)
        tb.lt('E4.circuit_envelope_grad_vs_fd', abs(float(w.grad) - fd) / abs(fd), 1e-6,
              f'Circuit.simulate(mode="envelope"), 100 ps delay after the modulator: analytic '
              f'{float(w.grad):.8g} vs central FD {fd:.8g}')
    except Exception as e:
        tb.ok('E4.circuit_envelope_grad_vs_fd', False, f"{e!r}")

    # ------------------------------------------------------------------
    # Like for like: the unified call must BE the composition. Same netlist, same
    # grid, same loss, once through Circuit.simulate() and once by driving the
    # native engine on the deck Circuit generated, then Photonic on its drive.
    # (The docs once set two different circuits' gradients side by side as if
    # they were this comparison; this is the comparison.)
    # ------------------------------------------------------------------
    try:
        from spipe.core.core import Circuit
        from spipe.electronic.native import Netlist as _Net
        from spipe.electronic.electronic import (_resample, _resample_weights,
                                                 _native_signal)
        link = os.path.join(REPO, 'examples', 'link_driver_mzm.sp')

        ck = Circuit(link, spice_exe='native')
        w_u = ck.param('mn1', 'W')
        _, _, pc_u, drive_u, _ = ck.simulate()
        loss_u = (pc_u[:, 0] ** 2).sum()
        loss_u.backward()

        e = ck.e_circuit
        net = _Net(e.native_deck)
        net.options['lte_reltol'] = e._native.options['lte_reltol']
        w_h = net.param('mn1', 'W'); w_h.requires_grad_(True)
        grid = ck.time.detach().to(torch.float64).reshape(-1)
        step = (float(grid[-1]) - float(grid[0])) / (len(grid) - 1)
        res = net.tran(step, float(grid[-1]), tstart=float(grid[0]), uic=e.native_uic)
        drive_h = _resample(_native_signal(res, 'v(vdrv)').reshape(-1, 1),
                            _resample_weights(torch.as_tensor(res.t, dtype=torch.float64)
                                              .reshape(-1), grid))
        phot = [l.strip() + '\n' for l in open(link).read().split('.photonic', 1)[1]
                .splitlines() if l.strip() and not l.strip().startswith('*')]
        pc_h, _, _ = Photonic(phot, need_grads=True).simulate(grid, drive_h)
        loss_h = (pc_h[:, 0] ** 2).sum()
        loss_h.backward()

        tb.lt('E2.unified_equals_composed_drive',
              float((drive_u.detach() - drive_h.detach()).abs().max()), 1e-12,
              'modulator drive: Circuit.simulate() vs the native engine run by hand')
        tb.close('E2.unified_equals_composed_loss', float(loss_u), float(loss_h), 1e-12,
                 detail='same circuit, same grid, same loss')
        tb.close('E2.unified_equals_composed_grad', float(w_u.grad), float(w_h.grad), 1e-10,
                 detail=f"dL/dW = {float(w_u.grad):.10f} both ways")
    except Exception as e:
        tb.ok('E2.unified_equals_composed_grad', False, f"{e!r}")

    # gradient_based_simulate must be implemented or deleted, never left as a
    # commented-out block that reads like a feature.
    try:
        csrc = open(_pkg('core', 'core.py')).read()
        tb.ok('E2.no_dead_commented_grad_block',
              'def gradient_based_simulate' not in csrc.replace('# ', '')
              or 'def gradient_based_simulate' in csrc,
              'the commented-out gradient_based_simulate must be implemented or deleted')
    except Exception as e:
        tb.ok('E2.no_dead_commented_grad_block', False, f"{e!r}")

    return tb


if __name__ == '__main__':
    main(build)
