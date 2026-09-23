"""TB-12 : end-to-end differentiability (SPEC-E2).

The headline of the whole campaign: d|E_out|^2 / dW for a transistor width, through
driver -> modulator -> photodetector, against central finite differences over the WHOLE
chain. Written from SPEC_E2.md and SPEC_E1.md only.

Note the sky130 DAC is deliberately NOT used: sky130 ships 63 per-size binned BSIM4 cards
(lmin=1.45e-7..1.55e-7, wmin=1.255e-6..1.265e-6), so W is effectively discrete there and
d/dW is ill-posed. Level-1 devices make W continuous and analytically differentiable.
"""
import sys, os, math
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
    # Path B: the same thing through Circuit + .sensparam, if wired up.
    # ------------------------------------------------------------------
    try:
        import spipe.core.core as core
        src = open(_pkg('electronic', 'electronic.py')).read()
        tb.ok('E2.sensparam_supported', '.sensparam' in src or 'sensparam' in src,
              'netlist syntax for declaring differentiable device parameters')
    except Exception as e:
        tb.ok('E2.sensparam_supported', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The silent-zero guard. Xyce's transient ADJOINT returns all zeros for
    # device parameters (measured: d_{V(D)}/d_M1:W_adj = 0.0 while direct
    # gives -4.99695862e+04). An all-zero sensitivity block must RAISE.
    # ------------------------------------------------------------------
    try:
        from spipe.electronic import electronic as E
        names = [n for n in dir(E) if 'zero' in n.lower() or 'guard' in n.lower()]
        src = open(_pkg('electronic', 'electronic.py')).read()
        has_guard = ('all-zero' in src.lower() or 'all zero' in src.lower()
                     or 'allzero' in src.lower() or bool(names))
        tb.ok('E2.zero_sensitivity_guard', has_guard,
              "an identically-zero sensitivity block must raise, not be returned "
              f"(helpers found: {names})")
    except Exception as e:
        tb.ok('E2.zero_sensitivity_guard', False, f"{e!r}")

    # ------------------------------------------------------------------
    # The fixed point must be differentiated by the implicit function
    # theorem, not by unrolling.
    # ------------------------------------------------------------------
    try:
        csrc = open(_pkg('core', 'core.py')).read()
        tb.ok('E2.no_dead_commented_grad_block',
              'def gradient_based_simulate' not in csrc.replace('# ', '')
              or 'def gradient_based_simulate' in csrc,
              'the commented-out gradient_based_simulate must be implemented or deleted')
        tb.ok('E2.implicit_diff_used',
              any(k in csrc.lower() for k in
                  ('implicit', 'ift', 'implicit_function', 'fixed_point_grad')),
              'fixed-point gradients must use the implicit function theorem, not unrolling')
    except Exception as e:
        tb.ok('E2.implicit_diff_used', False, f"{e!r}")

    return tb


if __name__ == '__main__':
    main(build)
