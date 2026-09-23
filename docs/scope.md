# Scope: what SPIPE models, and where it stops

SPIPE's core formulation is exact. This document states precisely where the *physics*
assumptions bind, so you can tell whether a given circuit is inside the valid regime.

## The core is exact

For a photonic circuit with `N_p` ports, SPIPE associates two unknowns with every port —
one wave in each direction — and assembles one linear constraint per device port
(`out = S · in`) plus one boundary condition per dangling node. The resulting system is
square by construction and is solved directly.

Two consequences worth stating:

- **Loops are handled exactly.** The direct solve sums the infinite series of round trips,
  so a recirculating mesh needs no iteration and no truncation.
- **Energy is conserved to solver precision.** On a lossless network — including one with
  optical feedback — total detected power equals injected power to machine zero in
  `complex128`.

The gradient is likewise exact: differentiating `A x = b` gives `dx/dθ = −A⁻¹ (dA/dθ) x`,
which matches central finite differences to ~1e-10 in double precision.

## Assumption 1: the modulator responds instantly

Stated in the paper. A drive `v(t)` produces phase `φ(t)` with no transient. Valid when the
modulator's own response time is far shorter than the drive's timescale — true for
free-carrier plasma dispersion (~0.1 ps) against a 10 Gsps DAC (100 ps).

**Lifted by:** the `mzm` model's `tau=` parameter, which gives the phase a first-order
(or `order=n` cascaded) lag. `tau=0` reproduces the instantaneous result bit-for-bit.

## Assumption 2: the photonic network settles instantly

**Not stated in the paper, and it is the one that actually binds.**

Because the network is solved in *steady state* at every time sample, SPIPE implicitly
assumes every optical transit and round-trip time is negligible against the modulation
timescale. Concretely, a 250 µm waveguide at `ng = 4` is 3.3 ps one way; a 3×3 mesh round
trip is tens of ps. At 0.1 Gsps (10 ns/sample) there is ~1000× margin. At 10 Gb/s it is
10–50 % of a bit. A high-Q ring has a photon lifetime of nanoseconds.

The failure mode is **structural, not gradual**. With zero optical memory there is no
delay, so resonator ring-up, delay-set oscillation and pattern-dependent ISI are not
approximated badly — they are absent from the model.

**Guarded by:** `Photonic.check_quasistatic(dt)` estimates the maximum group delay in the
network and warns, with both numbers, when it exceeds `0.1·dt`. Disable with
`config['quasistatic_check'] = False`.

**Lifted by:** `simulate(..., mode='envelope')`. The passive sub-network is linear and
time-invariant, so its transfer `H(ω)` over the `.freq` band is computed once and inverse-
FFT'd into an impulse response; the detected field is then a convolution, so light arriving
at time `t` carries the modulator state from `t − τ`. This upgrades the assumption from
*"optical memory is zero"* to *"optical memory is short compared with how fast the
modulator changes"*.

Envelope mode requires a `.freq` grid wide enough (`band ≫ 1/dt`) and fine enough
(`1/(2·df) ≫ τ_max`); it warns with concrete numbers when either is violated.
**It does not currently support gradients** — use `mode='quasistatic'` for the adjoint.

## Assumption 3: the frequency axis is incoherent channels

The photodetector sums `|p(ω_n)|²` over the `.freq` grid. That is correct when those
points are independent WDM carriers whose beat notes fall outside the detector bandwidth.
It is **wrong** if you read the grid as the Fourier decomposition of one modulated signal,
where the cross terms *are* the signal.

**Made explicit by:** the `coherent=` option on the `pd` line, which squares the summed
field and low-passes at the detector bandwidth `bw`, so the two regimes are one formula.
The default preserves the incoherent behaviour.

`coherent=1` needs `bw=` (there is nothing to reject without a pole) and a transient time
axis, which means a netlist with at least one modulator — a purely passive circuit is a
single sample and carries no time. Two properties are worth knowing before you read the
output:

- The reduction to the incoherent sum is **asymptotic, not exact**. A single-pole filter
  rejects a beat note at `Δf` by roughly `bw / Δf`, so carriers 500× above the bandwidth
  leave about 1.7 % ripple — measured at 2.5e-2 A on a 1.5 A photocurrent.
- The low-pass starts from the **first sample**, and at `t = 0` every carrier is in phase
  by construction, so the coherent photocurrent starts at its fully constructive value
  (4.5 for three unit channels, against a steady state of 1.5) and decays toward the
  incoherent value with the detector's own time constant `τ = 1/(2π·bw)`. Discard the
  first few `τ`, or start the record earlier than the window you care about.

`test/tb/tb03_oe_interface.py` pins both, and also pins that `Photonic` hands the detector
its time axis at all: before that was wired up, `bw=` and `coherent=` were accepted, warned
once, and silently fell back to an unfiltered incoherent sum.

## A worked demonstration of the boundary

`examples/oeo_optical_delay.py` builds an optoelectronic oscillator whose frequency is set
by a 2 m optical delay line. SPIPE **refuses to simulate it**, prints the group delay
(15.68 ns), the time step (25 ps) and their ratio (627), explains that the frequency-
selective element is structurally absent from the quasi-static model, and points at
`mode='envelope'`. It reports no oscillation frequency, because any number it produced
would be meaningless.

That is the intended behaviour: a simulator should decline rather than return a plausible
wrong answer.
