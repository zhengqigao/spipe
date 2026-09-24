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

**Guarded by:** `Photonic.check_quasistatic(dt)`, which warns, with both numbers, when the
network's delay exceeds `0.1·dt`. It uses the larger of two estimates: the longest light path
(delay lines, meshes), and the group delay `dφ/dω` measured at the simulated carriers. The
second one is what sees a resonator, whose photon lifetime is its round trip times its
finesse — on a high-Q ring, 2.2 ns where the path length alone suggests 1.7 ps. Disable with
`config['quasistatic_check'] = False`.

**Lifted by:** `simulate(..., mode='envelope')`. The passive sub-network is linear and
time-invariant, so its transfer `H(ω)` over the `.freq` band is computed once and inverse-
FFT'd into an impulse response; the detected field is then a convolution, so light arriving
at time `t` carries the modulator state from `t − τ`. This upgrades the assumption from
*"optical memory is zero"* to *"optical memory is short compared with how fast the
modulator changes"*.

Envelope mode requires a `.freq` grid wide enough (`band ≫ 1/dt`) and fine enough
(`1/(2·df) ≫ τ_max`); it warns with concrete numbers when either is violated.
It is differentiable with respect to the **modulator drive**: `backward()` through an
envelope-mode result fills in `drive.grad`, matching finite differences to `4e-09`. It also works
inside a `Circuit` (`simulate(mode='envelope')`), including gradients with respect to
`.sensparam` parameters. It is not yet differentiable with respect to *passive* device
parameters (a waveguide length, a coupler angle); use `mode='quasistatic'` for those. How to
size the grid, which carrier to read, and the settings are in [envelope.md](envelope.md).

### A known limitation: modulators inside optical loops

Envelope mode represents a delay shorter than the time step with a fractional-delay kernel.
Inside a feedback loop — **a modulator inside a ring**, say — that kernel can act as gain at
some carriers, and the calculation goes unstable. The ring's own physics is fine (the
steady-state solve is exact); the time-domain method is what fails. Measured on a ring with a
1.68 ps round trip: at 1 ps steps the output grew to 464× the input power; at 2 ps steps the
centre carrier looked fine while carriers across the band reached 10¹⁵×.

SPIPE checks for this rather than returning such numbers:

- **Runaway** is an error. If, at the end of the record, the detected power is more than ten
  times both the launched power and the circuit's own steady-state output, the run raises.
  (A resonator *can* briefly emit more than it receives — it releases stored energy when the
  input changes — so the test is runaway growth, not instantaneous power.)
- **Failure to settle** is a warning. Once the drive has been constant for longer than the
  network's memory, the result must equal the quasi-static steady state; if it differs by more
  than 0.1 %, SPIPE says so, with the size of the discrepancy.

Modulators *outside* a loop — driving a ring from outside, or feeding a delay line — are the
case envelope mode is built for, and they are unaffected.

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
by a 2 m optical delay line. SPIPE warns: it prints the group delay (15.68 ns), the time step
(25 ps) and their ratio (627). The example then stops rather than simulate, explaining that the
frequency-selective element is structurally absent from the quasi-static model, and points at
`mode='envelope'`. It reports no oscillation frequency, because any number the default mode
produced would be meaningless. The library does not stop you: a warning is all it gives, and
`Circuit.simulate()` on that netlist runs and returns such a number.

That is the intended behaviour: a simulator should decline rather than return a plausible
wrong answer.
