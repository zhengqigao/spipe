# SPIPE examples

Circuits that exercise parts of SPIPE the shipped `test1/` and `test2/` decks do not reach.

Everything in `test1/` and `test2/` is **feedback free** in the sense of Definition 1 of the
paper: no photocurrent affects any modulator drive. That is why the electronic/photonic fixed
point in those examples converges in exactly two evaluations — the round-trip map is a constant,
so one evaluation produces the answer and the second confirms it. It is also why the iteration
had never been stressed.

The circuits here have real feedback, and one of them deliberately has no answer at all.

| example | what it is | what it shows |
|---|---|---|
| `oeo_electronic.py` | optoelectronic oscillator, frequency set by an electronic band-pass | loop gain > 1; the solver has to converge on a limit cycle |
| `bistable_latch.py` | electro-optic latch with positive feedback | **two** stable fixed points; the answer depends on the initial guess |
| `bias_control.py` | integrating bias-control loop | negative feedback; setpoint checkable in closed form |
| `oeo_optical_delay.py` | OEO whose frequency is set by an *optical* delay | **refuses to simulate**; explains why and points at `mode='envelope'` |
| `derived/n1_dac.py` | Level-1 R-2R DAC | calibrated against the measured sky130 reference |
| `derived/n2_mzm_driver.py` | CMOS driver into the realistic `level3` RC modulator load | where `d(optical)/dW` lives |
| `derived/n3_tia.py` | real common-source TIA | replaces the ideal 70 dB block of `pd_level2` |
| `derived/n4_ptc_dotproduct.py` | the 2x2 photonic tensor core of `test2/test12.py` | same photonic netlist, Level-1 electronics |

## What they measured when they were written

HSPICE V-2023.12-SP2, `spipe.config` defaults unless the example says otherwise.

| example | result |
|---|---|
| `oeo_electronic.py` | oscillates at **2.0006 GHz** (FFT) / 1.9913 GHz (upward zero crossings) against an analytic closed-loop pole at 1.99609 GHz; limit-cycle drive amplitude 1.313 V against 1.410 V from the describing function; **45 fixed-point iterations**, residual 6.41e-2 -> 1.32e-3 |
| `bistable_latch.py` | two stable states, **0.119232 V** and **2.229391 V**, against closed-form roots 0.119238 V and 2.229372 V; basin boundary at the unstable root 0.698867 V, straddled by the `x0 = 0.65` / `0.75` runs; 9-25 Picard iterations each |
| `bias_control.py` | settles at vbias + 1.000 V for vbias = 0, +0.3, -0.4 V, within **2.2e-5 V** of the closed form in every case; 27-31 iterations |
| `oeo_optical_delay.py` | refuses; 15.68 ns of optical delay against a 25 ps time step, `max_group_delay / dt = 627` |
| `derived/n1_dac.py` | swing **3.272953 V** (sky130 3.274181), LSB **25.7713 mV** (25.7810), 1% settling **2.728 ns** (2.730), monotonic (sky130 is not), INL/DNL 0.381 / 0.748 LSB (sky130 2.517 / 2.517) |
| `derived/n1_dac.py --native` | the same 128-code transfer solved by `spipe.electronic.native`: **max &#124;native - HSPICE&#124; = 17 uV = 0.0007 LSB** |
| `derived/n2_mzm_driver.py` | eye 2.3 -> 385 uA as W goes 4 -> 64 um; **d(eye)/dW = 9.34 uA/um** at W = 16 um |
| `derived/n3_tia.py` | real TIA 7.99 kOhm and 480 ps against the ideal block's 9.99 kOhm and 60 ps |
| `derived/n4_ptc_dotproduct.py` | photonic section byte-identical to `test2/test12.sp`; 2 fixed-point iterations; DAC error 1.44 LSB worst case; dot products within 0.25-0.89 V of closed form on a +-20 V scale |

## Running them

```
python examples/oeo_electronic.py --plot
python examples/bistable_latch.py
python examples/bias_control.py
python examples/oeo_optical_delay.py
python examples/derived/n1_dac.py
python examples/derived/n1_dac.py --calibrate   # re-fit cload to the sky130 settling time
python examples/derived/n1_dac.py --native      # same transfer on spipe.electronic.native
python examples/derived/n4_ptc_dotproduct.py --tia   # also swap in N3's real front end
```

Common options: `--sim {hspice,xyce}`, `--spice-exe '<command line>'` (or the
`SPIPE_SPICE_EXE` environment variable), `--work-dir`, `--max-iter`, `--plot`. The default
executables are the ones `test1/` and `test2/` use. Each example writes the netlist it
simulated, the SPICE scratch files and any plots into `examples/_run/<name>/`, so the deck that
produced a number is always on disk next to it. Set `MPLBACKEND=Agg` when running headless.

## The solver these exercise

`spipe.core.core.solve_fixed_point(step, x0, config, max_iter=None) -> (x_star, info)` is the
electronic/photonic coupling, written without any reference to SPICE, so that its convergence
behaviour is testable with a three-line pure-Python `step`.
`Circuit.simulate()` reaches it through `gradient_free_simulate`, so the two cannot
drift apart.

* **Anderson acceleration** (`config['anderson_depth']`, default 5) is the workhorse. It costs
  no extra evaluations of `step` and converges on contraction factors at and above one, which
  plain Picard cannot.
* **Adaptive under-relaxation** backs off when an iteration lands far above the best residual
  seen, twice running. It deliberately does *not* react to a residual that merely rises: on a
  circuit that oscillates the residual rises for many iterations while the limit cycle is being
  discovered, and dropping the Anderson history there would remove the only thing that works.
* The **first update is a plain Picard step** (empty history, `beta = 1`), so a feedback-free
  circuit still converges in exactly two iterations, to the same numbers as before.
* **Non-convergence raises.** `FixedPointNotConverged` when the iteration budget runs out,
  `FixedPointDivergence` when the residual becomes non-finite or leaves the best residual far
  behind; both carry `.residuals`, `.iters` and `.info`. Silently returning the last iterate of
  a loop that did not converge is the one outcome the solver will not produce.
* The convergence criterion keeps its meaning: `||x - step(x)|| / sqrt(numel) <= rtol *
  ||x|| / sqrt(numel) + atol`, i.e. an rms-per-entry quantity that does not change meaning when
  the number of time samples or modulators changes.

`Circuit.simulate(x0=...)` takes an explicit initial guess. On a circuit with more than one
stable state that is the only way to choose which one you get, and `bistable_latch.py` uses it.

## A practical note on SPICE repeatability

A waveform fixed point compares two SPICE runs that differ by less than a millivolt, so the
simulator has to be that repeatable — and by default it is not. HSPICE prints its tables with
five significant digits and chooses its transient time steps adaptively from the waveforms, so
handing it a slightly different photocurrent also hands it a different integration grid. The
loop examples therefore set

```
.option numdgt=7 delmax=<dt/4> relv=1e-6 reli=1e-8 absv=1e-9 absi=1e-15
```

(see `_common.spice_options`). On `oeo_electronic.py` that takes the residual floor from about
3e-3 V to below 1e-3 V and the iteration count from 125 to 45. Without it the iteration stalls
on simulator noise rather than on the physics.

## An environment hazard, now fixed in the library

On some installations (verified on torch 2.3.1 + scipy 1.8.0), **importing torch before
scipy silently corrupts `scipy.interpolate.interp1d(kind='cubic')`**. It returns `nan` or
`inf` — a *different* wrong value on each run — while the input data is provably correct.

That function is `interp1d_warp`, which maps HSPICE/Xyce transient results back onto the
simulation time grid, so the corruption reached **every co-simulation result**, silently,
with exit status 0. Historically these scripts were protected only by accident: they
imported another package first, which pulled in numpy and scipy ahead of torch.

`spipe/__init__.py` now imports numpy and `scipy.interpolate` **before** torch, which makes
the safe order unconditional however your own script orders its imports. You should never
see this, but it is worth knowing the failure mode existed: it is upstream of the
fixed-point iteration, and the old solver returned those numbers without complaint.
