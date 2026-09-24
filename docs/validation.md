# Validation

Every number here is measured by the suite in `test/`, which checks against **closed-form
physics** wherever possible rather than against another simulator. Run it yourself:

```bash
python test/run_all.py          # single PASS/FAIL verdict
python test/run_all.py --quick  # the fast subset
```

## Summary: 391 checks, 11 benches

| bench | what it guards | checks |
|---|---|---|
| `tb01_photonic_algebra` | unitarity, reciprocity, energy, passivity, resonance, a ring against theory | 43 |
| `tb02_eo_interface` | the `mzm` modulator against textbook MZM physics, including passivity | 41 |
| `tb03_oe_interface` | photodetector noise, bandwidth, and the detector options through `Photonic` | 19 |
| `tb04_native_analytic` | built-in engine vs closed-form solutions | 35 |
| `tb05_native_crosstool` | built-in engine vs HSPICE and Xyce | 33 |
| `tb06_native_gradients` | autograd vs adjoint vs finite difference | 59 |
| `tb07_fixedpoint` | convergence, divergence, bistability, stability of the converged state | 35 |
| `tb08_lumerical_mesh` | photonic mesh vs Lumerical INTERCONNECT | 21 |
| `tb10_bugfix_X` | regression guards on fixed defects, plus the BJT's closed form | 63 |
| `tb11_envelope_P3` | optical memory / envelope propagation, its gradients and its failure checks | 15 |
| `tb12_end_to_end_grad` | `d\|E\|²/dW` through the whole chain | 27 |

`tb05` needs Xyce or HSPICE and is skipped without them, and two checks in `tb10` need a
CUDA GPU; everything else runs on PyTorch alone. `tb08` compares against a *stored* INTERCONNECT result, so it needs no Lumerical
licence — with INTERCONNECT on PATH it also re-checks that the stored reference still
matches a live run (measured drift: exactly 0).

## Photonic core

| property | measured |
|---|---|
| energy conservation, lossless circuit **with an optical loop** | error **exactly 0** (`complex128`) |
| Lorentz reciprocity across an asymmetric lossy network | relative error **0.0** |
| device S-matrices, unitarity `‖SᴴS − I‖` | ≤ 1.8e-16 |
| adjoint `dx/dθ` vs central finite difference | **2.0e-10** |

## Against an independent simulator

A programmable mesh of 2×2 couplers is the sharpest test of the claim that loops are
handled exactly: light recirculates indefinitely, and SPIPE sums the whole series in one
factorisation. Lumerical INTERCONNECT solves the same network by a different route.

| mesh | couplers | max \|E_SPIPE − E_INTERCONNECT\| | max difference in \|E\|² |
|---|---|---|---|
| 2×2 | 12 | **6.8e-07** | 1.2e-06 |
| 3×3 | 24 | **6.6e-07** | 9.4e-07 |

Both over 100 frequency points spanning 192.8–193.4 THz. That ~5e-7 is not a physics
disagreement: INTERCONNECT writes its results through `num2str()`, which is 7 significant
digits, so 1e-6 is the floor of any comparison made through that file. The two solvers
agree as closely as the comparison can resolve.

Runtime on the same problem: SPIPE 0.03 s, INTERCONNECT 8.1 s.

## Electro-optic interface

The `mzm` model is a real interferometer, so V_π is literally the voltage that takes the
output from full-on to full-off:

| property | requested | measured |
|---|---|---|
| V_π | 2.0 V | **2.000000 V** |
| extinction ratio | 10–40 dB | exact to **5e-13 %** |
| insertion loss | 0–6 dB | matches `10^(−IL/10)` to full double precision |
| unitarity in the lossless limit | 0 | 2.2e-16 |
| reciprocity `‖S − Sᵀ‖` | 0 | **exactly 0.0** |

`modm` (the paper's Eq. 5) is retained unchanged and is a *variable-ratio coupler*, not a
push-pull MZI — its V_π differs by 2× and its bias point by π/2. Both models ship; pick
the one that matches your device.

## Opto-electronic interface

| property | required | measured |
|---|---|---|
| shot-noise scaling `d log σ / d log I` | 0.5 | **0.500160** (7 decades) |
| σ vs `√(2qIB)` | — | within 0.2 % at every level |
| thermal noise independence over 1000× current | ratio 1 | 1.0016 |

## Built-in analog engine

Against closed-form solutions:

| check | tolerance | measured |
|---|---|---|
| RC step vs `1 − e^{−t/RC}` | 1e-6 | **8.0e-09** |
| series RLC vs analytic damped sinusoid | 1e-5 | 2.2e-08 |
| Shockley diode DC I–V | 1e-9 | **exact** |
| MOSFET level-1 square law, all three regions | 1e-9 | **0.000e+00** |
| linear DC vs an independent MNA solve | 1e-12 | **0.000e+00** |
| charge conservation | 1e-10 | 6.9e-16 |

Against the commercial tools, on identical netlists: RC ladder, series RLC, diode
rectifier, common-source amplifier, differential pair, CMOS inverter chain, and
ring-oscillator frequency — all within tolerance. Those tolerances are calibrated to **how
much HSPICE and Xyce disagree with each other** on the same decks, because demanding
better agreement than the two commercial tools achieve between themselves would not be a
meaningful bar. A useful anchor: for `VTO=0.7 KP=120u W=20u L=1u LAMBDA=0`, `Vgs=1.2` into
a 20 kΩ load from 3.0 V, HSPICE, Xyce and the textbook square law all give
`v(d) = 0.138384`, and so does SPIPE.

Gradients:

| check | measured |
|---|---|
| autograd vs time-domain adjoint | **exactly 0.00e+00** |
| autograd vs central finite difference | 7.7e-11 |
| adjoint cost, 5 parameters vs 1 | **1.03×** (a true single backward sweep) |

## End-to-end differentiability

The capability the method exists for — an optical output differentiated with respect to an
electronic device parameter, through driver → modulator → photodetector:

Checked against numerical differentiation — a central finite difference that re-runs the
whole co-simulation — on `examples/link_driver_mzm.sp`:

| | `dL/dW` |
|---|---|
| analytic, from `Circuit.simulate()` | `29.19573838` |
| finite difference, best step | `29.19573971` |
| **relative error** | **`4.6e-08`** |

`4.6e-08` is the bottom of a step-size sweep: larger steps are limited by the finite
difference's own truncation error, smaller ones by round-off. The limit is the finite
difference, not the gradient. See [differentiability.md](differentiability.md).

Note this requires the **built-in** backend. Xyce's device derivatives are numerical and
reach ~1e-2 on linear devices but not on MOSFET widths; HSPICE uses finite differences over
re-runs and reaches ~5e-2. Also note that **Xyce's transient adjoint silently returns zeros
for device parameters** — SPIPE forces `direct` and raises if an all-zero sensitivity block
comes back with a non-constant objective.

## Feedback

The fixed-point solver uses Anderson acceleration:

| case | result |
|---|---|
| feedback-free circuit | **exactly 2 iterations** |
| contraction factors to 0.99, and negative factors | 3 iterations |
| a *repelling* fixed point | converges (Picard diverges) |
| genuinely no fixed point | raises, with the residual history attached |
| bistable circuit | both stable states reachable; reports which one |

Demonstrated on real circuits: an optoelectronic oscillator reaching **2.000645 GHz** vs an
analytic **1.996090 GHz** (+0.23 %), and an automatic bias-control loop hitting its
closed-form setpoint to 2e-5 V.
