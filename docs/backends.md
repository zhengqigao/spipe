# Electronic backends

`Circuit(path, spice_exe=...)` chooses which engine solves the `.electronic` section.

| backend | `spice_exe=` | availability | gradients |
|---|---|---|---|
| built-in | `"native"` | ships with SPIPE | **exact** |
| Xyce | `"xyce"` or a path | free, open source, GPL-3.0 ([Xyce](https://xyce.sandia.gov/), Sandia National Laboratories) | numerical |
| HSPICE | `"hspice"` or a path | commercial (Synopsys) | finite differences over re-runs |

Two of the three are freely available: the built-in engine needs nothing beyond this
package, and Xyce is open source.

Check what is on your machine with `which` — note the capital X on Xyce:

```bash
which Xyce
which hspice
```

If those print a path, SPIPE finds them with no further setup. If not, put them on `PATH`
however your environment does it (that differs from site to site, so SPIPE assumes
nothing), or point `$SPIPE_XYCE` / `$SPIPE_HSPICE` at the executable.

## The built-in engine

`src/spipe/electronic/native/` is a complete analog simulator in Python and PyTorch. It
covers resistors, inductors, capacitors, independent and controlled sources, diodes, BJTs,
level-1 MOSFETs and switches, with DC and transient analysis, Newton–Raphson with damping,
and local-truncation-error-based adaptive time stepping.

The device models are deliberately simple, and the engine refuses rather than approximates
anything beyond them. MOSFETs are `LEVEL=1` (Shichman–Hodges); BJTs are Gummel–Poon
(`LEVEL=1`, or no `LEVEL`). A foundry card asking for BSIM, PSP or VBIC — `LEVEL=9` or `12`
on a BJT, as in the IHP SG13G2 PDK — is an error that tells you to use Xyce or HSPICE.
Gummel–Poon parameters the engine does not implement (series resistances `RB`/`RC`/`RE`,
high-level injection `IKF`/`IKR`) are ignored with a warning naming them, because they do
change the answer. The same goes for MOSFET model parameters that level 1 lacks (`RD`,
`RS`, ...). A misspelled MOSFET instance parameter is an error: `WW=20u` would otherwise run
with the default 100 µm width.

It reproduces closed-form solutions to machine precision and matches HSPICE and Xyce on
identical netlists — see [validation.md](validation.md). A useful anchor: for `VTO=0.7
KP=120u W=20u L=1u LAMBDA=0`, `Vgs=1.2` into a 20 kΩ load from 3.0 V, HSPICE, Xyce and the
textbook square law all give `v(d) = 0.138384`, and so does SPIPE.

It is also the only backend that differentiates exactly, because a device parameter is a
PyTorch tensor there rather than a number in a text file.

**It is slower than Xyce.** Here is one co-simulation of a two-stage CMOS driver into an `mzm`
and a detector, with 161 samples over 1.6 ns and two `.sensparam` widths. It was measured on
a shared server that was busy with other work, so treat the absolute times as rough;
the ratios are what matter:

| backend | plain run | run + `backward()` |
|---|---|---|
| built-in | 33 s | 91 s |
| Xyce | 3.6 s | 16 s |

Most of the time goes into the Newton solves, and a fixed-point co-simulation repeats the
electronic transient several times. Use Xyce while you explore a design. Use the built-in
engine when you need exact gradients, or when Xyce is not installed.

## Time-step accuracy of the built-in engine

The built-in engine splits each output interval of `.tran` into a number of equal internal
steps. It raises that number until its truncation-error estimate is met, up to a cap: 64 for
linear circuits, but only **8** for circuits with transistors or diodes, to keep Newton
solves affordable. A switching circuit whose edges are much faster than the output interval
can hit that cap before its error target, and the result is then less accurate than the
engine's own estimate says it should be. It returns anyway.

Measured on `examples/link_driver_mzm.sp`: 40 output samples over 40 ns, 0.2 ns edges, an
output node with a ~50 ps time constant:

| internal steps per output sample | runtime | worst error in the modulator drive |
|---|---|---|
| 8 (the default cap) | 6.3 s | 5.3 mV (a slowly decaying tail after each edge) |
| 16 | 10.4 s | 0.47 mV |
| 32 | 19.2 s | 0.11 mV |
| 64 | 31.6 s | 0.023 mV |

The gradient is exact for whichever discretisation ran. That is why finite differences agree
with it at every setting. The table is about how close that discretisation is to the
continuous circuit.

To trade speed for accuracy, fix the number of internal steps:

```python
spipe.config['native_nsub'] = 32
```

or give `.tran` more output points, which also lets the modulator see the drive more often.
A run that rings or shows a slow tail right after a fast edge is the symptom to look for.

**Detector outputs are the worst case.** A detector's output node usually has a time constant
of picoseconds (2 fF behind 1 kΩ plus the load), far below the spacing of the output
samples. There the trapezoidal rule does not settle: each change of photocurrent leaves a tail of
roughly `R · τ · ΔI / Δt` that only shrinks to about a third of itself per sample, where it should vanish at once. In the example above,
with `.print tran v(vo1)` added, the "0" level after a 500 V pulse reads 1.98, 0.73, 0.28 V
over the next samples, where the converged answer (`native_nsub = 64`) is 1.95, 0, 0. The
photocurrent itself and the modulator drive are not affected. If you read a detector's
*voltage*, fix `native_nsub` at 64 (the table above shows what that costs).

## A trap worth knowing about Xyce

Xyce's **transient adjoint** sensitivity silently returns **all zeros** for device
parameters. Measured on Xyce 7.10, for the derivative of a node voltage with respect to a
MOSFET width:

| setting | result |
|---|---|
| `.options sensitivity direct=1` | `-4.997e+04` — matches finite differences to 0.075 % |
| `.options sensitivity adjoint=1` | `0.0`, with no error and a zero exit status |

A silent zero gradient is the worst possible failure for an optimiser: it is
indistinguishable from a converged design. SPIPE therefore forces the direct method
whenever a `.sensparam` card is present, and **raises** if an all-zero sensitivity block
comes back while the objective is visibly moving, rather than handing you the zero.

Xyce's `.options sensitivity` is a single global setting rather than a per-request one, so
the device parameters and the photocurrent parameters cannot use different methods in the
same run.

## Accuracy of the external backends

Measured end to end on a level-1 CMOS driver into a modulator:

- **Xyce** — the dominant width sensitivity agrees with a central finite difference over
  the whole chain to about 8e-3; the smaller ones to no better than a factor of a few. Its
  device derivatives are numerical, and it says so. The finite-difference reference is
  itself inconsistent at the tens-of-percent level, because Xyce's transient is only
  reproducible to ~2e-4 relative with respect to its own inputs.
- **HSPICE** — reaches about 5e-2, by re-running the deck.
- **Built-in** — the same measurement comes out at 4e-5, and its autograd and adjoint agree
  with each other to exactly zero.

## SPICE repeatability, for feedback circuits

A waveform fixed point compares two SPICE runs that differ by less than a millivolt, so the
simulator has to be that repeatable — and by default it is not. HSPICE prints tables to five
significant digits and picks its time steps adaptively from the waveforms, so handing it a
slightly different photocurrent also hands it a different integration grid. The loop
examples therefore set

```
.option numdgt=7 delmax=<dt/4> relv=1e-6 reli=1e-8 absv=1e-9 absi=1e-15
```

On `examples/oeo_electronic.py` that takes the residual floor from about 3e-3 V to below
1e-3 V, and the iteration count from 125 to 45. Without it the iteration stalls on
simulator noise rather than on the physics.

## Lumerical INTERCONNECT

INTERCONNECT is not an electronic backend — it is the independent cross-check for the
*photonic* solver. `test/tb/tb08_lumerical_mesh.py` compares SPIPE against a stored
INTERCONNECT result and needs no licence; `examples/mesh_vs_lumerical.py` shows the same
comparison and can re-run INTERCONNECT live with `--live`.

The executable is lower case: `which interconnect`. On a machine with no display, set
`QT_QPA_PLATFORM=offscreen`, or INTERCONNECT exits with "no Qt platform plugin could be
initialized".
