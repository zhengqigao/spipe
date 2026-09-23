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

It reproduces closed-form solutions to machine precision and matches HSPICE and Xyce on
identical netlists — see [validation.md](validation.md). A useful anchor: for `VTO=0.7
KP=120u W=20u L=1u LAMBDA=0`, `Vgs=1.2` into a 20 kΩ load from 3.0 V, HSPICE, Xyce and the
textbook square law all give `v(d) = 0.138384`, and so does SPIPE.

It is also the only backend that differentiates exactly, because a device parameter is a
PyTorch tensor there rather than a number in a text file.

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
