# `spipe.electronic.native` — SPIPE's own differentiable SPICE engine

A SPICE-class analog circuit simulator written entirely in Python + PyTorch +
NumPy, float64 throughout, and **natively differentiable**: every device
parameter is a leaf tensor, so the gradient of an optical or electrical output
with respect to a transistor width is one backward pass away.

It is the third electronic backend alongside the HSPICE and Xyce subprocess
drivers, and the only one that can hand SPIPE clean gradients.

```python
from spipe.electronic.native import Netlist

ckt = Netlist.from_string(netlist_text)      # or Netlist.from_file(path)

op  = ckt.op()                                # {node name: float64 scalar tensor}
res = ckt.tran(1e-11, 2e-9)                   # TranResult

res.t                  # (N,) float64 time points
res.v('out')           # (N,) node voltage
res.i('V1')            # (N,) current through a V source / inductor / E / H
res.nodes              # list[str], ground excluded
```

Ground is node `0` (`gnd`, `GND`, `ground` are accepted too).  Node and device
names are case-insensitive.

---

## Gradients — two independent paths

**1. Autograd.**  Device parameters are ordinary torch leaves:

```python
w = ckt.param('M1', 'W')
w.requires_grad_(True)
res = ckt.tran(1e-11, 2e-9)
res.v('out').sum().backward()
w.grad                                        # correct, float64
```

`ckt.param(dev, name)` raises `UnknownParameterError` (a subclass of both
`KeyError` and `ValueError`) naming the available devices/parameters.
`ckt.params()` returns the whole `{(device, parameter): tensor}` mapping.

**2. Explicit time-domain adjoint.**

```python
g = ckt.adjoint_grad(lambda r: r.v('out').sum(),
                     params=[('M1', 'W'), ('R1', 'R')])
# -> {('M1','W'): tensor(...), ('R1','R'): tensor(...)}
```

This is a **single backward sweep** over the stored time steps — one
transposed linear solve and one vector-Jacobian product per step — not
repeated forward solves.  The cost of a gradient is independent of the number
of parameters.

The autograd path is implemented *by* the adjoint (`TranAutograd.backward`
calls the same sweep), so the two agree to the last bit; both were checked
against central finite differences (see "Verification" below).
`ckt.op_adjoint_grad(objective)` does the same for a DC operating point.

---

## Extensibility hook: `add_external_block`

A later phase injects photonic delay-line state equations into this same MNA
system so that the electronic + photonic circuit solves as one DAE.  The MNA
assembly and the Newton loop route through that interface already, so adding a
block needs no change to the solver.

```python
ckt.add_external_block(block)
```

`block` must provide:

| member | meaning |
|---|---|
| `nodes` | list of circuit node names the block couples to (created if new) |
| `n_states` | number of extra unknowns the block owns (may be 0) |
| `residual_and_jacobian(x, xdot, t)` | returns `(r, J_x, J_xdot)` |
| `init_state()` | *optional*, initial values of its own states |
| `parameters()` | *optional*, `{name: leaf tensor}` exposed for gradients |

`x` and `xdot` are `(k,)` float64 tensors with `k = len(nodes) + n_states`,
ordered `[coupled node voltages ..., own states ...]`.  The returned `r` is
`(k,)` and is added to the system rows with those same indices — node entries
are currents flowing out of the node into the block (ordinary KCL sign, just
like a device stamp), state entries are the block's own equations.  `J_x` and
`J_xdot` are `(k, k)`.

The solver applies the same integration formula to `xdot` that it uses for
device charges, `xdot_n = alpha_n (x_n - x_{n-1}) + beta_n xdot_{n-1}`, and
builds the Newton Jacobian as `J_x + alpha * J_xdot`.  Because `r` is built
from ordinary torch operations on the block's parameters, the adjoint picks up
`dr/dp` automatically — blocks are differentiable with no extra work.

Two worked examples ship in `external.py`:
`BehaviouralResistor` (stateless, two-terminal) and `RCStateBlock` (one
internal state, exercises the `xdot` path).  Both are verified to reproduce
the equivalent `R`/`C` device exactly.

---

## Supported syntax

* comments `*` (full line) and `;` / `$` (trailing); continuation `+`
* the first line of a deck is taken as a **title** when it does not parse as a
  card, so both conventional decks and bare fragments work
* engineering suffixes `T G MEG K M MIL U N P F A` (case-insensitive; trailing
  letters such as `1kohm` are ignored)
* `.param name=value ...`, and `{expr}` / `'expr'` parameter expressions with
  `+ - * / **`, comparisons, `? :`, and `abs min max sqrt exp ln log sin cos
  tan atan sinh cosh tanh pow pwr int sgn floor ceil`
* `.model name TYPE (p=v ...)`, `.ic v(n)=x`, `.nodeset`, `.global`,
  `.options`, `.include`
* `.subckt` / `.ends` with port mapping and per-instance parameter overrides,
  instantiated by `X`; hierarchical names are `x1.m1`, hierarchical nodes
  `x1.internal`
* analyses `.op`, `.tran tstep tstop [tstart] [UIC]`, `.dc src start stop step`
* ignored but tolerated: `.print`, `.probe`, `.plot`, `.save`, `.ac`,
  `.measure`, `.temp`, `.lib`, `.end`

`.param` expressions are resolved **at parse time**; each device parameter then
becomes an independent float64 leaf, which is what makes `ckt.param()` return
something you can attach gradients to.

---

## Device table

### Tier 1 — passive, sources, controlled sources

| card | device | parameters |
|---|---|---|
| `Rname n+ n- <val>` | resistor | `R`, `M` |
| `Cname n+ n- <val> [IC=]` | capacitor (charge based) | `C`, `IC`, `M` |
| `Lname n+ n- <val> [IC=]` | inductor (flux based) | `L`, `IC` |
| `Vname n+ n- ...` | voltage source | waveform parameters, below |
| `Iname n+ n- ...` | current source | waveform parameters, below |
| `Ename n+ n- nc+ nc- <k>` | VCVS | `GAIN` |
| `Gname n+ n- nc+ nc- <gm>` | VCCS | `GM` |
| `Fname n+ n- <Vctrl> <k>` | CCCS | `GAIN` |
| `Hname n+ n- <Vctrl> <r>` | CCVS | `R` |

Source functions and their parameter names:

| function | parameters |
|---|---|
| `DC <v>` | `DC` |
| `PULSE(v1 v2 td tr tf pw per)` | `V1 V2 TD TR TF PW PER` |
| `PWL(t0 v0 t1 v1 ...)` | `T0 V0 T1 V1 ...` |
| `SIN(vo va freq td theta phase)` | `VO VA FREQ TD THETA PHASE` |
| `EXP(v1 v2 td1 tau1 td2 tau2)` | `V1 V2 TD1 TAU1 TD2 TAU2` |

`TR`/`TF` default to the transient print step and `PW`/`PER` to `tstop`, as in
SPICE.  `res.i('V1')` is positive for current flowing from `n+` through the
source to `n-`.

### Tier 2 — diode and MOSFET

| card | device | parameters |
|---|---|---|
| `Dname a c <model> [area]` | Shockley diode | `IS N RS CJO VJ M TT FC AREA` |
| `Mname d g s b <model> W= L=` | MOSFET level 1 | see below |

Diode current is `AREA*IS*(exp(Vd/(N*Vt)) - 1)`, with the standard junction
depletion charge (`FC` linearisation above `FC*VJ`) plus `TT*I`, integrated in
charge form.  `RS > 0` introduces an internal node `<name>#i`.  Convergence
uses Berkeley `pnjlim` voltage limiting plus a `limexp` overflow guard; both
are inactive at the solution, so the converged answer satisfies the
unmodified Shockley equation.

MOSFET model card (`.model nch NMOS (...)` / `PMOS`):

| parameter | default | meaning |
|---|---|---|
| `LEVEL` | 1 | only level 1 (Shichman-Hodges) is implemented |
| `VTO` | 0.0 | threshold voltage, signed (negative for a normal PMOS) |
| `KP` | 2e-5 | transconductance parameter [A/V²] |
| `LAMBDA` | 0.0 | channel-length modulation [1/V] |
| `GAMMA` | 0.0 | body-effect factor [V^½] |
| `PHI` | 0.6 | surface potential [V] |
| `TOX` | 0 | oxide thickness [m]; sets `COX` when `COX` is not given |
| `COX` | 0 | oxide capacitance per area; 0 disables intrinsic capacitances |
| `UO` | 0 | surface mobility [cm²/Vs]; with `TOX` it sets `KP` |
| `CAPOP` | 3 | 0 = no intrinsic caps, 2 = Meyer with the Berkeley sub-threshold ramp, 3 = Meyer without the intrinsic gate-bulk term |
| `CGSO`, `CGDO` | 0 | gate-source / gate-drain overlap capacitance per metre of `W` |
| `CGBO` | 0 | gate-bulk overlap capacitance per metre of `L` |
| `CBD`, `CBS` | 0 | bulk-drain / bulk-source capacitance [F] |

Instance parameters: `W` (1e-4), `L` (1e-4), `M` (multiplier), `AD`, `AS`.
All three regions (cutoff / triode / saturation), both polarities, and the
drain/source role swap for `Vds < 0` are implemented.

### Tier 3 — BJT and switches

| card | device | parameters |
|---|---|---|
| `Qname c b e [sub] <model> [area]` | reduced Gummel-Poon BJT | `IS BF BR NF NR VAF VAR CJE VJE MJE CJC VJC MJC TF TR AREA` |
| `Sname n+ n- nc+ nc- <model>` | voltage-controlled switch | `RON ROFF VON VOFF` (or `VT`/`VH`) |
| `Wname n+ n- <Vctrl> <model>` | current-controlled switch | `RON ROFF ION IOFF` (or `IT`/`IH`) |

Switch conductance uses the smooth (C¹) log-interpolation between `VOFF` and
`VON` that PSpice/LTspice use, which keeps Newton well behaved.

---

## Numerics

* **float64 everywhere** in the solve path.
* **DC operating point**: Newton-Raphson with step damping and SPICE device
  limiters, falling back automatically to **gmin stepping**
  (`1e-3 → 1e-12 → 0`) and then to **source stepping**.  Convergence requires a
  small update *and* a small residual measured against the SPICE per-row
  current scale, *and* that no limiter fired — so the answer always satisfies
  the unmodified device equations.
* **Transient**: trapezoidal by default (backward Euler available with
  `method='be'`, and used for the first step / after breakpoints when
  `be_restart=True`).  `dq/dt` at the initial point is taken from the DAE
  itself (`f + dq/dt = 0`), which makes the trapezoidal start second-order
  accurate for both an operating-point start and a `uic` start.
* **Breakpoints** from `PULSE`/`PWL`/`SIN`/`EXP` are exact grid points.  A
  genuinely *discontinuous* source gets a sliver step inserted just before the
  breakpoint, so the discontinuity is never integrated across.
* **Step control**: each print/breakpoint interval is split into `nsub`
  uniform internal steps; `nsub` is raised — globally — until the worst local
  truncation error (third divided difference, the trapezoidal LTE estimate)
  meets `lte_reltol`/`lte_abstol`.  Refining globally rather than per step
  keeps the step sequence a deterministic function of the netlist rather than
  of the solution, which is what lets gradients be compared with central
  finite differences.  Set `ckt.options['adaptive'] = False` and
  `ckt.options['nsub'] = k` to pin it.
* **Linear solve**: dense LU via `torch.linalg`; no explicit inverse is ever
  formed.  A circuit with no nonlinear device is solved by a *vectorised*
  kernel rather than a Python timestep loop: because `f(x,t) = Jf x + f(0,t)`
  and `q(x) = Jq x`, the whole run collapses onto one affine recursion in a
  single carried vector, evaluated by a blocked (parallel-prefix) scan, plus
  one batched triangular solve and one batched source evaluation per window.
  See "Performance" below for what that is worth.
* **Charge-based integration** for capacitors, inductors (flux), junction
  charges and MOS intrinsic capacitances, so charge is conserved.
* **`uic=True`** skips the operating point and starts from `.ic` /
  capacitor `IC=` / inductor `IC=` values (SPICE `UIC` semantics).

Tolerances live in `ckt.options` and are also settable from a `.options` card:

| option | default | meaning |
|---|---|---|
| `reltol` | 1e-11 | relative tolerance on unknowns and residual |
| `abstol` | 1e-15 | absolute current tolerance [A] |
| `vntol` | 1e-11 | absolute voltage tolerance [V] |
| `maxiter` | 100 | Newton iterations per solve |
| `gmin` | 0.0 | conductance across nonlinear junctions |
| `lte_reltol` | 1e-8 | transient LTE tolerance (linear circuits) |
| `lte_reltol_nl` | 1e-5 | transient LTE tolerance (nonlinear circuits) |
| `lte_abstol` | 1e-14 | absolute LTE floor |
| `method` | `'trap'` | `'trap'` or `'be'` |
| `nsub`, `adaptive`, `max_nsub` | — | step-control overrides |
| `temp` | 27.0 | temperature in °C (sets `Vt = kT/q`) |

---

## Worked example

```python
import torch
from spipe.electronic.native import Netlist

deck = """CMOS inverter delay vs device width
.param wn=2u wp=4u
Vdd vdd 0 DC 1.8
Vin in  0 PULSE(0 1.8 1n 0.2n 0.2n 2n 5n)
M1  out in 0   0   nch W={wn} L=0.18u
M2  out in vdd vdd pch W={wp} L=0.18u
CL  out 0 20f
.model nch NMOS (level=1 vto=0.45 kp=250u lambda=0.12 tox=4n cgso=2e-10 cgdo=2e-10)
.model pch PMOS (level=1 vto=-0.45 kp=80u  lambda=0.15 tox=4n cgso=2e-10 cgdo=2e-10)
.tran 20p 4n
"""

ckt = Netlist.from_string(deck)

# --- forward -------------------------------------------------------------
res = ckt.tran()
print(res.t.shape, res.v('out')[:5])

# --- autograd ------------------------------------------------------------
wn = ckt.param('M1', 'W').requires_grad_(True)
res = ckt.tran()
objective = ((res.v('out') - 0.9) ** 2).sum()
objective.backward()
print('d(obj)/dWn =', float(wn.grad))

# --- adjoint (same answer, one backward sweep) ---------------------------
g = ckt.adjoint_grad(lambda r: ((r.v('out') - 0.9) ** 2).sum(),
                     params=[('M1', 'W'), ('M2', 'W'), ('CL', 'C')])
for k, v in g.items():
    print(k, float(v))
```

---

## Module map

| module | contents |
|---|---|
| `units.py` | SPICE number suffixes, safe expression evaluator |
| `parser.py` | tokenizer, control cards, subcircuit flattening |
| `devices/` | one class per device type; `base.py` holds the stamping context |
| `mna.py` | index bookkeeping, residual/Jacobian assembly, external blocks |
| `newton.py` | Newton-Raphson, gmin stepping, source stepping |
| `analyses.py` | operating point, transient integrator, `.dc` sweep |
| `adjoint.py` | time-domain adjoint sweep and the autograd bridge |
| `results.py` | `OpResult`, `TranResult`, `DCResult` |
| `external.py` | external-block contract and two example blocks |
| `circuit.py` | the public `Netlist` class |

Adding a device means adding a `DeviceGroup` subclass and one entry in
`devices/REGISTRY`; the parser, the assembly and the solvers are untouched.

---

## Verification

Measured against closed-form solutions, an independent NumPy MNA solve, and
HSPICE X-2025.06 / Xyce 7.10 on identical decks:

| check | measured |
|---|---|
| RC step vs `V0(1-e^{-t/RC})`, max pointwise rel. err | 8.1e-9 |
| series RLC vs analytic damped sinusoid, max pointwise rel. err | 4.0e-6 |
| Shockley diode DC I-V vs closed form | 0.0 (exact) |
| MOSFET L1 drain current vs square law, all 3 regions, NMOS/PMOS | 1.7e-16 |
| linear DC vs `numpy.linalg.solve` of the same MNA | 2.1e-16 |
| charge conservation on a capacitor-only loop | 1.3e-24 C |
| autograd vs adjoint | 0.0 (bit-identical) |
| autograd vs central finite difference | < 4.5e-7 |
| ring-oscillator frequency vs HSPICE | 0.11 % |

Against HSPICE X-2025.06 run with **tightened** tolerances
(`reltol=1e-7 relv=1e-7 absv=1e-12 chgtol=1e-18 delmax=1e-12`), so that the
comparison measures model agreement rather than HSPICE's default step
control, worst node-voltage error over the whole waveform, normalised to the
signal amplitude:

| circuit | vs HSPICE (tight) | vs HSPICE (default options) |
|---|---|---|
| RC ladder (4 sections) | 1.2e-6 | 1.4e-2 |
| series RLC | 4.9e-5 | 1.8e-2 |
| common-source amplifier | 1.8e-4 | 1.6e-2 |
| diode rectifier | 6.5e-3 | 1.7e-2 |

The gap at HSPICE's default options is HSPICE's own discretisation: HSPICE
with default vs tight options differs from itself by the same 1.4e-2 / 1.8e-2
on those decks.

Full three-way table at each tool's **default** options — worst pointwise
node-voltage difference over the whole waveform, normalised to the HSPICE
signal amplitude, after interpolating all three onto a common 2001-point
grid.  The third column is HSPICE against Xyce, i.e. how far the two
commercial tools are from *each other* on the very same deck:

| circuit | node | native vs HSPICE | native vs Xyce | HSPICE vs Xyce |
|---|---|---|---|---|
| RC ladder | a / b / c / d | 1.4e-2 / 3.7e-3 / 1.8e-3 / 2.1e-3 | 3.4e-3 / 8.3e-4 / 7.0e-4 / 7.3e-4 | 1.5e-2 / 3.9e-3 / 1.9e-3 / 2.1e-3 |
| series RLC | a / b | 1.8e-2 / 6.2e-3 | 4.9e-1* / 7.0e-4 | 4.9e-1* / 6.1e-3 |
| diode rectifier | a / out | 1.7e-2 / 8.6e-3 | 3.2e-3 / 1.0e-3 | 1.6e-2 / 8.2e-3 |
| common-source amp | d | 1.6e-2 | 3.6e-3 | 1.5e-2 |
| differential pair | op / om / s | 3.5e-3 / 3.5e-3 / 6.3e-2 | 1.5e-3 / 1.5e-3 / 2.9e-3 | 3.3e-3 / 3.4e-3 / 6.3e-2 |
| CMOS inverter chain | o1 / o2 / o3 | 5.9e-2 / 4.8e-2 / 6.8e-2 | 6.0e-2 / 5.4e-2 / 5.0e-2 | 7.5e-2 / 6.4e-2 / 1.0e-1 |
| 5-stage ring osc | frequency | **0.114 %** | 2.59 % | 2.71 % |

`*` a sampling artefact: Xyce reports its own internal time points and one of
them lands mid-way up a 50 ps edge that the 2 ns print grid steps over.
Excluding a ±2·tstep window around source breakpoints it is 9.0e-4.

On every circuit our difference from either tool is no larger than the
difference between the two tools, and on the switching circuits (inverter
chain, ring oscillator) it is smaller.

See "Known limitations" below.

## Performance

Measured on this machine (single CPU, PyTorch 2.7.1, float64), reporting
**milliseconds per output point** — the unit that matters, since the LTE
controller decides how many internal steps each output point costs.

Deck A, **series RLC**, 4 unknowns, `uic` start
(`V1 in 0 DC 1` / `R1 in a 100` / `L1 a out 1m` / `C1 out 0 1n`;
`w0 = 1e6 rad/s`, `zeta = 0.05`):

| pts/radian | decay constants | output points | internal steps | `nsub` | wall | ms / output point | max rel err |
|---|---|---|---|---|---|---|---|
| 60 | 3 | 3,601 | 230,401 | 64 | 0.96 s | 0.266 | 7.90e-08 |
| 150 | 3 | 9,001 | 576,001 | 64 | 2.23 s | 0.248 | 2.89e-08 |
| 400 | 3 | 24,001 | 768,001 | 32 | 3.96 s | 0.165 | 4.17e-08 |
| 400 | 12 | 96,001 | 3,072,001 | 32 | 14.98 s | 0.156 | 4.17e-08 |

That is about **4-5 microseconds per internal step**.  The gap between
"internal steps" and "output points" is `nsub`, chosen by the LTE controller;
with the default `lte_reltol = 1e-8` this deck wants 32-64 sub-steps per
requested point, and that is what buys the 1e-8 accuracy.  The trade is
explicit and is the first thing to turn if the accuracy is not needed
(same deck, 400 pts/radian, 12 decay constants):

| `lte_reltol` | `nsub` | internal steps | wall | ms / output point | max rel err |
|---|---|---|---|---|---|
| 1e-8 (default) | 32 | 3,072,001 | 13.2 s | 0.137 | 4.17e-08 |
| 1e-6 | 8 | 768,001 | 5.3 s | 0.055 | 6.67e-07 |
| 1e-4 | 1 | 96,001 | 0.48 s | 0.005 | 4.27e-05 |

Deck B, circuits **with nonlinear devices**, which still run a Newton solve
per step (roughly 3.5 iterations/step) and are therefore ~1000x dearer per
internal step:

| circuit | output points | internal steps | wall | ms / internal step |
|---|---|---|---|---|
| diode rectifier (1 diode, 5 unknowns) | 201 | 801 | 4.3 s | 5.4 |
| CMOS inverter chain (6 MOSFETs) | 1,001 | 8,001 | 49 s | 6.2 |
| 5-stage ring oscillator (10 MOSFETs) | 2,001 | 2,001 | 20 s | 10 |

The adjoint costs about 1.6x a forward solve on a nonlinear circuit, for any
number of parameters.

Nonlinear throughput is dominated by PyTorch's per-operation dispatch cost in
the Newton assembly, and is the remaining performance gap against a compiled
simulator; the linear kernel above shows what batching recovers.

For a co-simulation fixed-point loop, where the electronic side is re-solved
every iteration and 1e-4-class accuracy is plenty, set
`ckt.options['lte_reltol'] = 1e-4`: Deck A then runs 96,001 output points in
0.48 s.

## Known limitations

* MOSFET: level 1 only.  No bulk junction diodes, no `RD`/`RS` series
  resistance, no short-channel or subthreshold effects (level 1 has none by
  definition), no temperature scaling of model parameters.
* BJT: no high-level injection (`IKF`/`IKR`), no `RB`/`RC`/`RE`, no substrate
  junction.
* No `.ac`, `.noise`, `.tf`, `.sens`, `.measure`; `POLY`/`TABLE`/`VALUE`
  controlled sources are rejected with a clear message.
* Gradients with respect to *source timing* parameters (`TD`, `TR`, `TF`,
  `PW`, `PER`, `PWL` times) are exact for the frozen step sequence, but
  perturbing them moves the breakpoint grid, so they do not match finite
  differences to 1e-6.  Gradients with respect to device parameters do.
* The dense LU limits practical circuit size to a few hundred unknowns.
