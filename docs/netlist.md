# The SPIPE netlist

Everything SPIPE simulates is described by a text netlist. This page defines every
directive and every device, so nothing in the README or the examples is left unexplained.

One file may hold two sections:

```
.electronic
  ... ordinary SPICE, plus one SPIPE-specific card (.sensparam) ...

.photonic
  ... the optical network ...
```

A photonic-only netlist needs no section headers — hand the lines straight to
`Photonic(...)`. A file with both sections goes to `Circuit(path, spice_exe=...)`.

The two domains are coupled **implicitly, through shared node names**. A node written by
the electronics and read by a modulator carries the drive; a node written by a
photodetector and read by the electronics carries the photocurrent. SPIPE cuts the circuit
at exactly those nodes — there is no explicit "connect" statement.

---

## The `.photonic` section

### Directives

| directive | meaning |
|---|---|
| `.mode neff=<n> ng=<n> wl=<m>` | the waveguide mode. `neff` is the effective index at the reference wavelength `wl` (in metres), and `ng` the group index. The index at other wavelengths is linear in λ, `n(λ) = ng + (neff − ng)·λ/wl`, so a device of length `l` has phase `β·l = (ω/c)·n(λ)·l`; a resonance you design at one frequency is placed relative to `wl`, not to the `.freq` grid. Devices inherit these unless they override them. |
| `.freq <start> <stop> <count>` | optical frequencies to solve at, in Hz. `count = 1` is a single CW carrier. More than one point means that many **independent** channels — see *Incoherent by default* below. The grid is available afterwards as `Photonic(...).omega`, in rad/s. With `count = 1` only `start` is used. |
| `.source <A>@<node> ...` | the laser. `<A>` is a complex amplitude launched at `<node>`; list one per input. A detector turns `|A|²` into `r0·|A|²` amperes. Without a power budget `|A|²` is optical power in **watts**: with `r0=1` (A/W), `1.0@a1` is 1 W of light and 1 A of photocurrent, so a realistic 1 mW laser is `0.0316@a1`. Optional `power=<W>` and `eff=<0..1>` add a power budget, which changes that unit (see *The power budget* below). One amplitude per node, applied to every `.freq` channel alike. |
| `.prob <node>` | also report the complex optical field at `<node>`, without disturbing it. Each probe returns the two waves at that node: index 0 travels **into**, index 1 **out of**, the first device listed on that node. Without any `.prob` line the `probes` dict comes back empty. |

`.prob` is spelled without the final `e`.

### Device lines

```
<name> <optical ports...> [<electrical node> <level>] [key=value ...]
```

The **instance name's prefix picks the model**: a line starting `mzm0` instantiates an
`mzm`, a line starting `pd1` instantiates a photodetector. Names are otherwise free.

Passive devices:

| prefix | ports (in, out) | what it is | parameters |
|---|---|---|---|
| `wg` | 1, 1 | waveguide | `l=` length in metres, `alpha=` field transmission of the whole length (1 = lossless; see *Loss* below) |
| `ps` | 1, 1 | phase shifter | `ps=` phase in radians (`0.5pi` is accepted) |
| `mzi` | 2, 2 | a **coupler** of variable ratio (despite the name, a single coupler, not an interferometer): field through `cos θ`, cross `i·sin θ` (the `i` is the usual quarter-wave phase of a coupler), so power cross-coupling `κ² = sin²θ` — this is how to build a ring (see below). `l=` (default 0) only adds a common propagation phase and loss to both paths. | `theta=` coupling angle in rad, `l=`, `alpha=` |
| `pbum` | 2, 2 | programmable building-unit: the 2×2 Mach–Zehnder cell a photonic **mesh** is tiled from — coupler, a phase shift on each arm, coupler | `theta=` phase on the upper arm and `phi=` on the lower, in rad; `l=` arm length in m (**required**; 0 is allowed); `alpha=`; `cp_left=`, `cp_right=` coupler angles (default π/4, 50:50). With 50:50 couplers the split is set by `theta − phi`: 0 → all cross, π → all bar. |
| `splitter1to1` … `splitter1to4` | 1, N | ideal N-way power splitter | — |
| `wdm1to1`, `wdm1to2`, `wdm1to4` | 1, N | wavelength de-multiplexer (N = 1, 2 or 4; there is no `wdm1to3`): channel *k* of the `.freq` grid leaves by output port *k*. The grid must therefore have **exactly N points** — `wdm1to2` with a single `.freq` point is an error, and says so. | — |

Active devices — these take an **electrical node** and an **electrical load level**, and
are driven by a voltage:

| prefix | ports | what it is | key parameters |
|---|---|---|---|
| `mzm` | 2, 2 | Mach–Zehnder **modulator**, push–pull. `V_π` is literally the voltage that takes it from full-on to full-off. | `vpi=` (> 0, default 2), `vbias=`, `il=` insertion loss in dB (≥ 0), `er=` extinction ratio in dB, `chirp=`, `tau=` response time, `order=`; less common: `act_l=`, `kappa1=`, `kappa2=`, `wgu_l=`, `wgl_l=`, `dacoeff0=`… (see below) |
| `modm` | 2, 2 | the paper's Eq. 5 modulator: a **variable-ratio coupler**, not a push–pull MZI. Kept unchanged for reproducibility. Its `V_π` differs from `mzm`'s by 2× and its bias point by π/2 — pick the one that matches your device. | `coeff0=`, `coeff1=`, … (see below), `act_l=` active length in m, `wgu_l=`, `wgl_l=`, `alpha=` |
| `modp` | 1, 1 | phase-only modulator | `coeff0=`, `coeff1=`, … (see below), `act_l=` active length in m (required), `wg_l=`, `alpha=` |
| `pd` | 1 optical → 1 electrical | photodetector. It absorbs the light, so it must sit on an **output** — a node with one device; use `.prob` to look inside a circuit. | `r0=` responsivity in A/W (required), `bw=` bandwidth in Hz, `idark=` dark current in A, `noise=`, `temp=`, `rload=`, `inoise=`, `coherent=1` — see *Photodetector bandwidth and noise* below |

#### `mzm`: the drive

The drive sets the phase difference between the arms, `Δφ = π·(V − vbias)/vpi`. At
`V = vbias` all the light entering the first input (`a1` in `mzm0 a1 a2 b1 b2 ...`) leaves by the
second output (`b2`); at `V = vbias + vpi`, by the first (`b1`); halfway between, it splits
evenly.

#### `mzm`: insertion loss and extinction ratio

`il=` is the loss of a perfectly balanced device: with the default infinite `er`, peak
transmission is exactly `10^(−il/10)`. A finite `er` is modelled as an amplitude imbalance
between the two arms, which is what limits extinction. The weaker arm carries
`(1 − ε)/(1 + ε)` of the stronger one's amplitude, with `ε = 10^(−er/20)`. That gives exactly
the requested on/off ratio, and the device stays passive. The imbalance costs light, as it
does in a real device. With a finite `er`, peak transmission is `10^(−il/10) / (1 + ε)²`:
about 2.4 dB below `il` at `er = 10`, 0.8 dB at 20, and 0.3 dB at 30.

The less common `mzm` parameters, with their defaults:

| parameter | default | meaning |
|---|---|---|
| `act_l=` | 1 mm | active length. The phase swing is set by `vpi`, so `act_l` only scales the drive-dependent loss (`dacoeff*`). But it counts toward the circuit's **optical delay**, so a 1 mm modulator alone is a 13 ps delay (at `ng=4`) — enough to trigger the quasi-static warning below 130 ps between samples. Give the real length. |
| `kappa1=`, `kappa2=` | π/4 | input and output coupler angles; π/4 is 50:50 |
| `wgu_l=`, `wgl_l=` | 0 | extra passive length in the upper / lower arm (arm imbalance), in m |
| `dacoeff0=`, `dacoeff1=`, … | 0 | drive-dependent excess loss, a polynomial in the drive voltage, in 1/m, 1/(m·V), … |

#### `modm` and `modp`: the index-change coefficients

These two models describe the drive's effect as a polynomial in the drive voltage `V`, one
numbered coefficient per power:

```
dn/neff = coeff0 + coeff1*V + coeff2*V^2 + ...
```

The phase over the active length is then `β · act_l · (dn/neff)` with `β = neff·ω/c`. So the
coefficients describe the **relative** effective-index change, not `dn` itself: for an index
change of `dn/dV` per volt, write `coeff1 = (dn/dV) / neff`. Give at least one numbered
coefficient; a line without any is an error. A plain `coeff=` is not read.

`act_l` is the length over which the drive acts. It sets the modulation phase and counts
toward the circuit's optical delay, but the passive propagation phase `β·act_l` itself is not
included; add a `wg_l=` (or a `wg`) if that phase matters.

#### Building a ring resonator

There is no dedicated ring device: a ring is a coupler whose cross port is closed by a waveguide.

```
mzi0 a1 a2 b1 b2 theta=0.3176   # directional coupler, t = cos(theta) = 0.95
wg0  b2 a2 l=62.83e-6 alpha=0.97  # the ring: round-trip length and field transmission
```

Light enters at `a1` and leaves at `b1`. This reproduces analytic ring theory (free spectral
range, linewidth, extinction, drop port, intracavity build-up) to about 1e-12 —
`test/tb/tb01_photonic_algebra.py` checks the all-pass spectrum. To look at the field inside
the ring, use `.prob b2`, not a `pd`.

#### Loss

`alpha` is the field transmission of a device's propagation section — the length `l` (or
`wg_l`, `wgu_l`, `wgl_l`). With that length zero there is no section, and `alpha` has no effect
(SPIPE warns). For a lossy junction or a coupler's excess loss, add a `wg` with a negligible
length and the loss you want, e.g. `wg1 x y l=1e-12 alpha=0.977`: its phase and delay are
negligible, and `alpha` applies. `alpha` must be between 0 and 1; a negative value is an error.
`alpha > 1` would be gain; SPIPE warns. Loss is per device, not per unit length: for a
waveguide of `L` metres with `x` dB/cm, `alpha = 10^(−x · L/0.01 / 20)`.

### Electrical load levels

An active device presents a load to the circuit driving it. The `level` token selects
which equivalent circuit SPIPE inserts:

| level | on a modulator (`mod_level*`) | on a detector (`pd_level*`) |
|---|---|---|
| `level1` | near open circuit (1 GΩ) — use when you do not want the load to matter | the photocurrent into 2 fF, reaching the output through a 1 kΩ series resistor. It has no path to ground of its own: give the output node a load. |
| `level2` | — | diode junction behind a 70 dB ideal amplifier |
| `level3` | realistic depletion-mode RC: series access resistance, junction capacitance, pad capacitance | — |
| `debug` | bare resistor | bare current source |

You can add your own with `spipe.electronic_register(kind, level, subckt_text)`, where `kind`
is `'mod'` or `'pd'`. The text is a SPICE subcircuit named `<kind>_<level>`, whose first two ports
are the device's electrical node and ground, e.g. `.subckt pd_level4 n ground ...`. A detector's
subcircuit must contain the placeholder `Ipd <node+> <node->`, which SPIPE replaces with the
simulated photocurrent. `examples/derived/n3_tia.py` does exactly this to swap in a real
transimpedance amplifier. `spipe.electronic_reset()` removes what you registered.

### Photodetector bandwidth and noise

With only `r0=`, a detector is ideal: `I = r0·|E|²`, with infinite bandwidth and no noise.

`bw=<Hz>` gives it a single-pole response with that 3 dB frequency. It also switches on
**shot and thermal noise**, since a real detector has both. The noise is white at the diode,
with one-sided density `2q(I + idark) + 4kT/rload` A²/Hz, and passes through the same pole as
the signal. Its variance is therefore that density times the pole's noise bandwidth,
`(π/2)·bw`. Neighbouring samples are correlated as a real filtered signal would be, so
results do not depend on how finely `.tran` samples, once it resolves `bw`.

| parameter | default | meaning |
|---|---|---|
| `idark=` | 0 | dark current, A. Adds to the current and to its shot noise. |
| `noise=` | 1 | `0` keeps the bandwidth and drops the noise |
| `temp=`, `rload=` | 300 K, 50 Ω | set the thermal-noise term `4kT/rload`. `rload=` is **only** a noise parameter: the load the detector actually drives is whatever the `.electronic` section connects to it. |
| `inoise=` | — | input-referred noise density of the front end, A/√Hz; replaces the `temp=`/`rload=` term. Use it for a real amplifier. |
| `r1=`, `r2=`, …, `wl=` | — | frequency-dependent responsivity: `R(ω) = r0 + r1·(ω − ω0) + r2·(ω − ω0)² + …` about the angular frequency `ω0 = 2πc/wl`. So `r1` is in A/W per rad/s, and a realistic value is tiny (around 1e-16). `wl=` is required with them, and a negative responsivity gives a warning. |

Some limits to be aware of:
- **The electronic side adds no noise.** Amplifier noise enters only through `inoise=`.
- **One realisation per run.** Every iteration of one `Circuit.simulate()` sees the same
  noise, and `simulate(seed=...)` picks which realisation that is; the same seed gives the
  same result. A Monte Carlo study loops over seeds. `Photonic.simulate` draws fresh noise at
  every call, unless you pass it a `seed=`.
- **Gradients are for that one realisation.** Set `noise=0` to optimise the noiseless
  response.

A detector with no modulator anywhere in the circuit still takes a time axis:
`Photonic(netlist).simulate(t)` returns one sample per entry of `t`, which is how you see a
passive receiver's noise.

### Incoherent by default

With several `.freq` points the photodetector sums `|E(ω)|²` over them. That is right when
they are independent WDM carriers. If you mean them as one modulated signal whose cross
terms matter, put `coherent=1` and a `bw=` on the `pd` line. See `docs/scope.md`.

### The power budget

`power=` and `eff=` on the `.source` line turn on the `power` dict that `simulate()`
returns. They also fix what one unit of `|A|²` means:
- `power=` is the laser's electrical draw **per unit of `Σ|A|²`**, in watts;
- `eff=` is its wall-plug efficiency;
- so one unit of `|A|²` is `power·eff` watts of light.

The paper's decks write `.source 1.0@n1 power=0.005w eff=0.2`: a unit amplitude that stands for
1 mW of light, from a laser drawing 5 mW.

A detector's `r0=` is then in amperes per that unit. To model a real responsivity of `R` A/W,
write `r0 = R·power·eff`. Otherwise the detector and the power report describe different
lasers: `1.0@a1 power=1 eff=0.2` with `r0=1` reports 0.2 W launched, while the detector
reads 1 A.

---

### Numbers and units in the `.photonic` section

Values may carry SPICE scale factors — `f p n u k meg g t` — or units, case-insensitive:
lengths `nm um mm cm` (metres otherwise), times `fs ps ns us ms s`, frequencies
`hz khz mhz ghz thz`, `pi` for angles (`theta=0.25pi`), powers `mw w`, currents `ma a`.
**A bare `m` is an error** here: it would be metres in physics and milli in SPICE, and the
`.electronic` section of the same file is SPICE, so SPIPE refuses to guess.

## The `.electronic` section

Ordinary SPICE — device cards, `.model`, `.include` and so on, in the dialect of whichever
backend you chose — with two exceptions: SPIPE reads `.tran` itself, and adds one card of its
own, `.sensparam`.

### `.tran` — the co-simulation time grid

```
.tran <start> <stop> <points>
```

**This is not SPICE's `.tran <tstep> <tstop>`.** It defines the time samples at which the two
domains exchange signals: `points` evenly spaced samples from `start` to `stop`, in seconds.
SPICE suffixes are accepted (`.tran 0 40n 401` is 401 samples over 40 ns; `m` is milli). A
two-argument, SPICE-style line is an error rather than being misread.

Choose `points` so the grid resolves the fastest edge you care about. The electronic engine
steps internally as finely as it needs, but the modulator sees the drive only at these samples.

### The starting state

A transient starts from the SPICE `UIC` state: every capacitor with a grounded terminal
starts at 0 V (or its `IC=`), every inductor at 0 A, unless a `.ic` says otherwise — and a
voltage source always sets its node. So the first samples show the circuit charging up. The
built-in engine can start from the DC operating point instead:
`spipe.config['native_uic'] = False`.

### `.sensparam` — declaring differentiable device parameters

```
.sensparam MN1:W MN1:L RL:R
```

**This is SPIPE's own card, not a SPICE one.** It names the electronic device parameters
you want gradients with respect to, as `DEVICE:PARAMETER`. Each one becomes a PyTorch leaf
tensor you fetch with `Circuit.param('mn1', 'W')`, and `loss.backward()` fills in its
`.grad`. SPIPE translates the card into whatever the backend needs — Xyce's `.SENS
... param=`, or nothing at all for the built-in engine, which differentiates natively.

Without a `.sensparam` card SPIPE runs an ordinary simulation and builds no autograd graph.
That is the whole switch: declaring a parameter is what makes the run differentiable.

Device names are matched case-insensitively; parameter names follow SPICE (`W`, `L`, `R`).

### `.print tran` — reading electronic node voltages

```
.print tran v(vdrv) v(vo1)
```

Every node named here comes back in the first dict `Circuit.simulate()` returns, keyed in
lower case (`.print tran V(VO1)` gives `probes_e['v(vo1)']`), one value per `.tran` sample. Without the line
that dict is empty.

The probes are differentiable. On the built-in engine and Xyce their gradient is the total
derivative through the whole loop — a detector's output voltage depends on `W` through the
driver, the modulator and the light, and all of that is included. On HSPICE it covers only
the electronic circuit's direct dependence on the parameter, not the path through the light.

---

## A complete two-domain example

`examples/link_driver_mzm.sp`, a CMOS inverter driving a modulator:

```
.electronic
.model nch NMOS (LEVEL=1 VTO=0.7  KP=120u LAMBDA=0.02)
.model pch PMOS (LEVEL=1 VTO=-0.7 KP=40u  LAMBDA=0.02)
Vdd vdd 0 3.0
Vin g   0 PULSE(0 3 1n 0.2n 0.2n 10n 20n)
MN1 vdrv g 0   0   nch W=8u  L=0.5u
MP1 vdrv g vdd vdd pch W=16u L=0.5u
* a detector output needs a DC path to ground
Rload1 vo1 0 1k
Rload2 vo2 0 1k
.sensparam MN1:W MN1:L
.tran 0 4e-8 40

.photonic
.mode neff=2.35 ng=4.0 wl=1550e-9
.freq 193.1e12 193.1e12 1
.source 1.0@a1 0.0@a2
mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0
pd1 b1 vo1 level1 r0=1.0
pd2 b2 vo2 level1 r0=1.0
```

`vdrv` is written by the electronics and read by `mzm0`; `vo1` and `vo2` are written by the
detectors and read by the electronics. Those three nodes are the electronic–photonic
interfaces, and nothing else declares them as such.

A photodetector's output node must have a DC path to ground — hence `Rload1` and `Rload2`.
Without one the photocurrent charges the detector's capacitance without limit. The built-in
engine refuses such a circuit and names the node.

With `1.0@a1` (1 W of light) and `r0=1`, this example makes up to 0.5 A of photocurrent and
500 V across the 1 kΩ loads. The numbers are unrealistic but harmless here, because the
example is about the gradient. For realistic levels use a milliwatt laser (`0.0316@a1`).
