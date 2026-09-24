# SPIPE

**Differentiable SPICE-level co-simulation for integrated photonics and electronics.**

A photonic link is half optics and half circuitry, and the two are normally simulated by
different tools that do not talk to each other. SPIPE simulates both at once: SPICE for the
electronics, the scattering-matrix method for the photonics, coupled through the modulators
and photodetectors that join them. Nothing has to be rewritten in Verilog-A.

Because the whole chain is differentiable, you can ask for the derivative of an **optical**
output with respect to an **electronic** device parameter — how much the light at the end of
the link changes when you make a transistor wider. That is the question a circuit designer
actually needs answered, and it is what SPIPE was built for.

> Z. Gao, J. Gu, L. Daniel, R. Rohrer, D. S. Boning,
> "SPIPE: Differentiable SPICE-Level Co-Simulation Program for Integrated Photonics and
> Electronics," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and
> Systems*, vol. 45, no. 3, 2026. doi:[10.1109/TCAD.2025.3597958](https://doi.org/10.1109/TCAD.2025.3597958)

---

## Install

```bash
git clone https://github.com/zhengqigao/spipe.git
cd spipe
pip install -e .
```

Python ≥ 3.9, plus PyTorch, NumPy and SciPy. `pip install -e ".[examples]"` adds matplotlib
and h5py for the example scripts.

That is everything. SPIPE ships its own analog circuit simulator, so the package alone runs
every photonic circuit and every electronic–photonic co-simulation. External simulators are
optional and only used for cross-checks — see *Backends* below.

## Quick start

A netlist is a list of lines. This one splits light in two and measures both halves:

```python
from spipe.photonic.photonic import Photonic

netlist = [l + "\n" for l in [
    ".mode neff=2.35 ng=4.0 wl=1550e-9",   # the waveguide mode
    ".freq 193.1e12 193.1e12 1",           # one optical carrier at 193.1 THz
    ".source 1.0@a1 0.0@a2",               # light in at node a1, none at a2
    "mzi0 a1 a2 b1 b2 theta=0.25pi",       # a 50:50 coupler
    "pd1 b1 vo1 level1 r0=1.0",            # photodetector on b1
    "pd2 b2 vo2 level1 r0=1.0",            # photodetector on b2
]]

photocurrent, probes, power = Photonic(netlist).simulate()

photocurrent          # tensor([[0.5, 0.5]])  -- half the light to each detector
```

`simulate()` always returns the same three things:

| returned | shape | meaning |
|---|---|---|
| `photocurrent` | `(n_time, n_detectors)` | what each `pd` line measures, in netlist order. Summed over the optical channels, so with N channels a fully-transmitting path reads N, not 1. |
| `probes` | `{node: (n_time, n_freq, 2)}` | the complex field at nodes you asked about with a `.prob` line. The last axis holds the two waves at that node, `[into, out of]` **the first device listed on that node** — so on a node joining two devices, swapping their netlist lines swaps the two entries. Empty unless you add `.prob`. |
| `power` | `{name: (n_time,)}` | the laser power budget, in watts, if you characterise the laser with `power=` and `eff=` on the `.source` line; otherwise just `{'photonic': None}`. |

`n_time` is 1 for a passive circuit like this one.

A **modulator** is driven by a voltage, so it needs a time axis and one drive column per
modulator:

```python
import torch
from spipe.photonic.photonic import Photonic

netlist = [l + "\n" for l in [
    ".mode neff=2.35 ng=4.0 wl=1550e-9",
    ".freq 193.1e12 193.1e12 1",
    ".source 1.0@a1 0.0@a2",
    # a Mach-Zehnder modulator. `vdrv` is the electrical node carrying the drive;
    # `level3` is the electrical load it presents; vpi is its half-wave voltage.
    "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
    "pd1 b1 vo1 level1 r0=1.0",
    "pd2 b2 vo2 level1 r0=1.0",
]]

t     = torch.linspace(0, 1e-8, 5)                 # seconds
drive = torch.linspace(0, 2.0, 5).reshape(-1, 1)   # volts, 0 -> V_pi

photocurrent, probes, power = Photonic(netlist).simulate(t, drive)

photocurrent[:, 0]   # tensor([0.0000, 0.1464, 0.5000, 0.8536, 1.0000])
                     # full-off at 0 V to full-on at V_pi = 2 V
```

Everything returned is differentiable with respect to the drive, and with respect to any passive
parameter you make trainable with `Photonic.param('pbum0', 'theta')`. That is how you train a
programmable mesh; see [docs/differentiability.md](docs/differentiability.md).

Every directive and device is defined in **[docs/netlist.md](docs/netlist.md)**.

## Two things you can simulate

| you have | object | returns |
|---|---|---|
| a photonic netlist only | `Photonic(netlist)` | **3** — `photocurrent, probes, power` |
| one file with an `.electronic` **and** a `.photonic` section | `Circuit(path, spice_exe=...)` | **5** — `probes_e, probes_p, photocurrent, drive, power` |

`Photonic` solves the optical network and you supply the modulator drive yourself, as
above. `Circuit` additionally runs the electronics, feeds the photocurrents back, and
iterates until the two domains agree — so it also returns the electrical probes and the
converged drive. Both are called `simulate()`. The electrical probes are the node voltages
named on a `.print tran v(node) ...` line, keyed `'v(node)'`; without that line `probes_e`
is empty.

## End-to-end differentiability

This is what SPIPE is for:

```
   transistor W  ──►  driver  ──►  modulator  ──►  photodetector  ──►  |E_out|²
   └──────────────────────  d|E_out|² / dW  ◄───────────────────────────┘
```

You describe both domains in one file and call one function. Mark the electronic device
parameters you want gradients for with a `.sensparam` line:

```
.sensparam MN1:W MN1:L
```

`.sensparam` is SPIPE's own netlist card. It names device parameters as
`DEVICE:PARAMETER`, and each one becomes a PyTorch tensor you can read `.grad` from.
Declaring a parameter is also the switch that turns gradients on: with no `.sensparam`
line, `simulate()` runs an ordinary simulation and builds no graph.

With `examples/link_driver_mzm.sp` — a CMOS inverter driving a modulator, listed in full in
[docs/netlist.md](docs/netlist.md):

```python
from spipe import Circuit

ckt = Circuit("examples/link_driver_mzm.sp", spice_exe="native")

w = ckt.param("mn1", "W")                # the transistor width, as a tensor

_, _, photocurrent, drive, _ = ckt.simulate()

loss = (photocurrent[:, 0] ** 2).sum()   # any optical figure of merit
loss.backward()

w.grad                                   # d(loss)/dW = 29.19573838
```

That is the entire program. Checked against finite differences over the whole chain, the
gradient agrees to **4.6e-08** — as closely as a finite difference through a transient simulation can check it (the sweep is in [docs/differentiability.md](docs/differentiability.md)).

"Exact" means exact for the circuit **as simulated**. This example takes 40 samples, and the
built-in engine uses at most 8 internal steps per sample. The loss here depends on W only
through the sub-millivolt tails after each edge, which that discretisation gets to about 14%:
with `spipe.config['native_nsub'] = 64` the same gradient is 25.60. When the number itself
matters, raise `native_nsub` or sample more finely
([docs/backends.md](docs/backends.md)).

There is only **one** simulation call, and **`.sensparam` is the only switch**. Every
parameter it names comes back from `ckt.param(...)` already marked `requires_grad=True`, so
there is nothing to set in Python. Without a `.sensparam` line, `simulate()` runs an ordinary
simulation and builds no graph. The returned values are the same either way.

The mechanism, and why it stays cheap even with feedback, is in
[docs/differentiability.md](docs/differentiability.md).

## Backends

SPIPE can drive three electronic engines. `Circuit(..., spice_exe=...)` picks one.

| backend | availability | gradients |
|---|---|---|
| `native` | **ships with SPIPE**, nothing to install | **exact** |
| `xyce` | free and open source ([Xyce](https://xyce.sandia.gov/), GPL-3.0, Sandia) | numerical, and unreliable on transistor widths |
| `hspice` | commercial (Synopsys) | finite differences over re-runs |

**Use `native` for gradients.** It is a full analog simulator — R, L, C, sources,
controlled sources, diodes, BJTs, level-1 MOSFETs and switches, with DC and transient
analysis and adaptive time stepping — and it differentiates exactly. The other two are
there so you can check SPIPE against a tool you already trust. The caveats that come with
them are in [docs/backends.md](docs/backends.md).

To see what you have:

```bash
which Xyce            # capital X
which hspice
which interconnect    # Lumerical INTERCONNECT, lower case -- optional, photonics only
```

Anything that prints a path is ready to use.

## Does it give the right answer?

`test/` answers that automatically. Every check compares against something known to be
true — a closed-form solution, a conservation law, an analytically known derivative, or a
result from an independent simulator:

```bash
python test/run_all.py            # full suite, one PASS/FAIL verdict
python test/run_all.py --quick    # the fast subset
python test/run_all.py --list     # what each bench guards
```

**443 checks, and the exit status is 0 only if every one passed**, so it drops straight
into CI. `--quick` runs the fast ~310 of them in about a minute and a half. Almost all of it needs
nothing but PyTorch: the handful of checks that call HSPICE, Xyce, Lumerical INTERCONNECT or a
GPU **skip themselves** when that tool is not installed, rather than failing — so on a bare
machine you will see a slightly smaller total and a few skips. A few highlights:

| check | measured |
|---|---|
| energy conservation on a lossless circuit **with an optical loop** | error **exactly 0** |
| a photonic mesh vs **Lumerical INTERCONNECT** | agrees to `5e-7`, the limit of INTERCONNECT's own output precision |
| the built-in circuit engine vs closed-form solutions | to machine precision |
| the built-in engine vs **HSPICE and Xyce** on identical netlists | within how much those two disagree with each other |
| gradient of the optical output with respect to a transistor width, vs finite differences over the whole chain | agrees to `4.6e-08`, the limit finite differences can resolve |

The numbers and how they were taken: [docs/validation.md](docs/validation.md).

## Examples

`examples/` is documentation you can run. Each script is heavily commented and prints a
report explaining what it shows. Unlike `test/`, they assert nothing — they show you
numbers and leave the judgement to you.

```
examples/
├── mesh_vs_lumerical.py    photonic mesh checked against Lumerical INTERCONNECT
├── oeo_electronic.py       optoelectronic oscillator (electronic-delay loop)
├── bistable_latch.py       optical bistability -- two stable states
├── bias_control.py         automatic modulator bias control at quadrature
├── oeo_optical_delay.py    a case the default mode cannot represent, diagnosed
├── link_driver_mzm.sp      the two-domain netlist used above
├── derived/                the paper's circuits on level-1 devices: any engine, no PDK needed
└── paper/                  the figures from the TCAD paper
```

See [examples/README.md](examples/README.md) for what each one measured when it was
written, and for the one thing that needs an external download: `paper/ptc_hspice/` uses
the SkyWater SKY130 foundry models, which are third-party and not shipped here. Nothing
else needs them.

## What SPIPE does not model

SPIPE solves the photonic network in **steady state at each time sample**. That is exact
when light crosses the circuit far faster than the modulation changes, and it is what makes
the method fast. Outside that regime the limitation is structural rather than gradual: with
no optical memory there is no delay, so resonator ring-up and delay-set oscillation are
**absent** rather than approximated.

SPIPE is explicit about this rather than quiet. It warns, with numbers, when a circuit is
approaching the boundary; `mode='envelope'` lifts the restriction by giving the network a
real impulse response; and `examples/oeo_optical_delay.py` shows an oscillator the default
mode cannot represent. The library itself only warns; the example then stops and prints why,
instead of reporting a plausible wrong number.

The full discussion, including two further assumptions worth knowing about:
[docs/scope.md](docs/scope.md).

## Documentation

| | |
|---|---|
| [docs/netlist.md](docs/netlist.md) | every directive and device, defined |
| [docs/differentiability.md](docs/differentiability.md) | how the gradients are computed |
| [docs/backends.md](docs/backends.md) | choosing an electronic engine, and their traps |
| [docs/scope.md](docs/scope.md) | the physics assumptions and where they bind |
| [docs/envelope.md](docs/envelope.md) | optical memory: delays and resonators in time (`mode='envelope'`) |
| [docs/performance.md](docs/performance.md) | scaling, memory, the GPU, precision, and every setting |
| [docs/validation.md](docs/validation.md) | every measured number |

## Layout

| path | contents |
|---|---|
| `src/spipe/` | the package |
| `src/spipe/photonic/` | scattering-matrix solver, device models, envelope propagation |
| `src/spipe/electronic/` | the built-in simulator, plus HSPICE and Xyce drivers |
| `src/spipe/core/` | the electronic–photonic coupling |
| `test/` | the automated regression suite |
| `examples/` | annotated, runnable demonstrations |
| `docs/` | the pages listed above |
| `scripts/` | helpers (e.g. locating the SKY130 models) |

## Citing

```bibtex
@article{gao2026spipe,
  author  = {Gao, Zhengqi and Gu, Jiaqi and Daniel, Luca and Rohrer, Ron and Boning, Duane S.},
  title   = {{SPIPE}: Differentiable {SPICE}-Level Co-Simulation Program for Integrated
             Photonics and Electronics},
  journal = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  volume  = {45},
  number  = {3},
  year    = {2026},
  doi     = {10.1109/TCAD.2025.3597958}
}
```

## Roadmap

Honest about what is not done yet. These are open work items, roughly in the order we think
they matter; nothing here is a promise of a date.

**1. An open-source PDK for the electronics.** The paper's examples use the SkyWater SKY130
models, which are third-party and whose public form does not parse in HSPICE. The **IHP
SG13G2 Open PDK** (130 nm SiGe BiCMOS, Apache-2.0) is a better fit, and we have checked it:
its transistor models run in both Xyce and HSPICE after a small, mechanical rewrite, and
Xyce returns analytic sensitivities for them that agree with finite differences to 2e-5 —
so `d(optical)/dW` on a real foundry transistor is within reach. Two honest limits. The
built-in engine cannot run them (they are PSP and VBIC compact models, thousands of lines
each); that route stays Xyce or HSPICE. And the open PDK ships **no photonic models** —
IHP's photonic processes are available only under NDA — so the optics would still come from
SPIPE's own device library.

**2. Gradients in envelope mode — done for the drive.** `mode='envelope'` lifts the
zero-optical-memory assumption (see [docs/scope.md](docs/scope.md)), which is exactly the
regime — resonators, delay-set oscillation — where you most want to optimise. It is now
differentiable with respect to the modulator drive, matching finite differences to 4e-09.
What remains is differentiating with respect to *passive* parameters (a waveguide length, a
coupler angle) in envelope mode, which needs a differentiable path through the passive
network's transfer function.

**3. A faster photonic solve — aimed at the right place.** The scattering-matrix system
`A x = b` is already assembled sparse and solved with a sparse LU once it is large, and the
adjoint reuses that factorisation, so it is close to the best available *solver*: a
matrix-free iterative solve was measured 100–600× slower, because a low-loss mesh's round
trip gain sits near 1. The time is spent elsewhere. Profiling shows 40–99 % of a forward
solve is the Python loop that builds each device, and 90–96 % of a backward pass is the
per-modulator Jacobian. Batching both is the real speed-up.

**4. GPU.** The photonic solve runs correctly on a GPU (`spipe.config['device'] =
torch.device('cuda')`; see [docs/performance.md](docs/performance.md)), and gradients now do too. But
measured end to end it is *slower* than the CPU (0.26–0.65×), because that same
one-device-at-a-time assembly loop launches many tiny GPU operations. Batching the assembly
(item 3) is what would let a GPU pay off; we will publish measurements on a modern GPU
rather than a prediction.

**5. Choosing your precision.** `spipe.config` has two dtypes: `real_dtype` for the device
parameters and `complex_dtype` for the solve.
- **Mixed precision is safe.** Setting only `complex_dtype = torch.complex64` halves the dense
  solve's memory. It stays within about 1e-5 of double precision, even on a resonant ring. The
  sparse solver used for large circuits works in double precision regardless.
- **A `float32` real dtype is not safe.** Every propagation phase `β·l` is then rounded to about
  6e-8 of itself. On a mesh of short waveguides that is harmless (1e-6). On a resonator it is
  not: a 1 cm ring's phase of 1e5 rad is rounded to 8e-3 rad, wider than its linewidth, and
  its spectrum came out 93 % wrong. SPIPE now warns when a `float32` run has a phase that large.

The plan is one `precision` setting instead of two independent dtypes. (`complex32` is not possible: PyTorch has no half-precision complex linear
algebra.)

Contributions and bug reports are welcome. If something in here is wrong, or a number does
not reproduce, please open an issue — `test/run_all.py` is the right thing to run first.

## Licence

MIT — see [LICENSE](LICENSE).
