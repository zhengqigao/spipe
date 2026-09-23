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
| `probes` | `{node: (n_time, n_freq, 2)}` | the complex field at nodes you asked about with a `.prob` line. The last axis is `[inward, outward]`. Empty unless you add `.prob`. |
| `power` | `{name: (n_time,)}` | the laser power budget, in watts. Only filled in if you characterise the laser with `power=` and `eff=` on the `.source` line. |

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

Every directive and device is defined in **[docs/netlist.md](docs/netlist.md)**.

## Two things you can simulate

| you have | object | returns |
|---|---|---|
| a photonic netlist only | `Photonic(netlist)` | **3** — `photocurrent, probes, power` |
| one file with an `.electronic` **and** a `.photonic` section | `Circuit(path, spice_exe=...)` | **5** — `probes_e, probes_p, photocurrent, drive, power` |

`Photonic` solves the optical network and you supply the modulator drive yourself, as
above. `Circuit` additionally runs the electronics, feeds the photocurrents back, and
iterates until the two domains agree — so it also returns the electrical probes and the
converged drive. Both are called `simulate()`.

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
gradient is right to **8 significant figures**.

There is only **one** simulation call. `simulate()` follows the usual PyTorch convention:
gradients cost nothing unless you ask for them, and you ask for them by declaring a
parameter and leaving `requires_grad` on. The returned values are the same either way.

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

**345 checks, and the exit status is 0 only if every one passed**, so it drops straight
into CI. `--quick` runs the fast ~230 of them in under a minute. Almost all of it needs
nothing but PyTorch: the handful of checks that call HSPICE, Xyce or Lumerical INTERCONNECT
**skip themselves** when that tool is not installed, rather than failing — so on a bare
machine you will see a slightly smaller total and a few skips. A few highlights:

| | measured |
|---|---|
| energy conservation on a lossless circuit **with an optical loop** | error **exactly 0** |
| a photonic mesh vs **Lumerical INTERCONNECT** | agrees to `5e-7`, the limit of INTERCONNECT's own output precision |
| the built-in circuit engine vs closed-form solutions | to machine precision |
| the built-in engine vs **HSPICE and Xyce** on identical netlists | within how much those two disagree with each other |
| `d|E|²/dW` vs finite differences over the whole chain | `1.9e-09` |

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
├── oeo_optical_delay.py    a case SPIPE deliberately REFUSES to simulate
├── link_driver_mzm.sp      the two-domain netlist used above
├── derived/                circuits that run on the built-in engine, no PDK needed
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
real impulse response; and `examples/oeo_optical_delay.py` is an oscillator SPIPE *refuses*
to simulate, printing why instead of returning a plausible wrong number.

The full discussion, including two further assumptions worth knowing about:
[docs/scope.md](docs/scope.md).

## Documentation

| | |
|---|---|
| [docs/netlist.md](docs/netlist.md) | every directive and device, defined |
| [docs/differentiability.md](docs/differentiability.md) | how the gradients are computed |
| [docs/backends.md](docs/backends.md) | choosing an electronic engine, and their traps |
| [docs/scope.md](docs/scope.md) | the physics assumptions and where they bind |
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

**1. An open-source PDK, end to end.** SPIPE's paper examples use the SkyWater SKY130
models, which are third-party and have to be fetched separately, and whose public form does
not parse in HSPICE. We are evaluating the **IHP SG13G2 Open PDK** — a genuinely open
130 nm BiCMOS process with an electronic–photonic offering — so that a new user can
reproduce a full electronic–photonic result with nothing but `pip install`. The open
questions are which simulator dialects its models ship in, and whether the built-in engine
(which today knows R, L, C, sources, diodes, BJTs and level-1 MOSFETs) can consume them at
all, or whether that route needs Xyce.

**2. Gradients in envelope mode.** `mode='envelope'` is what lifts the zero-optical-memory
assumption (see [docs/scope.md](docs/scope.md)), and it is exactly the regime — high-Q
resonators, delay-set oscillation — where you would most want to optimise a design. It does
not currently support the adjoint, so today you have to choose between optical memory and
gradients. Removing that choice is the single most valuable thing on this list.

**3. A sparser, cheaper photonic solve.** The scattering-matrix system `A x = b` is
assembled explicitly. `A` is very sparse — each device only couples its own ports — so most
of what is stored and factorised is zeros. Worth evaluating: sparse assembly with a sparse
direct solve, and a matrix-free iterative solve that never forms `A` at all. The constraint
is that the adjoint currently reuses the forward factorisation, so anything matrix-free has
to pay for its gradients differently. Whether that trade is worth it is an empirical
question we have not yet answered.

**4. GPU, honestly measured.** SPIPE is written in PyTorch partly so it can run on a GPU,
but we have no benchmark showing that it actually helps, and a circuit that is many small
per-frequency solves may well be slower on a GPU than on a CPU. The work is to measure it
properly across problem sizes, fix whatever device handling is wrong, and then say plainly
where the crossover is — including saying "use the CPU" if that is the answer.

**5. Choosing your precision.** `spipe.config` already carries `real_dtype` and
`complex_dtype`, and the suite runs in `complex128`. Making `complex64` a properly supported,
tested option would roughly halve memory and help on GPUs, at an accuracy cost that should
be stated rather than discovered. (A caution from experience: a mismatch between the two
dtypes once silently destroyed precision while everything appeared to work, so this needs
real tests, not a config flag.)

Contributions and bug reports are welcome. If something in here is wrong, or a number does
not reproduce, please open an issue — `test/run_all.py` is the right thing to run first.

## Licence

MIT — see [LICENSE](LICENSE).
