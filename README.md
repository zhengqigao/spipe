# SPIPE

**Differentiable SPICE-level co-simulation for integrated photonics and electronics.**

SPIPE simulates heterogeneous photonic–electronic circuits by keeping each domain in its
native formulation — SPICE for the electronics, the scattering-matrix method for the
photonics — and coupling them through the modulators and photodetectors that join them.
No Verilog-A conversion of photonic compact models, and the whole chain is differentiable,
so you can take the derivative of an *optical* output with respect to an *electronic*
device parameter.

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

Requires Python ≥ 3.9, PyTorch, NumPy and SciPy. `pip install -e ".[examples]"` adds
matplotlib and h5py for the example scripts.

That is everything you need: SPIPE ships its own analog simulator, so the package alone
runs every photonic circuit and every electronic–photonic co-simulation.

The `examples/paper/` scripts additionally reproduce the paper against external tools —
**Xyce** (free, open source) or **HSPICE** (commercial) for the electronics, and
**Lumerical INTERCONNECT** for the photonic cross-check. Put them on `PATH`, or point
`$SPIPE_XYCE` / `$SPIPE_HSPICE` at them.

## Quick start

A passive circuit needs nothing but the netlist:

```python
from spipe.photonic.photonic import Photonic

netlist = [l + "\n" for l in [
    ".mode neff=2.35 ng=4.0 wl=1550e-9",
    ".freq 193.1e12 193.1e12 1",          # 1 frequency channel
    ".source 1.0@a1 0.0@a2",
    "mzi0 a1 a2 b1 b2 theta=0.25pi",      # 50:50 coupler
    "pd1 b1 vo1 level1 r0=1.0",           # detector 1
    "pd2 b2 vo2 level1 r0=1.0",           # detector 2
    ".prob b1",                           # also report the field at node b1
]]
photocurrent, probes, power = Photonic(netlist).simulate()

photocurrent.shape        # (1, 2)     -> (n_time, n_detectors)
photocurrent              # tensor([[0.5, 0.5]])   half the light to each port
probes["b1"].shape        # (1, 1, 2)  -> (n_time, n_freq, 2)
```

An **active** device (`mzm`, `modm`, `modp`) is driven by an electrical signal, so it needs
a time axis and one drive column per modulator:

```python
import torch
from spipe.photonic.photonic import Photonic

netlist = [l + "\n" for l in [
    # --- optical mode: effective index, group index, reference wavelength ---
    ".mode neff=2.35 ng=4.0 wl=1550e-9",

    # --- which optical frequencies to solve at: start, stop, number of points.
    #     One point here, so a single CW carrier at 193.1 THz. Several points
    #     means several independent WDM channels (see "What comes back").
    ".freq 193.1e12 193.1e12 1",

    # --- laser input: complex amplitude @ node. Light enters at a1, nothing at a2.
    ".source 1.0@a1 0.0@a2",

    # --- the modulator ------------------------------------------------------
    #   mzm0           instance name; the `mzm` prefix picks the model
    #   a1 a2          two optical inputs  (left ports)
    #   b1 b2          two optical outputs (right ports)
    #   vdrv           ELECTRICAL node carrying the drive -- this is the
    #                  electronic-photonic interface
    #   level3         equivalent electrical load seen by the driver
    #                  (realistic depletion-mode RC; `level1` is near open-circuit)
    #   vpi=2.0        voltage for a full on->off swing
    #   vbias=0.0      operating point
    #   il=0.0         insertion loss in dB (0 = lossless)
    "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",

    # --- detectors: optical node, electrical node, model, responsivity -------
    "pd1 b1 vo1 level1 r0=1.0",
    "pd2 b2 vo2 level1 r0=1.0",
]]

# Time axis and drive. `drive` has one COLUMN PER MODULATOR, in the order the
# modulators appear in the netlist -- here just mzm0, so a single column.
t     = torch.linspace(0, 1e-8, 5)                 # (n_time,)            seconds
drive = torch.linspace(0, 2.0, 5).reshape(-1, 1)   # (n_time, n_mod)      volts, 0 -> V_pi

photocurrent, probes, power = Photonic(netlist).simulate(t, drive)

photocurrent[:, 0]   # detector 1, one value per time point:
                     # tensor([0.0000, 0.1464, 0.5000, 0.8536, 1.0000])
                     # full-off at 0 V -> full-on at V_pi = 2 V, following
                     # sin^2(pi*v / 2*vpi) exactly. V_pi really is V_pi.
```

> The drive would normally come from a *circuit* rather than a hand-written ramp. To couple
> one in, write a single file with a `.electronic` section and a `.photonic` section and
> hand it to `spipe.Circuit` — SPIPE then solves the electronics (with its own engine, Xyce
> or HSPICE), feeds the modulator voltages into the photonic solve, feeds the photocurrents
> back, and iterates to a self-consistent solution. See `examples/`.

### What comes back

| returned | type / shape | meaning |
|---|---|---|
| `photocurrent` | `float64`, **`(n_time, n_detectors)`** | detector current, one column per `pd` line **in netlist order**. Each entry is `Σ_ω R(ω)·\|E(ω,t)\|²` — already **summed over the frequency channels**, so with `N` channels each carrying unit power a fully-transmitting path reads `N`, not 1. |
| `probes` | `dict[str, complex128]`, each **`(n_time, n_freq, 2)`** | the complex field at every node named on a `.prob` line. The last axis is `[inward, outward]`: index `0` is the wave travelling *into* the first device attached to that node, index `1` the wave travelling *out*. Magnitude and phase are both physical. |
| `power` | `dict[str, tensor]`, each **`(n_time,)`** | power budget in **watts** — see below. |

`n_time` is `len(t)` (or `1` for a purely passive circuit), `n_freq` is the number of
points on the `.freq` line, and `n_detectors` is the number of `pd` lines.

The `power` dict is only populated when the laser is characterised with `power=` (electrical
watts drawn) and `eff=` (wall-plug efficiency) on the `.source` line; otherwise it is
`{"photonic": None}`.

```python
".source 1.0@a1 0.0@a2 power=0.1 eff=0.2"
```

| key | meaning |
|---|---|
| `laser_drive_power` | electrical power drawn by the laser, i.e. `power × Σ\|A\|²` |
| `optical_input_power` | optical power launched into the circuit, `laser_drive_power × eff` |
| `optical_output_power` | optical power reaching the detectors |
| `optical_loss` | `optical_input_power − optical_output_power` |
| `photonic` | alias of `laser_drive_power`, kept for backward compatibility |

Output and loss always sum exactly to the input, and the reported consumption does not
depend on how you normalise the source amplitudes.

For a full electronic–photonic co-simulation, write one text file with a `.electronic`
and a `.photonic` section and hand it to `spipe.Circuit`; see `examples/`.

## End-to-end differentiability

This is what SPIPE is for. You can differentiate an **optical** quantity with respect to an
**electronic** device parameter — across the modulator, which is normally where the
gradient chain breaks:

```
   transistor W  ──►  driver  ──►  modulator  ──►  photodetector  ──►  |E_out|²
   └──────────────────────  d|E_out|² / dW  ◄───────────────────────────┘
```

Nothing special is required: SPIPE's built-in engine is written in PyTorch, so a device
parameter is a leaf tensor and `.backward()` simply works.

```python
import torch
from spipe.electronic.native import Netlist
from spipe.photonic.photonic import Photonic

ckt = Netlist.from_string(driver_netlist)     # a CMOS driver into the modulator load
w   = ckt.param("MN1", "W")                   # the pull-down width: a float64 leaf tensor
w.requires_grad_(True)

res   = ckt.tran(2e-11, 3e-8)                 # electrical transient
optic = Photonic(photonic_netlist, need_grads=True)
out, _, _ = optic.simulate(res.t, res.v("mod").reshape(-1, 1))

(out[:, 0] ** 2).sum().backward()
w.grad        # d|E_out|² / dW  -- exact, one backward pass
```

Measured against central finite differences over the **whole** chain:

| | |
|---|---|
| analytic | `-115016.7648` |
| finite difference | `-115016.7645` |
| **relative error** | **`1.9e-09`** |

Verified on two independent circuits (`test/tb/tb12_end_to_end_grad.py`).

**How it works, and why it is cheap.** Three pieces compose:

1. **Photonics** — differentiating `A x = b` gives `dx/dθ = −A⁻¹ (dA/dθ) x`, so the
   sensitivity costs one extra solve with a factorisation you already have.
2. **Electronics** — the built-in engine provides both autograd and an explicit
   time-domain adjoint. They agree to `0.00e+00`, and the adjoint is a *single* backward
   sweep: five parameters cost **1.03×** the time of one, not 5×.
3. **The coupling** — when photocurrent feeds back into modulator drive, the fixed point is
   differentiated with the **implicit function theorem**, not by unrolling the iteration.
   One small linear solve, independent of how many iterations ran, and it reports if the
   coupling Jacobian is ill-conditioned rather than returning a silent wrong number.

`Circuit.differentiable_simulate()` wires all three together for a co-simulation; declare
which electronic parameters are differentiable with `.sensparam` in the netlist:

```
.sensparam MN1:W MN1:L RL:R
```

**Use the `native` backend for this.** Xyce's device derivatives are numerical and reach
about 1e-2 on linear devices but not on MOSFET widths; HSPICE re-runs finite differences
and reaches about 5e-2. Only the built-in engine gives exact gradients.

Note that `mode='envelope'` (see *Scope*) does **not** currently support gradients — use
the default `mode='quasistatic'` for the adjoint.

## Layout

| path | contents |
|---|---|
| `src/spipe/` | the package |
| `src/spipe/photonic/` | scattering-matrix solver, device models, envelope propagation |
| `src/spipe/electronic/` | HSPICE / Xyce drivers, sensitivity plumbing |
| `src/spipe/electronic/native/` | **built-in** differentiable analog simulator (pure Python/PyTorch) |
| `src/spipe/core/` | the electronic–photonic fixed point |
| `test/` | **automated regression suite** — 306 machine-checked assertions. Run it to verify a new version of SPIPE. |
| `examples/` | **annotated demonstrations** — commented scripts showing how to drive SPIPE, plus the paper's figures |
| `docs/` | physics, scope and architecture notes |
| `scripts/` | helper scripts (e.g. fetching the SKY130 models) |

## Three electronic backends

| backend | availability | `d(optical)/d(device parameter)` |
|---|---|---|
| `native` | **ships with SPIPE** — pure Python/PyTorch, nothing to install | **exact** — autograd and a time-domain adjoint agree to `0.00e+00` |
| `xyce` | **free and open source** ([Xyce](https://xyce.sandia.gov/), [GitHub](https://github.com/Xyce/Xyce), GPL-3.0, Sandia National Laboratories) | numerical, via `.SENS` — use `direct`, not `adjoint` (see below) |
| `hspice` | commercial (Synopsys) | finite differences over re-runs |

So **two of the three backends are freely available**: the built-in engine needs nothing
beyond this package, and Xyce is open source.

The built-in engine covers R, L, C, V, I, controlled sources, diodes, BJTs, MOSFET level-1
and switches, with DC and transient analysis and adaptive step control. It matches HSPICE
and Xyce on the circuits in `test/tb/tb05_native_crosstool.py` and reproduces closed-form
solutions to machine precision (see `docs/validation.md`).

> **A trap worth knowing about Xyce.** Its *transient adjoint* silently returns **all
> zeros** for device parameters — measured: `direct=1` gives `d V(d)/d M1:W = -4.997e+04`,
> matching finite differences to 0.075 %, while `adjoint=1` returns `0.0` with no error.
> SPIPE therefore forces the direct method for device parameters and **raises** if an
> all-zero sensitivity block comes back with a non-constant objective, rather than handing
> you a plausible-looking zero gradient.

## Verifying a build

`test/` exists to answer one question automatically: **is this version of SPIPE correct?**
Every check compares against a known-true answer — a closed-form solution, a conservation
law, or an analytically known derivative — and either passes or fails. Run the whole thing
with one command and read the exit status:

```bash
python test/run_all.py            # full suite, single PASS/FAIL verdict
python test/run_all.py --quick    # skip the slow benches (~20 s, 203 checks)
python test/run_all.py --list     # what each bench guards
```

```
  tb01_photonic_algebra      ok     42 pass   0 fail   photonic invariants
  tb04_native_analytic       ok     35 pass   0 fail   built-in engine vs closed form
  ...
  PASS: 203 passing, 0 failing, 1 skipped (20s)
```

Exit status is 0 only if every check passed, so it drops straight into CI or a pre-commit
hook. This is the regression gate to run before trusting any change to the solver.

Most benches need nothing but PyTorch. `tb05_native_crosstool` compares the built-in engine
against **Xyce** (free, open source) and HSPICE, and is skipped automatically when neither
is on `PATH`. To include it:

```bash
export SPIPE_CAD_SETUP='module load xyce hspice'   # whatever puts them on PATH
python test/run_all.py
```

Ground truth is mathematics wherever possible rather than another simulator, so a failure
means the code is wrong — not that a reference trace drifted. See `docs/validation.md`.

## Examples

`examples/` is documentation you can run. Each script is heavily commented and
prints an engineering report explaining what it demonstrates — start here to learn
how to drive SPIPE. Unlike `test/`, these do not assert: they show you numbers and
leave the judgement to you.

```
examples/
├── oeo_electronic.py       optoelectronic oscillator (electronic-delay loop)
├── bistable_latch.py       optical bistability — two stable states
├── bias_control.py         automatic MZM bias control at quadrature
├── oeo_optical_delay.py    a case SPIPE deliberately REFUSES to simulate (see below)
├── derived/                Level-1 circuits that run on the built-in engine
└── paper/                  the figures from the TCAD paper
    ├── mesh_lumerical/     programmable photonic mesh vs Lumerical INTERCONNECT
    └── ptc_hspice/         photonic tensor core driven by an 8-bit SKY130 DAC
```

The `paper/ptc_hspice` examples need the SKY130 device models, which are third-party and
not vendored here:

```bash
./scripts/fetch_sky130.sh
```

That sparse-checks out only the two cell libraries used, from
[google/skywater-pdk-libs-sky130_fd_pr](https://github.com/google/skywater-pdk-libs-sky130_fd_pr)
(Apache-2.0, © The SkyWater PDK Authors).

## Scope — what SPIPE does and does not model

SPIPE solves the photonic network in **steady state at each time sample**. That is exact
when every optical transit time is short compared with the modulation timescale, and it is
what makes the method fast. Outside that regime the limitation is *structural*, not
gradual: with no optical memory there is no delay, so resonator ring-up and delay-set
oscillation are **absent** rather than approximated.

Two things follow, and SPIPE is explicit about both:

- A **quasi-static guard** warns, with numbers, when the estimated maximum group delay
  approaches the time step.
- **`mode='envelope'`** lifts the restriction by giving the passive network a real impulse
  response. It reproduces a ring resonator's photon lifetime and reduces exactly to the
  default in the adiabatic limit. It does not currently support gradients — use
  `mode='quasistatic'` for the adjoint.

`examples/oeo_optical_delay.py` is a deliberate demonstration of the boundary: it refuses
to simulate an oscillator whose frequency is set by an optical delay, and explains why,
rather than returning a plausible-looking number.

See `docs/scope.md` for the full discussion.

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

## Licence

MIT — see [LICENSE](LICENSE). The SKY130 models fetched by `scripts/fetch_sky130.sh` are
Apache-2.0 and carry their own licence.
