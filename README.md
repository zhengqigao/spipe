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

The examples that drive a commercial simulator additionally need **HSPICE** or **Xyce** on
`PATH` (or pointed at by `$SPIPE_HSPICE` / `$SPIPE_XYCE`); the Lumerical comparison needs
**INTERCONNECT**. The built-in engine needs none of them.

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

An **active** device (`mzm`, `modm`, `modp`) needs a drive waveform, so pass a time axis
and one column per modulator:

```python
import torch
from spipe.photonic.photonic import Photonic

netlist = [l + "\n" for l in [
    ".mode neff=2.35 ng=4.0 wl=1550e-9",
    ".freq 193.1e12 193.1e12 1",
    ".source 1.0@a1 0.0@a2",
    "mzm0 a1 a2 b1 b2 vdrv level3 vpi=2.0 vbias=0.0 il=0.0",
    "pd1 b1 vo1 level1 r0=1.0",
    "pd2 b2 vo2 level1 r0=1.0",
]]
t     = torch.linspace(0, 1e-8, 5)                 # (n_time,)
drive = torch.linspace(0, 2.0, 5).reshape(-1, 1)   # (n_time, n_modulators): 0 -> V_pi

photocurrent, probes, power = Photonic(netlist).simulate(t, drive)

photocurrent[:, 0]   # tensor([0.0000, 0.1464, 0.5000, 0.8536, 1.0000])
                     # full-off -> full-on across exactly one V_pi
```

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

## Layout

| path | contents |
|---|---|
| `src/spipe/` | the package |
| `src/spipe/photonic/` | scattering-matrix solver, device models, envelope propagation |
| `src/spipe/electronic/` | HSPICE / Xyce drivers, sensitivity plumbing |
| `src/spipe/electronic/native/` | **built-in** differentiable analog simulator (no licence needed) |
| `src/spipe/core/` | the electronic–photonic fixed point |
| `test/` | acceptance suite — 306 checks against closed-form physics |
| `examples/` | runnable demonstrations, including the paper's figures |
| `docs/` | physics, scope and architecture notes |
| `scripts/` | helper scripts (e.g. fetching the SKY130 models) |

## Three electronic backends

| backend | needs a licence | gradients |
|---|---|---|
| `native` | no | **exact** — autograd and a time-domain adjoint, agreeing to machine precision |
| `xyce` | yes | device parameters via `.SENS` (numerical); use `direct`, not `adjoint` |
| `hspice` | yes | finite differences over re-runs |

The built-in engine covers R, L, C, V, I, controlled sources, diodes, BJTs, MOSFET level-1
and switches, with DC and transient analysis and adaptive step control. It matches HSPICE
and Xyce on the circuits in `test/tb/tb05_native_crosstool.py` and reproduces closed-form
solutions to machine precision (see `docs/validation.md`).

## Running the tests

```bash
python -m pip install -e .
for t in test/tb/tb*.py; do python "$t"; done
```

Most of the suite needs nothing but PyTorch. `tb05_native_crosstool.py` compares against
HSPICE and Xyce; point `$SPIPE_CAD_SETUP` at whatever shell snippet puts them on `PATH`:

```bash
export SPIPE_CAD_SETUP='module load hspice xyce'
python test/tb/tb05_native_crosstool.py
```

## Examples

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
