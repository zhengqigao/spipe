# Optical memory: `mode='envelope'`

By default SPIPE solves the photonic network in **steady state** at every time sample, so light
takes no time to cross it. For most electronic–photonic links that is accurate: a 250 µm
waveguide is 3 ps, far below a nanosecond sample. It stops being accurate when the optics are
slow compared with the samples: a long delay line, or a resonator whose photon lifetime
spans many samples. SPIPE then warns (see [scope.md](scope.md)), and `mode='envelope'` is the
remedy.

```python
photocurrent, probes, power = Photonic(netlist).simulate(t, drive, mode='envelope')
_, _, photocurrent, drive, _ = Circuit('link.sp', 'native').simulate(mode='envelope')
```

The passive part of the circuit is linear and does not change in time. So its response over the
`.freq` band is computed once and turned into an impulse response. Light reaching a detector at
time `t` then carries the modulator state from `t − τ`, which is the physics the steady-state
solve leaves out. Once the drive stops changing, the result settles to exactly the steady-state
answer.

[`examples/envelope_delay_ring.py`](../examples/envelope_delay_ring.py) runs in a few seconds and
shows both cases:

```
delay line: quasistatic 50% at 210 ps, envelope at 310 ps (drive at 200 ps, delay 100 ps)
ring: build-up 5.2479x (analytic 5.2479x), lifetime fitted 163.0 ps, analytic 163.0 ps
```

## Sizing the `.freq` grid

The `.freq` grid is what the impulse response is built from, so it has to satisfy two conditions,
where `dt` is the transient time step:

- **Wide enough: the band must be several times `1/dt`.** Each carrier needs the circuit's
  response over the `1/dt` around it. A 10 ps step needs a band of several times 100 GHz.
- **Fine enough: `1/(2·df)` must be well above the circuit's memory.** For a delay line that
  memory is the delay. For a resonator it is several **photon lifetimes**, which can be far
  longer than the round trip. The example's ring has a 163 ps lifetime. At `df` = 0.25 GHz,
  which gives ±2 ns of memory, the fitted lifetime is exact. At 1 GHz (±500 ps) it comes out at
  17.8 ps.

SPIPE warns, with the numbers filled in, when either condition fails. A useful start is
`df ≤ 1/(10·dt)` together with a band of a few times `1/dt`.

## Reading the result: which carrier

The carriers within half a window of either end of the band do not have the full `1/dt` of
response around them. `config['envelope_edge']` decides what happens to them. The table is
measured on a lossless 100 ps delay line, with 24.7 % of the carriers at the edges:

| `envelope_edge` | what the edge carriers get | detected energy | light before it could arrive |
|---|---|---|---|
| `'zerofill'` (default) | the part of their window that exists | exact | up to 0.5 % |
| `'drop'` | nothing: removed from the detector sum | 75 % (only the full-window carriers) | none |
| `'quasistatic'` | no memory at all | exact | 24.7 %, instantly |

So a **photocurrent**, which sums every carrier, is slightly blurred in time at the default
setting. For a clean time response, choose one of these:
- widen the band, so fewer carriers sit at the edges;
- set `'drop'` and accept a lower current;
- or read the **field at one central carrier** from a `.prob` node, e.g.
  `probes['c1'][:, k, 1]` with `k` near the middle of the grid.

For a resonator, read the field at one carrier in any case. The detector adds up carriers at
every detuning from resonance, and that sum is not a single exponential.

## Gradients

Envelope mode is differentiable with respect to:
- **the modulator drive**, in a `Photonic`;
- **`.sensparam` electronic parameters**, through a `Circuit`. The optical delay inside the
  loop is part of the derivative. On a driver–modulator–100 ps delay line, it matches finite
  differences to 1.2e-8, and it differs from the quasistatic gradient, as it should.

It is not yet differentiable with respect to passive photonic parameters (`Photonic.param`); that
raises an error. Use the default mode for those.

## Settings

| `spipe.config[...]` | default | meaning |
|---|---|---|
| `'envelope_edge'` | `'zerofill'` | edge-carrier policy, see above |
| `'envelope_adiabatic_ratio'` | `1e-2` | below this relative change of the response across a carrier's window, the circuit has no memory worth keeping, and the run is handed to the steady-state solver: identical results, no cost |
| `'envelope_tap_tol'` | `1e-12` | impulse-response taps smaller than this fraction of the largest are dropped |
| `'envelope_max_bytes'` | 2 GiB | memory limit for the impulse responses; exceeding it is an error that says how much is needed |

## Limits

- **The time grid must be uniform.**
- **A modulator inside an optical loop can make the method unstable**, for example a ring with
  a modulator in it whose round trip is shorter than the time step. SPIPE raises an error on
  runaway growth, and warns when the result does not settle to the steady state. See
  [scope.md](scope.md).
