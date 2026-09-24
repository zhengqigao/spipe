# Performance, memory and the GPU

How SPIPE scales, where the time goes, and the settings that trade speed, memory and accuracy.
All numbers were measured on a shared 56-core server with 8 Tesla K80 GPUs, so the absolute times
are rough; the comparisons between rows are what carry over.

## How it scales

| case | forward | backward | peak memory |
|---|---|---|---|
| 64×64 passive mesh, 2211 devices | 1.6 s | — | 0.5 GB |
| same, gradient for all 2016 phase shifters (`Photonic.param`) | 1.8 s | 3.8 s | 0.5 GB |
| 64×64 mesh, 100 wavelengths | 8.3 s | — | 0.7 GB |
| 129 devices with 7 modulators, 2000 time samples | 3.1 s | — | 0.8 GB |
| same, drive gradient | 3.1 s | 7.3 s | 1.5 GB |
| 51 devices with 4 modulators, 10 000 time samples | 1.4 s | — | 0.75 GB |
| co-simulation `link_driver_mzm.sp`, 40 samples (built-in engine) | 8.9 s | — | 0.4 GB |
| same, with the gradient | 15.2 s | 4.1 s | 0.75 GB |

The main points:
- **Passive photonics scales well.** Thousands of devices, and one adjoint solve covers every
  parameter.
- **A modulated circuit costs one solve per time sample.** Its gradient is dominated by the
  modulator Jacobian, which takes 90–96 % of backward.
- **A co-simulation's cost is the electronic transient.** The built-in engine runs Newton
  device by device in Python. It is about 10× slower than Xyce (see [backends.md](backends.md)),
  and linear in the number of samples.

## Memory

Every (time sample, wavelength) pair is its own linear system. For small circuits SPIPE solves
them all at once as one dense batch, which is fast. It keeps that batch within
`config['photonic_dense_budget']` (256 MB):
- **Without a gradient,** it solves the batch in blocks of time samples. The results are
  identical, and the memory stays bounded however long the run.
- **With a gradient,** every factorisation has to be kept for backward. Over the budget, SPIPE
  switches to the sparse solver, whose factors are small.

Before this, a 129-device circuit over 2000 samples took 13 GB, and 5000 samples ran out of
memory.

Envelope mode has two limits of its own:
- `config['envelope_max_bytes']` (2 GB) for its filters;
- `config['envelope_grad_max_bytes']` (16 GB) for a gradient's autograd graph, which grows with
  time samples × carriers × modulator ports.

Over either limit is an error that says what to reduce. See [envelope.md](envelope.md).

## The GPU

```python
spipe.config['device'] = torch.device('cuda')
```

Everything runs there, and results agree with the CPU bit for bit in double precision. **It is
not faster yet.** The photonic assembly builds the circuit one device at a time, and that
launches many small GPU operations. On the circuits above the GPU was 0.3–0.7× the speed of the
CPU.

Large circuits (over 2048 unknowns, or over the memory budget) use the sparse solver, which is
SciPy on the CPU; SPIPE warns when that happens with a GPU device.
`config['photonic_solver'] = 'dense'` keeps the solve on the GPU if it fits in GPU memory.

## Precision

`config['complex_dtype'] = torch.complex64` halves the dense solve's memory and stays within
about 1e-5, even on a resonant ring. Leave `config['real_dtype']` at `torch.float64`. With
float32 there, long propagation phases lose their accuracy: a 1 cm ring came out 93 % wrong.
SPIPE warns when a float32 run has phases that large.

## Settings

| `spipe.config[...]` | default | what it trades |
|---|---|---|
| `'device'` | CPU | where tensors live; see above |
| `'real_dtype'`, `'complex_dtype'` | float64, complex128 | precision and memory; see above |
| `'photonic_solver'` | `'auto'` | `'dense'` / `'sparse'` to force a solver |
| `'photonic_dense_budget'` | 256 MB | when the dense batch is split or goes sparse |
| `'photonic_jac_chunk'` | automatic | time samples per modulator-Jacobian evaluation in backward |
| `'quasistatic_check'` | `True` | the optical-delay check. It is skipped when the path length alone already warns; otherwise, on a large mesh, it can cost a few simulations' worth of time |
| `'native_nsub'` | adaptive, at most 8 | internal steps per sample on the built-in engine: accuracy against time ([backends.md](backends.md)) |
| `'fd_step'` | 1e-3 | relative step of the HSPICE back end's finite-difference gradient |
| `'envelope_*'` | — | see [envelope.md](envelope.md) |
| `'max_iter'`, `'atol'`, `'rtol'`, `'seed'` | 100, 1e-3, 1e-3, 0 | the electronic–photonic fixed point, and its random start |
