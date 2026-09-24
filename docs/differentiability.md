# How SPIPE computes gradients

The headline capability is `d(optical output) / d(electronic device parameter)` — through
the driver, the modulator, the photodetector, and the feedback between them. This page
explains how, and why it does not cost what you might expect.

The netlist side of it — the `.sensparam` card — is defined in [netlist.md](netlist.md).

## How the gradient is computed

**1. The photonics.** The optical network is a linear system `A x = b`, where `x` holds two
unknowns per port (one wave in each direction). Differentiating it gives

```
dx/dθ = −A⁻¹ (dA/dθ) x
```

`A` has already been factorised to get `x`, so a sensitivity costs one extra solve against
a factorisation you already own, not a new simulation. Measured against central finite
differences: **2.0e-10**.

**2. The electronics.** The built-in engine provides both PyTorch autograd and an explicit
time-domain adjoint. The autograd path is built on that adjoint, so the two agree to
**exactly 0.00e+00**. That is a consistency property of the implementation, not an independent
check; the independent check is against finite differences (7.7e-11). The adjoint is a
*single* backward sweep over the transient regardless of how many parameters you declared, so five parameters
cost **1.03×** the time of one — not 5×.

**3. The coupling.** When photocurrent feeds back into a modulator drive, the two domains
are solved as a fixed point. Its gradient is taken with the **implicit function theorem**,
not by back-propagating through the iteration.

## Why the implicit function theorem matters

Unrolling the fixed-point loop would work, and it is what a naive implementation does. It
has two problems:

- Memory grows with the iteration count, because every intermediate state stays alive for
  the backward pass.
- Worse, the answer depends on *where the iteration happened to stop*. Two runs that
  converged to the same waveform to 1e-12 can report visibly different gradients.

At a converged fixed point `V* = g(f(V*))`, the implicit function theorem gives the exact
gradient of the converged solution from one linear solve:

```
(I − Aᵀ) λ = ∂J/∂V ,   A = d(g∘f)/dV
```

whose size is the number of coupling unknowns and whose cost is **independent of how many
iterations ran**. `test/tb/tb12_end_to_end_grad.py` checks this against the closed form,
and checks that the dense and matrix-free paths give the same answer — which is the
observable signature that no loop was unrolled.

Three paths, in order of cost:

| path | when | cost |
|---|---|---|
| decoupled | no photocurrent reaches any modulator (every example SPIPE ships) | one extra backward pass; `(I − Aᵀ)⁻¹ b = b` *exactly*, an identity rather than an approximation |
| dense | a small coupling system | `n` products build `Aᵀ`, then one solve; also reports the condition number and the loop gain |
| matrix free | a large one | a Neumann iteration, converging exactly when the fixed point did |

If the coupling Jacobian is ill-conditioned — a feedback loop with gain at or beyond 1,
where the fixed point is only marginally stable and the derivative is genuinely unbounded —
SPIPE **raises** rather than returning a large, meaningless number.

## Turning gradients on and off

There is one `simulate()`, and `.sensparam` is the switch. Gradients are not free. On the
README example, recording the graph makes the run 1.7× slower, and `backward()` adds about a
quarter of that again: 8.9 s plain, against 15.2 s + 4.1 s with the gradient, 2.2× in all
(measured on a shared server; the ratio is what carries over). So `simulate()` builds the graph
only when something needs it:

| situation | what happens |
|---|---|
| no `.sensparam` in the netlist | plain run, no graph |
| `.sensparam` declared (the parameter already requires grad) | **builds the graph** |
| you switched off **every** declared parameter with `ckt.param(...).requires_grad_(False)` — optional | plain run, no graph |
| called inside `torch.no_grad()` | plain run, no graph |

The returned values are the same either way — measured to agree to `5.6e-16` on
`examples/link_driver_mzm.sp`, i.e. machine precision. The differentiable path solves the
fixed point gradient-free and then rebuilds one round trip at the converged point, so it
returns `g(f(V*))` where the plain path returns `V*`; at a converged fixed point those are
the same number.

(Earlier releases showed a `9.3e-08` gap here that no convergence tolerance could close. It
was not residual: the fixed-point iterate was being created in float32 regardless of
`config['real_dtype']`, and the gap was float32 rounding of the modulator drive. That is
fixed, and `test/tb/tb12_end_to_end_grad.py` guards it.)

`differentiable_simulate()` still exists if you want to force the issue, but you should not
normally need it.

## Measured accuracy

The gradient is checked against **numerical differentiation**: a central finite difference
that re-runs the whole co-simulation with `W` nudged up and then down,
`(L(W+h) − L(W−h)) / 2h`. It uses `examples/link_driver_mzm.sp` — a CMOS inverter driving a
Mach–Zehnder modulator — with the loss `L = Σ_t photocurrent₁(t)²`:

| | `dL/dW` |
|---|---|
| analytic, from `Circuit.simulate()` | `29.19573838` |
| finite difference, best step | `29.19573971` |
| **relative error** | **`4.6e-08`** |

A single finite-difference number can mislead, because the step `h` trades two errors
against each other, so the value above comes from a sweep over `h`:

| step `h/W` | `1e-2` | `1e-3` | `1e-4` | `1e-5` | `1e-6` | `1e-7` |
|---|---|---|---|---|---|---|
| relative error | `1.9e-04` | `1.9e-06` | **`4.6e-08`** | `1.3e-07` | `6.4e-07` | `1.2e-05` |

With a large step the finite difference is itself inaccurate (its error falls as `h²`).
With a small step it drowns in round-off from the transient solve (its error grows as
`1/h`). The analytic gradient does not change with `h` at all. That V-shape is the signature
of an exact analytic gradient: the remaining disagreement belongs to the finite difference,
and `4.6e-08` is as closely as a finite difference through an adaptive-step transient can
check it.

`test/tb/tb12_end_to_end_grad.py` repeats this comparison on every run.

What the comparison proves is that the gradient is exact **for the discretisation that ran**. How
close that discretisation is to the continuous circuit is a separate question. It is the
question of [backends.md](backends.md)'s time-step section, and this example is a demanding
case for it:
- Its 40 samples land on the rails, not on the edges, so `L` depends on `W` only through
  sub-millivolt tails a few samples after each edge.
- The default of at most 8 internal steps per sample resolves those tails to about 14%: at
  `native_nsub = 64` the gradient is `25.60`, and at 128 it is `25.59`.
- HSPICE and Xyce disagree with each other on the same sub-millivolt effect, because it sits
  below their default accuracy.

Sample the same circuit every 0.1 ns (`.tran 0 4e-8 401`) and the loss depends on the edges
themselves. Then the default and converged discretisations agree to 3e-4 (−1.69955e5 against
−1.69901e5), and Xyce (−1.65e5) and HSPICE with `.option delmax=10p` (−1.74e5) land within a
few percent.

## Training photonic parameters

In a photonic circuit on its own (`Photonic`), two kinds of quantity are differentiable:
- **the modulator drive**: make the drive tensor require grad;
- **any passive device parameter**, such as a phase shift, a coupler angle, a length or a loss:
  `Photonic.param` turns the number on the netlist line into a trainable tensor.

This is how you train a programmable mesh:

```python
ph = Photonic(netlist)
theta = ph.param('pbum0', 'theta')          # float64 leaf, requires_grad=True
phi = ph.param('pbum0', 'phi')
opt = torch.optim.Adam([theta, phi], lr=0.05)
for step in range(200):
    opt.zero_grad()
    photocurrent, probes, _ = ph.simulate()
    loss = ((photocurrent - target) ** 2).sum()
    loss.backward()                         # one adjoint solve for every parameter
    opt.step()
```

Every output is differentiable, including the complex fields that `.prob` returns. A fidelity
loss on the field, which sees phase, therefore works as well as one on detected power. The
gradients match finite differences to about 1e-9, for `pbum`, `ps`, `mzi`, `wg` lengths and
losses, in circuits with loops.

To set a parameter by hand, use `theta.data.fill_(0.7)` with a Python float. Copying from a
default `torch.tensor(0.7)` passes through float32 and loses about 1e-8.

Two things are not supported yet:
- `Photonic.param` in `mode='envelope'`, which raises an error (the drive *is* differentiable
  there);
- a `Photonic.param` inside a `Circuit`, which also raises an error. The derivative through
  the electronic–photonic loop is not implemented for photonic parameters; `.sensparam`
  electronic parameters are the ones that work there.

## Two limitations

- Use the **`native`** backend. Xyce's device derivatives are numerical and do not hold up
  on MOSFET widths; HSPICE re-runs finite differences. See [backends.md](backends.md).
- `mode='envelope'` (see [scope.md](scope.md)) is differentiable with respect to the
  **modulator drive** — matching finite differences to `4e-09` — but not yet with respect to
  passive device parameters such as a waveguide length. The default `mode='quasistatic'`
  is (see *Training photonic parameters* above).
