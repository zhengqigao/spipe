# How SPIPE computes gradients

The headline capability is `d(optical output) / d(electronic device parameter)` — through
the driver, the modulator, the photodetector, and the feedback between them. This page
explains how, and why it does not cost what you might expect.

The netlist side of it — the `.sensparam` card — is defined in [netlist.md](netlist.md).

## Three pieces compose

**1. The photonics.** The optical network is a linear system `A x = b`, where `x` holds two
unknowns per port (one wave in each direction). Differentiating it gives

```
dx/dθ = −A⁻¹ (dA/dθ) x
```

`A` has already been factorised to get `x`, so a sensitivity costs one extra solve against
a factorisation you already own, not a new simulation. Measured against central finite
differences: **2.0e-10**.

**2. The electronics.** The built-in engine provides both PyTorch autograd and an explicit
time-domain adjoint. They agree to **exactly 0.00e+00**. The adjoint is a *single* backward
sweep over the transient regardless of how many parameters you declared, so five parameters
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

There is one `simulate()`. It decides for itself:

| situation | what happens |
|---|---|
| no `.sensparam` in the netlist | plain run, no graph |
| `.sensparam` declared but `requires_grad=False` | plain run, no graph |
| called inside `torch.no_grad()` | plain run, no graph |
| a declared parameter requires grad | **builds the graph**, about 1.65× the plain runtime |

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

Central finite differences over the **whole** chain, on a CMOS inverter driving a
Mach–Zehnder modulator:

| | through `Circuit.simulate()` | composing the two solvers by hand |
|---|---|---|
| analytic | `29.19573838` | `-115016.7648` |
| finite difference | `29.19573971` | `-115016.7645` |
| **relative error** | **`4.6e-08`** | **`1.9e-09`** |

A single finite-difference number can mislead, so the first column comes from a step-size
sweep. The error falls as `h²` while truncation dominates, bottoms out at `h/W = 1e-4`,
then *rises* as `1/h` once round-off in the transient solve takes over:

| step `h/W` | `1e-2` | `1e-3` | `1e-4` | `1e-5` | `1e-6` | `1e-7` |
|---|---|---|---|---|---|---|
| relative error | `1.9e-04` | `1.9e-06` | **`4.6e-08`** | `1.3e-07` | `6.4e-07` | `1.2e-05` |

That V-shape is the signature of an exact analytic gradient: the disagreement is the finite
difference's, not the adjoint's, and `4.6e-08` is simply as closely as a finite difference
through an adaptive-step transient can check it.

Both routes are checked in `test/tb/tb12_end_to_end_grad.py`. The unified call is what you
want in practice; composing `Netlist` and `Photonic` yourself is only useful when you need
to insert something between the two domains.

## Two limitations

- Use the **`native`** backend. Xyce's device derivatives are numerical and do not hold up
  on MOSFET widths; HSPICE re-runs finite differences. See [backends.md](backends.md).
- `mode='envelope'` (see [scope.md](scope.md)) does **not** currently support gradients.
  Use the default `mode='quasistatic'` for the adjoint.
