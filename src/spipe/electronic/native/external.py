"""External block interface -- the extensibility hook.

An *external block* injects its own equations into the very same MNA system
that the electrical devices are assembled into, so that an electronic +
photonic system is solved as a single DAE by a single Newton loop.  This is
the mechanism a later phase uses to add photonic delay-line state equations.

The contract
------------

A block is any object implementing::

    class MyBlock:
        name = "pd1"                # optional, used in error messages
        nodes = ["a", "b"]          # circuit node names this block couples to
        n_states = 2                # extra unknowns the block owns (may be 0)

        def init_state(self):
            "Optional. (n_states,) float64 tensor of initial state values."

        def parameters(self):
            "Optional. {name: float64 leaf tensor} exposed for gradients."

        def residual_and_jacobian(self, x, xdot, t):
            '''Return (r, J_x, J_xdot).

            x, xdot : (k,) float64 tensors, k = len(nodes) + n_states,
                      ordered [coupled node voltages ..., own states ...].
            t       : float, current time (0.0 during a DC solve, where
                      xdot is the zero vector).

            r     : (k,) residual contribution, added to the system rows that
                    correspond to the same k indices.  Entries for the coupled
                    nodes are *currents flowing out of the node into the
                    block* (standard KCL sign, exactly like a device stamp);
                    entries for the block's own states are its state
                    equations.
            J_x    : (k, k) tensor, dr/dx.
            J_xdot : (k, k) tensor, dr/dxdot.
            '''

The block's rows and columns are the *same* index set, so the contribution is
square and lands on the diagonal block of the MNA matrix plus the coupling
entries -- exactly what a two-terminal device stamp does.

The solver treats ``xdot`` with the same integration formula it uses for
device charges, i.e. at step *n*

    xdot_n = alpha_n * (x_n - x_{n-1}) + beta_n * xdot_{n-1}

with ``(alpha, beta) = (1/h, 0)`` for backward Euler and ``(2/h, -1)`` for
trapezoidal.  The Newton Jacobian therefore uses ``J_x + alpha * J_xdot``.

Because ``r`` is built from ordinary torch operations on the block's
parameter tensors, the time-domain adjoint picks up ``dr/dp`` automatically:
blocks are differentiable with no extra work.
"""

from __future__ import annotations

from typing import Dict, List

import torch

DTYPE = torch.float64

__all__ = ["ExternalBlock", "BehaviouralResistor", "RCStateBlock"]


class ExternalBlock:
    """Convenience base class; see the module docstring for the contract."""

    name = "block"
    nodes: List[str] = []
    n_states = 0

    def init_state(self):
        """Initial values for this block's own states (DC start)."""
        return torch.zeros(self.n_states, dtype=DTYPE)

    def parameters(self) -> Dict[str, torch.Tensor]:
        """Differentiable leaf tensors owned by the block."""
        return {}

    def residual_and_jacobian(self, x, xdot, t):  # pragma: no cover - abstract
        """Return ``(r, J_x, J_xdot)`` -- see the module docstring."""
        raise NotImplementedError


class BehaviouralResistor(ExternalBlock):
    """Trivial example block: a resistor whose conductance is a leaf tensor.

    It owns no state and couples to two circuit nodes, so its residual is the
    familiar ``[+i, -i]`` KCL pair.  Useful as a template and as a
    self-consistency check: a circuit with ``BehaviouralResistor(a, b, R)``
    must give exactly the same answer as the same circuit with ``Rx a b R``.

    Example
    -------
    >>> from spipe.electronic.native import Netlist
    >>> from spipe.electronic.native.external import BehaviouralResistor
    >>> ckt = Netlist.from_string("V1 in 0 DC 1\\nR1 in out 1k\\n")
    >>> blk = BehaviouralResistor('out', '0', 1e3)
    >>> ckt.add_external_block(blk)
    >>> float(ckt.op()['out'])
    0.5
    """

    n_states = 0

    def __init__(self, node_p, node_n, resistance, name="bres"):
        self.name = name
        self.nodes = [str(node_p), str(node_n)]
        self.R = torch.tensor(float(resistance), dtype=DTYPE)

    def parameters(self):
        """The resistance, exposed as a differentiable leaf."""
        return {"R": self.R}

    def residual_and_jacobian(self, x, xdot, t):
        """KCL pair ``[+i, -i]`` with ``i = (v_p - v_n)/R``."""
        g = 1.0 / self.R
        i = g * (x[0] - x[1])
        r = torch.stack([i, -i])
        jx = torch.stack([
            torch.stack([g, -g]),
            torch.stack([-g, g]),
        ])
        jd = torch.zeros(2, 2, dtype=DTYPE)
        return r, jx, jd


class RCStateBlock(ExternalBlock):
    """Example block *with* internal state, to exercise the ``xdot`` path.

    Models a grounded capacitor ``C`` on ``node`` using an explicit internal
    state ``q`` (the charge) instead of a device charge:

        KCL(node) :  + dq/dt
        state eq  :  q - C * v(node) = 0

    Electrically identical to ``C1 node 0 C`` -- which makes it a convenient
    correctness check for the external-block integration path.
    """

    n_states = 1

    def __init__(self, node, capacitance, name="rcblk"):
        self.name = name
        self.nodes = [str(node)]
        self.C = torch.tensor(float(capacitance), dtype=DTYPE)

    def parameters(self):
        """The capacitance, exposed as a differentiable leaf."""
        return {"C": self.C}

    def residual_and_jacobian(self, x, xdot, t):
        """``[dq/dt, q - C*v]``: the node KCL row and the state equation."""
        v, q = x[0], x[1]
        qdot = xdot[1]
        r = torch.stack([qdot, q - self.C * v])
        zero = torch.zeros((), dtype=DTYPE)
        one = torch.ones((), dtype=DTYPE)
        jx = torch.stack([
            torch.stack([zero, zero]),
            torch.stack([-self.C, one]),
        ])
        jd = torch.stack([
            torch.stack([zero, one]),
            torch.stack([zero, zero]),
        ])
        return r, jx, jd
