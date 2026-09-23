"""SPIPE's native differentiable SPICE-class analog circuit simulator.

Pure Python + PyTorch + NumPy, float64 throughout, natively differentiable
through both reverse-mode autograd and an explicit time-domain adjoint.

    from spipe.electronic.native import Netlist

    ckt = Netlist.from_string(netlist_text)
    op  = ckt.op()                          # {node: float64 scalar tensor}
    res = ckt.tran(1e-12, 5e-9)             # TranResult

    w = ckt.param('M1', 'W').requires_grad_(True)
    res = ckt.tran(1e-12, 5e-9)
    res.v('out').sum().backward()           # autograd path
    g = ckt.adjoint_grad(lambda r: r.v('out').sum(),
                         params=[('M1', 'W')])   # adjoint path

See ``README.md`` in this package for the supported syntax, the device table
and worked examples.
"""

from __future__ import annotations

from .circuit import (CircuitError, ConvergenceError, Netlist,
                      SpiceSyntaxError, UnknownParameterError)
from .external import BehaviouralResistor, ExternalBlock, RCStateBlock
from .newton import NewtonOptions
from .results import DCResult, OpResult, TranResult

__all__ = [
    "Netlist",
    "TranResult", "OpResult", "DCResult",
    "ExternalBlock", "BehaviouralResistor", "RCStateBlock",
    "NewtonOptions",
    "CircuitError", "ConvergenceError", "SpiceSyntaxError",
    "UnknownParameterError",
]

__version__ = "1.0.0"
