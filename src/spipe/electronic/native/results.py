"""Analysis result containers."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

DTYPE = torch.float64

__all__ = ["OpResult", "TranResult", "DCResult"]


class _Lookup:
    """Shared name -> column resolution for node voltages and branch currents."""

    def _vcol(self, name):
        key = str(name).lower()
        if key in ("0", "gnd", "gnd!", "ground"):
            return None
        m = self._nodemap
        if key in m:
            return m[key]
        if key.startswith("v(") and key.endswith(")"):
            return self._vcol(key[2:-1])
        raise KeyError(
            "unknown node %r; available nodes: %s"
            % (name, ", ".join(sorted(m)))
        )

    def _icol(self, name):
        key = str(name).lower()
        if key.startswith("i(") and key.endswith(")"):
            return self._icol(key[2:-1])
        m = self._branchmap
        if key in m:
            return m[key]
        raise KeyError(
            "no branch current for %r; currents are available for voltage "
            "sources, inductors, E and H elements: %s"
            % (name, ", ".join(sorted(m)))
        )


class OpResult(dict, _Lookup):
    """Operating point.

    A plain ``dict`` mapping **node name -> float64 scalar tensor**, with
    ``.v()`` / ``.i()`` accessors for convenience.
    """

    def __init__(self, nodes, x, nodemap, branchmap):
        super().__init__()
        self._x = x
        self._nodemap = nodemap
        self._branchmap = branchmap
        self.nodes = list(nodes)
        for nm in nodes:
            self[nm] = x[nodemap[nm]]

    def v(self, name):
        """Node voltage as a float64 scalar tensor (``0`` for ground)."""
        c = self._vcol(name)
        return torch.zeros((), dtype=DTYPE) if c is None else self._x[c]

    def i(self, name):
        """Branch current of a voltage source / inductor / E / H element."""
        return self._x[self._icol(name)]

    @property
    def x(self):
        """Raw MNA solution vector."""
        return self._x


class TranResult(_Lookup):
    """Transient analysis result.

    Attributes
    ----------
    t : torch.Tensor
        ``(N,)`` float64 time points.
    nodes : list[str]
        node names, ground excluded.
    x : torch.Tensor
        ``(N, n)`` raw MNA solution; ``v()``/``i()`` are views into it and
        stay attached to the autograd graph.
    """

    def __init__(self, t, x, nodes, nodemap, branchmap, record=None):
        self.t = t
        self.x = x
        self.nodes = list(nodes)
        self._nodemap = nodemap
        self._branchmap = branchmap
        self.record = record

    def v(self, name):
        """``(N,)`` node-voltage waveform."""
        c = self._vcol(name)
        if c is None:
            return torch.zeros_like(self.t)
        return self.x[:, c]

    def i(self, name):
        """``(N,)`` branch-current waveform."""
        return self.x[:, self._icol(name)]

    def __len__(self):
        return int(self.t.numel())

    def __repr__(self):  # pragma: no cover
        return "<TranResult %d points, %d unknowns>" % (len(self), self.x.shape[1])


class DCResult(_Lookup):
    """``.dc`` sweep result.

    ``sweep`` holds the swept source values; ``v()``/``i()`` return waveforms
    of the same length.
    """

    def __init__(self, sweep, x, nodes, nodemap, branchmap, source=""):
        self.sweep = sweep
        self.x = x
        self.nodes = list(nodes)
        self._nodemap = nodemap
        self._branchmap = branchmap
        self.source = source

    def v(self, name):
        """``(N,)`` node voltage across the sweep."""
        c = self._vcol(name)
        if c is None:
            return torch.zeros_like(self.sweep)
        return self.x[:, c]

    def i(self, name):
        """``(N,)`` branch current across the sweep."""
        return self.x[:, self._icol(name)]

    def __len__(self):
        return int(self.sweep.numel())
