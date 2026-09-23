"""Modified-nodal-analysis assembly.

The whole circuit -- devices *and* external blocks -- is written as the
semi-explicit DAE

.. math::

    f(x, t) \\;+\\; \\frac{d}{dt} q(x) \\;+\\; r_{blk}(x, \\dot x, t) \\;=\\; 0

``x`` stacks node voltages (ground excluded) and the branch currents required
by voltage sources, inductors, ``E`` and ``H`` elements, followed by the state
variables owned by external blocks.

Only :class:`MNASystem` knows about indices; devices and blocks never do.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .devices import REGISTRY, DTYPE, LoadContext
from .parser import Element, ParsedNetlist

__all__ = ["MNASystem", "CircuitError"]

BOLTZMANN = 1.380649e-23
CHARGE = 1.602176634e-19


class CircuitError(RuntimeError):
    """Raised for structural problems: unknown device, unconnected node, ..."""


class MNASystem:
    """Index bookkeeping + residual/Jacobian assembly."""

    def __init__(self, parsed: ParsedNetlist, options: Optional[dict] = None):
        self.parsed = parsed
        self.options = dict(options or {})
        self.temp_c = float(self.options.get("temp", parsed.options.get("temp", 27.0)))
        self.vt = BOLTZMANN * (273.15 + self.temp_c) / CHARGE
        self.attach_params = False

        self._node_names: List[str] = []
        self._node_map: Dict[str, int] = {}
        self._internal: List[str] = []
        self._branch_map: Dict[Tuple[str, int], int] = {}
        self._branch_names: List[str] = []
        self.blocks: List[object] = []
        self._block_index: List[torch.Tensor] = []
        self._block_states: List[Tuple[int, int]] = []

        self.groups = []
        self.device_by_name: Dict[str, object] = {}
        self.group_of_device: Dict[str, object] = {}

        self._build(parsed.elements)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build(self, elements: Sequence[Element]):
        by_letter: Dict[str, list] = {}
        for elem in elements:
            cls = REGISTRY.get(elem.letter)
            if cls is None:
                raise CircuitError(
                    "unknown device type %r for instance %s (card: %s)"
                    % (elem.letter.upper(), elem.name, elem.source)
                )
            if elem.name in self.device_by_name:
                raise CircuitError("duplicate device name %r" % elem.name)
            dev = cls.build(elem, self)
            self.device_by_name[elem.name] = dev
            by_letter.setdefault(elem.letter, []).append(dev)

        # public nodes first, in netlist order of first appearance
        for elem in elements:
            for nm in elem.nodes:
                self._register_node(nm)
        for letter in by_letter:
            for dev in by_letter[letter]:
                for nm in dev.nodes:
                    self._register_node(nm)

        # branches, in device order
        for letter, devs in by_letter.items():
            cls = REGISTRY[letter]
            if cls.n_branch:
                for dev in devs:
                    for k in range(cls.n_branch):
                        self._branch_map[(dev.name, k)] = len(self._branch_names)
                        self._branch_names.append(dev.name)

        order = ["r", "c", "l", "v", "i", "e", "g", "f", "h", "d", "m", "q", "s", "w"]
        for letter in order:
            if letter in by_letter:
                grp = REGISTRY[letter](by_letter[letter], self)
                self.groups.append(grp)
                for dev in by_letter[letter]:
                    self.group_of_device[dev.name] = grp

        self.n_nodes = len(self._node_names)
        self.n_branch = len(self._branch_names)
        self.n_block_states = 0
        self._refresh_size()
        # resolve controlling-source references now, so a bad reference is
        # reported at parse time rather than from inside the Newton loop
        for grp in self.groups:
            if hasattr(grp, "ctrl_idx"):
                grp.ctrl_idx()

    def _refresh_size(self):
        self.n = self.n_nodes + self.n_branch + self.n_block_states
        self.ground = self.n  # index of the discard bin

    def _register_node(self, name: str) -> int:
        name = name.lower()
        if name == "0":
            return -1
        idx = self._node_map.get(name)
        if idx is None:
            idx = len(self._node_names)
            self._node_map[name] = idx
            self._node_names.append(name)
            if "#" in name:
                self._internal.append(name)
        return idx

    # ------------------------------------------------------------------
    # lookups used by device groups
    # ------------------------------------------------------------------
    def node_index(self, name: str) -> int:
        """Global index of *name*; ground maps to the discard bin."""
        name = str(name).lower()
        if name in ("0", "gnd", "gnd!", "ground"):
            return self.ground
        idx = self._node_map.get(name)
        if idx is None:
            raise CircuitError("unknown node %r" % name)
        return idx

    def branch_index(self, devname: str, k: int = 0) -> int:
        try:
            return self.n_nodes + self._branch_map[(devname.lower(), k)]
        except KeyError:
            raise CircuitError("device %r has no branch current" % devname)

    def ctrl_branch_index(self, ctrl: str, requester: str) -> int:
        key = (str(ctrl).lower(), 0)
        if key not in self._branch_map:
            raise CircuitError(
                "%s refers to controlling source %r, which is not a voltage "
                "source / inductor present in the circuit" % (requester, ctrl)
            )
        return self.n_nodes + self._branch_map[key]

    def model_params(self, name, allowed, devname) -> dict:
        if name is None:
            raise CircuitError("device %s has no model name" % devname)
        card = self.parsed.models.get(str(name).lower())
        if card is None:
            raise CircuitError(
                "device %s refers to undefined .model %r" % (devname, name)
            )
        if allowed and card.mtype.lower() not in allowed:
            raise CircuitError(
                "device %s uses .model %r of type %r; expected one of %s"
                % (devname, name, card.mtype, ", ".join(allowed))
            )
        return card.params

    def model_type(self, name) -> str:
        card = self.parsed.models.get(str(name).lower())
        return card.mtype.lower() if card else ""

    # ------------------------------------------------------------------
    # public naming
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> List[str]:
        """Circuit node names, ground excluded, internal nodes hidden."""
        return [n for n in self._node_names if "#" not in n]

    @property
    def all_nodes(self) -> List[str]:
        return list(self._node_names)

    def branch_devices(self) -> List[str]:
        return list(self._branch_names)

    # ------------------------------------------------------------------
    # external blocks
    # ------------------------------------------------------------------
    def add_external_block(self, block) -> None:
        """Attach *block* (see :mod:`spipe.electronic.native.external`).

        Nodes named by the block that do not yet exist are created, so a block
        may introduce its own terminals.  All index bookkeeping is rebuilt
        afterwards, which is why blocks may be added at any time before an
        analysis runs.
        """
        nodes = [str(nm).lower() for nm in getattr(block, "nodes", [])]
        nstate = int(getattr(block, "n_states", 0))
        if nstate < 0:
            raise CircuitError("external block %r declares n_states < 0"
                               % getattr(block, "name", block))
        for nm in nodes:
            if nm not in ("0", "gnd", "gnd!", "ground") and nm not in self._node_map:
                self._register_node(nm)
        self.n_nodes = len(self._node_names)
        self.blocks.append(block)
        self._block_states.append((0, nstate))
        self._reindex_blocks()
        if hasattr(block, "attach"):
            s0, ns = self._block_states[-1]
            idx = self._block_index[-1].tolist()
            block.attach(self, idx[:len(nodes)], list(range(s0, s0 + ns)))

    def _reindex_blocks(self):
        """Recompute every block's global index vector."""
        self.n_block_states = sum(ns for _, ns in self._block_states)
        self._refresh_size()
        off = self.n_nodes + self.n_branch
        states = []
        index = []
        for blk, (_, ns) in zip(self.blocks, self._block_states):
            nodes = [str(nm).lower() for nm in getattr(blk, "nodes", [])]
            idx = [self.node_index(nm) for nm in nodes] + list(range(off, off + ns))
            states.append((off, ns))
            index.append(torch.tensor(idx, dtype=torch.long))
            off += ns
        self._block_states = states
        self._block_index = index
        for g in self.groups:
            g._idx_cache.clear()

    def block_initial_state(self) -> torch.Tensor:
        """Initial values for every block state (used to seed the DC solve)."""
        out = torch.zeros(self.n, dtype=DTYPE)
        for blk, (s0, ns) in zip(self.blocks, self._block_states):
            if ns and hasattr(blk, "init_state"):
                v = blk.init_state()
                if v is not None:
                    out[s0:s0 + ns] = torch.as_tensor(v, dtype=DTYPE).reshape(-1)
        return out

    # ------------------------------------------------------------------
    # assembly
    # ------------------------------------------------------------------
    def eval(self, x, t=0.0, xdot=None, need_jac=True, xold=None, gmin=0.0,
             gshunt=0.0, src_scale=1.0, dc=False, limiting=True,
             charge_only=False):
        """Assemble the DAE pieces at state *x*.

        Returns a dict with ``f``, ``q`` (``(n,)``), ``Jf``, ``Jq``
        (``(n,n)`` or ``None``) and, if external blocks are present,
        ``rb``, ``Jbx``, ``Jbd``.
        """
        n = self.n
        xf = torch.cat([x, torch.zeros(1, dtype=DTYPE)])
        xo = xf if xold is None else torch.cat([xold, torch.zeros(1, dtype=DTYPE)])
        ctx = LoadContext(xf, t, n, need_jac=need_jac, xold=xo, gmin=gmin,
                          src_scale=src_scale, dc=dc, limiting=limiting)
        for g in self.groups:
            if charge_only and not g.has_charge:
                continue
            g.load(ctx)
        f, q, jf, jq = ctx.finish()

        if gshunt:
            f = f + gshunt * x
            if need_jac:
                jf = jf + gshunt * torch.eye(n, dtype=DTYPE)

        out = {"f": f, "q": q, "Jf": jf, "Jq": jq, "rb": None,
               "Jbx": None, "Jbd": None}
        out["limited"] = ctx.limited
        if need_jac:
            fm, qm = ctx.magnitudes()
            out["fmag"], out["qmag"] = fm, qm

        if self.blocks and not charge_only:
            xd = torch.zeros(n, dtype=DTYPE) if xdot is None else xdot
            xdf = torch.cat([xd, torch.zeros(1, dtype=DTYPE)])
            rb = torch.zeros(n + 1, dtype=DTYPE)
            jbx = torch.zeros(n + 1, n + 1, dtype=DTYPE) if need_jac else None
            jbd = torch.zeros(n + 1, n + 1, dtype=DTYPE) if need_jac else None
            for blk, idx in zip(self.blocks, self._block_index):
                r, jx, jd = blk.residual_and_jacobian(xf[idx], xdf[idx], t)
                r = torch.as_tensor(r, dtype=DTYPE).reshape(-1)
                if r.numel() != idx.numel():
                    raise CircuitError(
                        "external block %r returned %d residual entries but "
                        "declares %d indices"
                        % (getattr(blk, "name", blk), r.numel(), idx.numel())
                    )
                rb = rb.index_add(0, idx, r)
                if need_jac:
                    out["fmag"] = out["fmag"].index_add(
                        0, torch.clamp(idx, max=n - 1),
                        torch.as_tensor(r, dtype=DTYPE).reshape(-1).detach().abs())
                if need_jac:
                    rr = idx.reshape(-1, 1).expand(idx.numel(), idx.numel()).reshape(-1)
                    cc = idx.reshape(1, -1).expand(idx.numel(), idx.numel()).reshape(-1)
                    jbx.index_put_((rr, cc), torch.as_tensor(jx, dtype=DTYPE)
                                   .reshape(-1).detach(), accumulate=True)
                    jbd.index_put_((rr, cc), torch.as_tensor(jd, dtype=DTYPE)
                                   .reshape(-1).detach(), accumulate=True)
            out["rb"] = rb[:n]
            if need_jac:
                out["Jbx"] = jbx[:n, :n]
                out["Jbd"] = jbd[:n, :n]
        return out

    # ------------------------------------------------------------------
    def zero_rows(self, J) -> List[str]:
        """Names of unknowns whose Jacobian row is identically zero."""
        bad = []
        rows = (J.abs().sum(dim=1) == 0).nonzero().flatten().tolist()
        for i in rows:
            bad.append(self.unknown_name(i))
        return bad

    def unknown_name(self, i: int) -> str:
        if i < self.n_nodes:
            return "v(%s)" % self._node_names[i]
        i -= self.n_nodes
        if i < self.n_branch:
            return "i(%s)" % self._branch_names[i]
        return "state[%d]" % (i - self.n_branch)

    def breakpoints(self, tstart, tstop) -> List[float]:
        bps = set()
        for g in self.groups:
            for b in g.breakpoints(tstart, tstop):
                bps.add(float(b))
        return sorted(bps)

    @property
    def is_linear(self) -> bool:
        """True when every equation is linear in x (fast path for transient)."""
        if self.blocks:
            return False
        return all(getattr(g, "letter", "") not in ("d", "m", "q", "s", "w")
                   for g in self.groups)

    def source_residual(self, t, dc=False) -> torch.Tensor:
        """``f(x=0, t)`` for a linear circuit.

        Every linear device contributes zero at ``x = 0``; only the
        independent sources remain, so the whole right-hand side reduces to a
        single scatter of the source values.
        """
        if getattr(self, "_src_struct", None) is None:
            idx, sign, grps = [], [], []
            for g in self.groups:
                lt = getattr(g, "letter", "")
                if lt == "v":
                    idx.append(g.branch_idx())
                    sign.append(-torch.ones(g.N, dtype=DTYPE))
                    grps.append((g, 1))
                elif lt == "i":
                    idx.append(g.node_idx(0))
                    sign.append(torch.ones(g.N, dtype=DTYPE))
                    idx.append(g.node_idx(1))
                    sign.append(-torch.ones(g.N, dtype=DTYPE))
                    grps.append((g, 2))
            self._src_struct = (
                torch.cat(idx) if idx else torch.zeros(0, dtype=torch.long),
                torch.cat(sign) if sign else torch.zeros(0, dtype=DTYPE),
                grps,
            )
        idx, sign, grps = self._src_struct
        out = torch.zeros(self.n + 1, dtype=DTYPE)
        if not grps:
            return out[:self.n]
        vals = []
        for g, rep in grps:
            v = g.values_at(t, dc)
            vals.extend([v] * rep)
        v = torch.cat(vals) if len(vals) > 1 else vals[0]
        return out.index_add(0, idx, sign * v)[:self.n]

    def source_residual_batch(self, t: torch.Tensor) -> torch.Tensor:
        """``(K, n)`` stack of ``f(x=0, t_k)`` for the K times in *t*.

        The batched twin of :meth:`source_residual`: one scatter for the whole
        window instead of one per time point.
        """
        if getattr(self, "_src_struct", None) is None:
            self.source_residual(0.0)
        idx, sign, grps = self._src_struct
        K = int(t.numel())
        out = torch.zeros(K, self.n + 1, dtype=DTYPE)
        if not grps:
            return out[:, :self.n]
        vals = []
        for g, rep in grps:
            v = g.values_batch(t)
            vals.extend([v] * rep)
        V = torch.cat(vals, dim=1) if len(vals) > 1 else vals[0]
        out.index_add_(1, idx, sign.unsqueeze(0) * V)
        return out[:, :self.n]

    def source_values(self, t) -> torch.Tensor:
        """All independent-source values at time *t* (detached), for
        classifying a breakpoint as a jump or a kink."""
        vals = []
        for g in self.groups:
            if hasattr(g, "devices") and getattr(g, "letter", "") in ("v", "i"):
                for d in g.devices:
                    p = {k: v.detach() for k, v in d.params.items()}
                    vals.append(d.extra["wf"].value(t, p).reshape(1))
        if not vals:
            return torch.zeros(0, dtype=DTYPE)
        return torch.cat(vals)

    def set_tran_defaults(self, tstep, tstop):
        for g in self.groups:
            if hasattr(g, "set_tran_defaults"):
                g.set_tran_defaults(tstep, tstop)

    def invalidate_param_cache(self):
        for g in self.groups:
            g.invalidate_cache()
