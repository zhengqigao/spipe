"""SPICE netlist parser.

Produces a flat, subcircuit-expanded intermediate representation consisting of

* :class:`Element` records  -- one per device instance,
* :class:`ModelCard` records -- ``.model`` cards,
* analysis / control directives.

The parser is deliberately separate from device models and from the solver:
adding a device requires no change here beyond (optionally) a node-count entry
in :data:`NODE_COUNTS`.

Supported syntax
----------------
``*`` full-line comments, ``;`` trailing comments, ``+`` line continuation,
``.param``, ``.model``, ``.ic``, ``.nodeset``, ``.subckt``/``.ends``/``X``,
``.op``, ``.tran``, ``.dc``, ``.options``, ``.include``, ``.global``,
``.end`` and SPICE engineering suffixes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .units import SpiceSyntaxError, eval_expr, is_number

__all__ = [
    "Element",
    "ModelCard",
    "ParsedNetlist",
    "parse_netlist",
    "NODE_COUNTS",
]

#: number of connection nodes taken from the card, per device letter
NODE_COUNTS = {
    "r": 2,
    "c": 2,
    "l": 2,
    "v": 2,
    "i": 2,
    "e": 4,
    "g": 4,
    "f": 2,
    "h": 2,
    "d": 2,
    "m": 4,
    "q": 3,
    "s": 4,
    "w": 2,
}

_GROUND_NAMES = {"0", "gnd", "gnd!", "ground"}


@dataclass
class Element:
    """One flattened device instance."""

    letter: str                      # lowercase device letter, e.g. 'm'
    name: str                        # full instance name, lowercase, hierarchical
    nodes: List[str]                 # connection node names (lowercase)
    model: Optional[str] = None      # referenced .model name (lowercase) or None
    args: List[str] = field(default_factory=list)      # positional leftovers
    kwargs: Dict[str, str] = field(default_factory=dict)  # KEY=VALUE pairs (lowercased keys)
    ctrl: Optional[str] = None       # controlling source name for F/H/W
    params: Dict[str, float] = field(default_factory=dict)  # resolved .param scope
    source: Optional[str] = None     # raw card text (for error messages)


@dataclass
class ModelCard:
    name: str
    mtype: str
    params: Dict[str, float]


@dataclass
class ParsedNetlist:
    title: str = ""
    elements: List[Element] = field(default_factory=list)
    models: Dict[str, ModelCard] = field(default_factory=dict)
    params: Dict[str, float] = field(default_factory=dict)
    ic: Dict[str, float] = field(default_factory=dict)
    nodeset: Dict[str, float] = field(default_factory=dict)
    options: Dict[str, object] = field(default_factory=dict)
    tran: Optional[Tuple[float, float, float, bool]] = None
    dc_sweeps: List[Tuple[str, float, float, float]] = field(default_factory=list)
    has_op: bool = False
    globals: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# lexical pre-processing
# --------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    out = []
    in_sq = False
    for ch in line:
        if ch == "'":
            in_sq = not in_sq
        if not in_sq and (ch == ";" or ch == "$"):
            break
        out.append(ch)
    return "".join(out)


def _logical_lines(text: str, basedir: Optional[str] = None, _depth: int = 0) -> List[str]:
    """Join continuations, drop comments, expand ``.include``."""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: List[str] = []
    for line in raw:
        s = line.rstrip()
        if not s.strip():
            continue
        if s.lstrip().startswith("*"):
            continue
        s = _strip_comment(s)
        if not s.strip():
            continue
        if s.lstrip().startswith("+"):
            if not lines:
                raise SpiceSyntaxError("continuation line with nothing to continue: %r" % line)
            lines[-1] = lines[-1] + " " + s.lstrip()[1:].strip()
        else:
            lines.append(s.strip())

    if _depth > 8:
        return lines
    out: List[str] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith(".include") or low.startswith(".inc "):
            toks = _tokenize(ln)
            if len(toks) < 2:
                raise SpiceSyntaxError("malformed .include: %r" % ln)
            path = toks[1].strip("'\"")
            if basedir and not os.path.isabs(path):
                path = os.path.join(basedir, path)
            with open(path, "r") as fh:
                out.extend(_logical_lines(fh.read(), os.path.dirname(path), _depth + 1))
        else:
            out.append(ln)
    return out


_TOK_RE = re.compile(
    r"""'[^']*'|"[^"]*"|\{[^}]*\}|\([^()]*\)|[^\s,=()]+|=|\(|\)""",
    re.VERBOSE,
)


def _tokenize(line: str) -> List[str]:
    """Split a card into tokens.  Parenthesised groups stay together, commas
    act as whitespace, ``=`` is its own token."""
    return [t for t in _TOK_RE.findall(line) if t.strip()]


def _split_kwargs(tokens: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Separate ``KEY=VALUE`` pairs from positional tokens."""
    pos: List[str] = []
    kw: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 2 < len(tokens) + 0 and i + 1 < len(tokens) and tokens[i + 1] == "=":
            kw[t.lower()] = tokens[i + 2] if i + 2 < len(tokens) else ""
            i += 3
            continue
        if "=" in t and not t.startswith(("'", '"', "{", "(")):
            k, _, v = t.partition("=")
            if k:
                kw[k.lower()] = v
                i += 1
                continue
        pos.append(t)
        i += 1
    return pos, kw


def _norm_node(n: str) -> str:
    n = n.strip().lower()
    return "0" if n in _GROUND_NAMES else n


# --------------------------------------------------------------------------
# subcircuit handling
# --------------------------------------------------------------------------

@dataclass
class _Subckt:
    name: str
    ports: List[str]
    defaults: Dict[str, str]
    lines: List[str]


def parse_netlist(text: str, basedir: Optional[str] = None) -> ParsedNetlist:
    """Parse *text* into a :class:`ParsedNetlist` (subcircuits expanded)."""
    lines = _logical_lines(text, basedir)
    net = ParsedNetlist()

    # ---- title line -------------------------------------------------------
    if lines and not _is_card(lines[0]):
        net.title = lines[0]
        lines = lines[1:]

    # ---- collect subcircuit definitions ----------------------------------
    subckts: Dict[str, _Subckt] = {}
    top: List[str] = []
    stack: List[_Subckt] = []
    for ln in lines:
        low = ln.lower()
        if low.startswith(".subckt") or low.startswith(".macro"):
            toks = _tokenize(ln)[1:]
            pos, kw = _split_kwargs(toks)
            if not pos:
                raise SpiceSyntaxError("malformed .subckt: %r" % ln)
            sub = _Subckt(pos[0].lower(), [_norm_node(p) for p in pos[1:]], kw, [])
            stack.append(sub)
            continue
        if low.startswith(".ends") or low.startswith(".eom"):
            if not stack:
                raise SpiceSyntaxError(".ends without .subckt")
            sub = stack.pop()
            subckts[sub.name] = sub
            continue
        if stack:
            stack[-1].lines.append(ln)
        else:
            top.append(ln)
    if stack:
        raise SpiceSyntaxError(".subckt %r is never closed with .ends" % stack[-1].name)

    # ---- global .param pass ----------------------------------------------
    params: Dict[str, float] = {}
    for ln in top:
        if ln.lower().startswith(".param"):
            _apply_param_card(ln, params)
    net.params = dict(params)

    # ---- walk the top level ----------------------------------------------
    _walk(top, subckts, net, params, prefix="", node_map=None, depth=0)
    return net


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:#\[\]$+-]*$")
_SRC_WORDS = {"dc", "ac", "pulse", "pwl", "sin", "sine", "exp", "sffm"}


def _is_card(line: str) -> bool:
    """True if *line* is a real card rather than the deck's title line.

    SPICE consumes the first line of a deck as a title, so a deck that starts
    with ``.model`` would silently lose it.  We therefore only treat the first
    line as a title when it does *not* parse as a card -- which keeps both
    conventional decks ("RC ladder test") and bare fragments
    ("V1 in 0 DC 1") working.
    """
    s = line.strip()
    if not s:
        return False
    if s.startswith("."):
        return True
    toks = _tokenize(s)
    if len(toks) < 3:
        return False
    letter = s[0].lower()
    if letter not in NODE_COUNTS and letter != "x":
        return False
    if not _NAME_RE.match(toks[0]):
        return False
    pos, kw = _split_kwargs(toks[1:])
    if kw:
        return True
    nn = 2 if letter == "x" else NODE_COUNTS[letter]
    if len(pos) < nn:
        return False
    for n in pos[:nn]:
        if not (_NAME_RE.match(n) or n.isdigit()):
            return False
    rest = pos[nn:]
    if letter in ("r", "c", "l"):
        return bool(rest) and is_number(rest[0])
    if letter in ("v", "i"):
        if not rest:
            return False
        head = rest[0].lower().split("(")[0]
        return is_number(rest[0]) or head in _SRC_WORDS
    if letter in ("e", "g"):
        return bool(rest) and is_number(rest[0])
    if letter in ("f", "h"):
        return len(rest) >= 2 and is_number(rest[1])
    if letter in ("d", "m", "q", "s", "w", "x"):
        return bool(rest)
    return True


def _apply_param_card(line: str, params: Dict[str, float]) -> None:
    toks = _tokenize(line)[1:]
    _, kw = _split_kwargs(toks)
    for k, v in kw.items():
        params[k] = eval_expr(v, params)


def _walk(lines, subckts, net, params, prefix, node_map, depth):
    if depth > 32:
        raise SpiceSyntaxError("subcircuit nesting deeper than 32 levels")

    local_params = dict(params)
    for ln in lines:
        if ln.lower().startswith(".param"):
            _apply_param_card(ln, local_params)

    def xnode(raw: str) -> str:
        n = _norm_node(raw)
        if n == "0":
            return "0"
        if node_map is not None and n in node_map:
            return node_map[n]
        if node_map is not None and n in net.globals:
            return n
        return prefix + n if node_map is not None else n

    for ln in lines:
        low = ln.lower()
        if low.startswith("."):
            _handle_dot(ln, net, local_params, prefix, node_map, xnode, depth)
            continue

        toks = _tokenize(ln)
        name = toks[0]
        letter = name[0].lower()
        full = (prefix + name).lower()

        if letter == "x":
            pos, kw = _split_kwargs(toks[1:])
            if not pos:
                raise SpiceSyntaxError("malformed subcircuit call: %r" % ln)
            subname = pos[-1].lower()
            if subname not in subckts:
                raise SpiceSyntaxError(
                    "subcircuit %r used by %s is not defined" % (subname, name)
                )
            sub = subckts[subname]
            conns = [xnode(p) for p in pos[:-1]]
            if len(conns) != len(sub.ports):
                raise SpiceSyntaxError(
                    "instance %s connects %d nodes but .subckt %s declares %d"
                    % (name, len(conns), subname, len(sub.ports))
                )
            sub_params = dict(local_params)
            for k, v in sub.defaults.items():
                sub_params[k] = eval_expr(v, sub_params)
            for k, v in kw.items():
                sub_params[k] = eval_expr(v, local_params)
            nmap = {p: c for p, c in zip(sub.ports, conns)}
            nmap["0"] = "0"
            _walk(
                sub.lines, subckts, net, sub_params,
                prefix=full + ".", node_map=nmap, depth=depth + 1,
            )
            continue

        if letter not in NODE_COUNTS:
            raise SpiceSyntaxError(
                "unknown device letter %r in card: %r" % (name[0], ln)
            )

        nn = NODE_COUNTS[letter]
        pos, kw = _split_kwargs(toks[1:])
        if len(pos) < nn:
            raise SpiceSyntaxError(
                "device %s needs %d nodes, card only has %d tokens: %r"
                % (name, nn, len(pos), ln)
            )
        nodes = [xnode(p) for p in pos[:nn]]
        rest = pos[nn:]

        elem = Element(
            letter=letter, name=full, nodes=nodes,
            kwargs={k: v for k, v in kw.items()},
            params=local_params, source=ln,
        )

        # controlling source name for F / H / W
        if letter in ("f", "h"):
            if not rest:
                raise SpiceSyntaxError("%s needs a controlling source name: %r" % (name, ln))
            elem.ctrl = (prefix + rest[0]).lower()
            rest = rest[1:]
        elif letter == "w":
            if len(rest) < 2:
                raise SpiceSyntaxError("%s needs <vctrl> <model>: %r" % (name, ln))
            elem.ctrl = (prefix + rest[0]).lower()
            elem.model = rest[1].lower()
            rest = rest[2:]
        elif letter in ("d", "m", "q", "s"):
            if letter == "q" and len(rest) >= 2 and not is_number(rest[0]):
                # optional 4th (substrate) node before the model name
                nodes = nodes + [xnode(rest[0])]
                elem.nodes = nodes
                rest = rest[1:]
            if not rest:
                raise SpiceSyntaxError("%s needs a model name: %r" % (name, ln))
            elem.model = rest[0].lower()
            rest = rest[1:]

        elem.args = rest
        net.elements.append(elem)


def _handle_dot(ln, net, params, prefix, node_map, xnode, depth):
    toks = _tokenize(ln)
    card = toks[0].lower()

    if card == ".model":
        if len(toks) < 3:
            raise SpiceSyntaxError("malformed .model: %r" % ln)
        mname = toks[1].lower()
        mtype = toks[2].strip("()").lower()
        rest = toks[3:]
        joined = " ".join(t.strip("()") for t in rest)
        # the type token may itself carry the opening paren with params
        if "(" in toks[2]:
            head, _, tail = toks[2].partition("(")
            mtype = head.lower()
            joined = tail.strip(")") + " " + joined
        _, kw = _split_kwargs(_tokenize(joined))
        mp = {}
        for k, v in kw.items():
            try:
                mp[k] = eval_expr(v, params)
            except SpiceSyntaxError:
                mp[k] = v
        net.models[mname] = ModelCard(mname, mtype, mp)
        return

    if card == ".temp":
        vals = [eval_expr(t, params) for t in toks[1:] if is_number(t)]
        if vals:
            net.options["temp"] = vals[0]
        return

    if card in (".param", ".include", ".inc", ".end", ".probe", ".print",
                ".plot", ".save", ".title", ".lib", ".width",
                ".measure", ".meas", ".ac", ".four", ".noise", ".tf",
                ".disto", ".sens", ".step", ".alter", ".protect", ".unprotect",
                ".data", ".enddata", ".endl", ".del"):
        return

    if card == ".global":
        for t in toks[1:]:
            n = _norm_node(t)
            if n not in net.globals:
                net.globals.append(n)
        return

    if card == ".options" or card == ".option":
        pos, kw = _split_kwargs(toks[1:])
        for k, v in kw.items():
            try:
                net.options[k] = eval_expr(v, params)
            except SpiceSyntaxError:
                net.options[k] = v
        for t in pos:                      # bare flags such as `.options post`
            net.options.setdefault(t.lower(), True)
        return

    if card in (".ic", ".nodeset"):
        target = net.ic if card == ".ic" else net.nodeset
        body = ln.split(None, 1)[1] if len(ln.split(None, 1)) > 1 else ""
        pat = re.compile(
            r"([A-Za-z_][\w.:#\[\]]*\s*\([^)]*\)|[A-Za-z_0-9][\w.:#\[\]]*)"
            r"\s*=\s*('[^']*'|\{[^}]*\}|[^\s,=]+)"
        )
        for m in pat.finditer(body):
            key = m.group(1).strip()
            mm = re.match(r"^[vViI]\s*\(\s*(.*?)\s*\)$", key)
            if mm:
                key = mm.group(1)
            key = _norm_node(key)
            if node_map is not None:
                key = node_map.get(key, prefix + key)
            target[key] = eval_expr(m.group(2), params)
        return

    if card == ".op":
        net.has_op = True
        return

    if card == ".tran":
        pos, kw = _split_kwargs(toks[1:])
        uic = any(p.lower() == "uic" for p in pos)
        pos = [p for p in pos if p.lower() != "uic"]
        if len(pos) < 2:
            raise SpiceSyntaxError("malformed .tran: %r" % ln)
        tstep = eval_expr(pos[0], params)
        tstop = eval_expr(pos[1], params)
        tstart = eval_expr(pos[2], params) if len(pos) > 2 else 0.0
        if "uic" in kw:
            uic = True
        net.tran = (tstep, tstop, tstart, uic)
        return

    if card == ".dc":
        pos, _ = _split_kwargs(toks[1:])
        i = 0
        while i + 3 < len(pos) + 1 and i + 3 <= len(pos):
            src = pos[i].lower()
            start = eval_expr(pos[i + 1], params)
            stop = eval_expr(pos[i + 2], params)
            step = eval_expr(pos[i + 3], params)
            net.dc_sweeps.append((src, start, stop, step))
            i += 4
        return

    # unknown dot-card: ignore but remember
    net.options.setdefault("_ignored_cards", [])
    net.options["_ignored_cards"].append(ln)
