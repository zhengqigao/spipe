"""SPICE number / expression utilities.

Handles the SPICE engineering suffixes (``meg``, ``k``, ``u``, ``n``, ``p``,
``f`` ...) and a small, *safe* arithmetic expression evaluator used for
``.param`` substitution and for ``W='2*wn'`` style device parameters.

Expressions are resolved **numerically at parse time**: every device parameter
ends up as an independent float64 leaf tensor, so that
``ckt.param('M1', 'W').requires_grad_(True)`` behaves the way the public API
promises.
"""

from __future__ import annotations

import ast
import math
import re

__all__ = ["parse_number", "eval_expr", "SpiceSyntaxError"]


class SpiceSyntaxError(ValueError):
    """Raised for malformed netlist numbers or expressions."""


# Ordered longest-first so that 'meg'/'mil' win over 'm'.
_SUFFIXES = (
    ("meg", 1e6),
    ("mil", 25.4e-6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
)

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def parse_number(text: str) -> float:
    """Parse a SPICE numeric literal, e.g. ``4.7k``, ``1meg``, ``0.18u``, ``1e-9``.

    Trailing alphabetic noise after a recognised suffix is ignored, matching
    SPICE (``1kohm`` == 1000.0).

    Raises
    ------
    SpiceSyntaxError
        if *text* does not begin with a number.
    """
    s = str(text).strip()
    m = _NUM_RE.match(s)
    if not m:
        raise SpiceSyntaxError("not a number: %r" % (text,))
    value = float(m.group(0))
    rest = s[m.end():].strip().lower()
    if rest:
        for suf, mult in _SUFFIXES:
            if rest.startswith(suf):
                value *= mult
                break
    return value


def is_number(text: str) -> bool:
    """True if *text* parses as a SPICE numeric literal."""
    try:
        parse_number(text)
        return True
    except SpiceSyntaxError:
        return False


# --------------------------------------------------------------------------
# Expression evaluation
# --------------------------------------------------------------------------

_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "pow": math.pow,
    "pwr": lambda x, y: math.copysign(abs(x) ** y, x),
    "int": int,
    "sgn": lambda x: (x > 0) - (x < 0),
    "floor": math.floor,
    "ceil": math.ceil,
}

_CONSTS = {"pi": math.pi, "e": math.e, "true": 1.0, "false": 0.0}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}

_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _strip_wrappers(text: str) -> str:
    s = text.strip()
    while len(s) >= 2 and (
        (s[0] == "{" and s[-1] == "}")
        or (s[0] == "'" and s[-1] == "'")
        or (s[0] == '"' and s[-1] == '"')
    ):
        s = s[1:-1].strip()
    return s


def _eval_node(node, params):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, params)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return 1.0 if node.value else 0.0
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise SpiceSyntaxError("bad constant %r in expression" % (node.value,))
    if isinstance(node, ast.Name):
        key = node.id.lower()
        if key in params:
            return float(params[key])
        if key in _CONSTS:
            return _CONSTS[key]
        raise SpiceSyntaxError("unknown parameter %r in expression" % (node.id,))
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise SpiceSyntaxError("unsupported operator in expression")
        return op(_eval_node(node.left, params), _eval_node(node.right, params))
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, params)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return 0.0 if val else 1.0
        raise SpiceSyntaxError("unsupported unary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SpiceSyntaxError("unsupported call in expression")
        fname = node.func.id.lower()
        fn = _FUNCS.get(fname)
        if fn is None:
            raise SpiceSyntaxError("unknown function %r in expression" % (node.func.id,))
        return float(fn(*[_eval_node(a, params) for a in node.args]))
    if isinstance(node, ast.IfExp):
        return (
            _eval_node(node.body, params)
            if _eval_node(node.test, params)
            else _eval_node(node.orelse, params)
        )
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise SpiceSyntaxError("chained comparisons are not supported")
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise SpiceSyntaxError("unsupported comparison")
        return 1.0 if op(_eval_node(node.left, params), _eval_node(node.comparators[0], params)) else 0.0
    raise SpiceSyntaxError("unsupported expression construct %s" % type(node).__name__)


def eval_expr(text, params=None):
    """Evaluate a netlist value: a plain SPICE number, or an arithmetic
    expression in ``{...}`` / ``'...'`` referring to ``.param`` names.

    Parameters
    ----------
    text : str
        the raw token, e.g. ``'4.7k'``, ``"{2*wn}"``, ``"'wn*2'"``.
    params : dict[str, float] | None
        parameter table (keys lowercased).

    Returns
    -------
    float
    """
    params = params or {}
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip()
    stripped = _strip_wrappers(raw)
    if not stripped:
        raise SpiceSyntaxError("empty value")
    # fast path: plain number with optional suffix
    if _NUM_RE.match(stripped):
        tail = stripped[_NUM_RE.match(stripped).end():].strip()
        if tail == "" or tail.lower().isalpha():
            return parse_number(stripped)
    key = stripped.lower()
    if key in params:
        return float(params[key])
    # replace SPICE suffixed literals inside the expression before parsing
    expr = _suffix_substitute(stripped)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise SpiceSyntaxError("cannot parse expression %r: %s" % (raw, exc))
    return float(_eval_node(tree, params))


_TOKEN_RE = re.compile(
    r"(?<![\w.])(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?([a-zA-Zµ]+)?"
)


def _suffix_substitute(expr: str) -> str:
    """Rewrite ``2.5k`` -> ``2500.0`` inside an expression string."""

    def repl(m):
        num = m.group(1) + (m.group(2) or "")
        suf = (m.group(3) or "").lower()
        if not suf:
            return num
        for s, mult in _SUFFIXES:
            if suf.startswith(s):
                return repr(float(num) * mult)
        # not a suffix -> it is a name glued to a number, leave untouched
        return m.group(0)

    return _TOKEN_RE.sub(repl, expr)
