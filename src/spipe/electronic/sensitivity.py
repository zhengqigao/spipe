"""Differentiable *electronic device* parameters -- the ``.sensparam`` card.

SPIPE has always been able to differentiate an optical output with respect to the
*electrical drive* of a modulator, and a photocurrent with respect to an optical field.  The
link that was missing is the one the paper is motivated by: the derivative of an optical
output with respect to an **electronic device parameter** -- a transistor width, a load
resistance -- which is the quantity a circuit designer actually turns.

This module is the declaration half of that link.  A netlist declares which device
parameters are differentiable in its ``.electronic`` section::

    .sensparam M1:W M1:L R3:R

and :class:`~spipe.electronic.electronic.Electronic` then

* exposes each of them as a float64 **leaf tensor** (``Electronic.param('M1', 'W')``), the
  same idiom :mod:`spipe.electronic.native` uses, so an optimiser can read ``.grad`` and
  write a new value;
* names them in the SPICE sensitivity request (Xyce ``.SENS ... param=``) alongside the
  photocurrent PWL parameters;
* returns their gradients from the backward pass.

Two classes of parameter, two analysis modes
--------------------------------------------
The photocurrent parameters ``param_{i}_{j}`` number ``pd_cnt * num_time_ein`` -- hundreds to
thousands of them -- while the device parameters declared here are a handful.  Xyce's
``direct`` sensitivity method costs one extra linear solve **per parameter**, so on paper the
adjoint is the right method for the photocurrent block and direct for the device block.

That is not the choice this code makes, and the reason is measured rather than theoretical.
On Xyce 7.10, a transient ``.SENS`` on a level-1 MOSFET width returns

======================================  ==========================
``.options sensitivity direct=1``       ``d V(d)/d M1:W = -4.997e4``  (0.075 % of central FD)
``.options sensitivity adjoint=1``      ``d V(d)/d M1:W =  0.0``      (and ``V(d)`` prints 0 too)
======================================  ==========================

Xyce warns that "at least one specified sensitivity parameter lacks an analytic derivative,
so numerical derivatives will be used", and its *transient adjoint* path does not deliver
them: it returns an identically zero block, silently.  A silent zero gradient is the worst
possible failure mode for an optimiser -- it looks like a converged design.

Xyce's ``.options sensitivity`` is a single global setting, not a per-``.SENS`` one, so the
two classes cannot use different methods in one run.  Therefore:

* **device parameters declared with** ``.sensparam`` **force** ``direct``.  If the caller
  asked for ``use_adjoint=True`` a warning says so and the request is overridden.
* **without** ``.sensparam`` the analysis mode is unchanged: ``use_adjoint`` is honoured
  exactly as before, and the ``.sens`` card carries exactly the photocurrent parameters it
  always did.  (The generated deck is not quite byte-for-byte what it was, because
  ``construct_for_xyce`` was writing its ``.PRINT TRAN`` card without a trailing newline and
  therefore ran it into the following ``.PRINT TRAN power...`` card -- which Xyce rejects, so
  no Xyce deck carrying a modulator or a detector could be run at all.  The HSPICE deck *is*
  byte-for-byte unchanged.)
* whichever method runs, :func:`guard_all_zero` and :func:`guard_analysis_ran` check what
  came back and raise rather than handing back a confident-looking array of zeros.

Cost, and what it buys
----------------------
The photocurrent block dominates the run time: ``direct`` costs one extra linear solve per
parameter per time step, so 401 samples with one detector turns a 4 ms transient into a
five-minute one.  It is only *needed* by a circuit whose photocurrent reaches a modulator, so
``spipe.config['xyce_sens_photocurrent'] = False`` drops it and SPIPE then establishes the
coupling by measurement instead -- see
:meth:`~spipe.electronic.electronic.Electronic.probe_decoupling`.

Accuracy
--------
Xyce's device-parameter derivatives are *numerical* (it says so), and their accuracy follows
the size of the derivative relative to the solver's own reproducibility.  Measured end to end
on a level-1 CMOS driver into an MZM: the dominant width sensitivity agrees with a central
finite difference over the whole chain to 8e-3, the sub-dominant ones to no better than a
factor of a few, and the finite-difference reference itself is inconsistent at the tens of
percent level because Xyce's transient is only reproducible to ~2e-4 relative with respect to
its own inputs.

The native back end (:mod:`spipe.electronic.native`) has neither trap -- ``W`` is a torch leaf
there, its adjoint is exact, and the same measurement comes out at 4e-5 -- which is why it is
the preferred back end for this.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = ['DeviceParameter', 'ZeroSensitivityError', 'parse_sensparam',
           'format_key', 'guard_all_zero', 'guard_analysis_ran', 'substitute_device_value',
           'sensparam_name']

#: A declared differentiable device parameter, ``('M1', 'W')``.  The device name keeps the
#: netlist's spelling (hierarchical names such as ``x1.m1`` are allowed); the parameter name
#: is upper-cased, which is how every SPICE dialect and the native engine spell it.
DeviceParameter = Tuple[str, str]


class ZeroSensitivityError(RuntimeError):
    """A sensitivity block came back identically zero while the objective was not constant.

    Raised instead of returning the block.  See :func:`guard_all_zero`.

    Attributes
    ----------
    parameters : list of str
        The parameters whose sensitivity block was identically zero.
    analysis : str
        The sensitivity method that produced it (``'direct'`` or ``'adjoint'``).
    backend : str
        Which electronic back end produced it.
    """

    def __init__(self, message: str, parameters: Sequence[str], analysis: str, backend: str):
        super().__init__(message)
        self.parameters = list(parameters)
        self.analysis = analysis
        self.backend = backend


# ``M1:W``, ``x1.m1:W``, ``R3 : R``.  The device name may contain the hierarchy separator
# '.', so the split is on the *last* colon.
_SENSPARAM_TOKEN = re.compile(r'^\s*(?P<device>[^\s:]+)\s*:\s*(?P<param>[A-Za-z_][A-Za-z_0-9]*)\s*$')


def format_key(device: str, name: str) -> str:
    """The canonical ``DEVICE:PARAM`` spelling of a declared parameter."""
    return f"{device}:{name.upper()}"


def sensparam_name(device: str, name: str) -> str:
    """The SPICE ``.PARAM`` identifier that carries this device parameter's value.

    Device and parameter names go through the same mangling on every back end, so a netlist
    rewritten for one engine can be diffed against the other.
    """
    stem = re.sub(r'[^A-Za-z0-9_]', '_', f"{device}_{name}")
    return f"sensparam_{stem.lower()}"


def parse_sensparam(lines: Iterable[str]) -> Tuple[List[str], List[DeviceParameter]]:
    """Pull every ``.sensparam`` card out of an ``.electronic`` section.

    :param lines: the lines of the ``.electronic`` section, as
        :func:`spipe.core.core._parse_file` produced them.
    :returns: ``(lines_without_sensparam, declared)`` where ``declared`` is a list of
        ``(device, PARAM)`` pairs in declaration order, with duplicates removed.
    :raises RuntimeError: on a token that is not ``DEVICE:PARAM``, naming the offending
        token and the expected syntax.

    The card accepts whitespace- or comma-separated tokens and may appear more than once::

        .sensparam M1:W M1:L
        .sensparam R3:R, C1:C

    Lines that are not ``.sensparam`` are returned untouched, in order, so a netlist that
    does not use the card produces exactly the list it was given.
    """
    kept: List[str] = []
    declared: List[DeviceParameter] = []
    seen = set()

    for line in lines:
        stripped = line.split('#')[0].strip()
        if not stripped.lower().startswith('.sensparam'):
            kept.append(line)
            continue
        body = stripped[len('.sensparam'):]
        tokens = [tok for tok in re.split(r'[\s,]+', body) if tok]
        if not tokens:
            raise RuntimeError(
                "'.sensparam' was given no parameters. The syntax is "
                "'.sensparam DEVICE:PARAM [DEVICE:PARAM ...]', e.g. '.sensparam M1:W R3:R'.")
        for token in tokens:
            match = _SENSPARAM_TOKEN.match(token)
            if match is None:
                raise RuntimeError(
                    f"'.sensparam' could not parse {token!r}. Every entry must be "
                    f"'DEVICE:PARAM' -- the instance name of a device in the .electronic "
                    f"section, a colon, and the parameter name, e.g. 'M1:W', 'R3:R', "
                    f"'x1.m2:L'.")
            device = match.group('device')
            name = match.group('param').upper()
            key = (device.lower(), name)
            if key in seen:
                continue
            seen.add(key)
            declared.append((device, name))
        # The card is a SPIPE directive, not SPICE: drop it, but keep the line count so that
        # error messages elsewhere still point at the right line.
        kept.append('\n')

    return kept, declared


def guard_all_zero(block: 'torch.Tensor',
                   parameters: Sequence[str],
                   objective: Optional['torch.Tensor'],
                   analysis: str,
                   backend: str,
                   objective_name: str = 'the objective') -> None:
    """Raise if *block* is identically zero while *objective* is not constant.

    This is the guard against the measured Xyce trap documented in the module docstring: a
    transient adjoint sensitivity request for a device parameter comes back as an array of
    exact zeros, with no error and no non-zero exit status.  Returning that array would tell
    an optimiser it had reached a stationary point.

    :param block: the sensitivity values for one parameter (any shape).  Checked with an
        exact ``== 0`` test, not a tolerance: a genuinely tiny gradient is a legitimate
        answer, an *exactly* zero one over a whole transient is not.
    :param parameters: the names of the parameters the block belongs to, for the message.
    :param objective: the objective waveform, if available.  When it is constant (a node
        that genuinely does not move) a zero sensitivity is correct and nothing is raised.
        Pass ``None`` when the waveform is not to hand; the check then fires on the zero
        block alone.
    :param analysis: ``'direct'`` or ``'adjoint'`` -- named in the message because switching
        method is the fix.
    :param backend: ``'xyce'``, ``'hspice'`` or ``'native'``.
    :raises ZeroSensitivityError:
    """
    if block.numel() == 0:
        return
    if bool((block != 0).any()):
        return

    if objective is not None and objective.numel() > 1:
        spread = float(objective.max() - objective.min())
        reference = float(objective.abs().max())
        if spread <= 1e-12 * max(reference, 1.0):
            # The objective really is constant over the record; zero is the right answer.
            return

    listed = ', '.join(parameters) if parameters else '(unnamed)'
    hint = ''
    if analysis == 'adjoint':
        hint = (" Xyce's *transient* adjoint returns an identically zero block for device "
                "parameters (measured on 7.10: direct=1 gives -4.997e+04 for d V(d)/d M1:W, "
                "adjoint=1 gives 0.0). Re-run with the direct method -- SPIPE does that "
                "automatically when a '.sensparam' card is present, so this deck was built "
                "with an explicit use_adjoint=True.")
    elif backend == 'xyce':
        hint = (" Check that the parameter name matches a device instance parameter Xyce "
                "knows (it is case sensitive in the .SENS card), and that the device "
                "actually influences the objective node.")
    raise ZeroSensitivityError(
        f"The {backend} {analysis} sensitivity block for {listed} is identically zero, but "
        f"{objective_name} is not constant over the transient. SPIPE refuses to return a "
        f"silently zero gradient: to an optimiser it is indistinguishable from a converged "
        f"design.{hint}",
        parameters, analysis, backend)


def guard_analysis_ran(seen_objective: 'torch.Tensor',
                       objective: 'torch.Tensor',
                       analysis: str,
                       backend: str,
                       parameters: Sequence[str] = ()) -> None:
    """Raise if the sensitivity analysis reported a *zero objective* the transient did not.

    Xyce's transient adjoint does not merely return zero derivatives: it prints the objective
    itself as ``0.00000000e+00`` for every time point, while the ordinary transient output of
    the same run shows the node swinging rails.  That disagreement is unambiguous -- it says
    the sensitivity analysis produced nothing at all, rather than that the derivative happens
    to be zero -- and it is the one check that catches the trap even for the *photocurrent*
    parameters, whose block is legitimately zero on any feedback-free circuit.

    :param seen_objective: the objective column(s) of the sensitivity output.
    :param objective: the same node(s) from the transient output.
    :raises ZeroSensitivityError:
    """
    if seen_objective.numel() == 0 or objective is None or objective.numel() == 0:
        return
    if bool((seen_objective != 0).any()):
        return
    spread = float(objective.max() - objective.min())
    if spread <= 1e-12 * max(float(objective.abs().max()), 1.0):
        return
    listed = ', '.join(parameters) if parameters else 'the declared parameters'
    hint = ''
    if analysis == 'adjoint':
        hint = (" This is Xyce's transient adjoint (measured on 7.10): with "
                "'.options sensitivity adjoint=1' both the objective and every sensitivity "
                "print as exactly zero, with no error and a zero exit status. Use "
                "'.options sensitivity direct=1' -- SPIPE selects it automatically whenever a "
                "'.sensparam' card is present.")
    raise ZeroSensitivityError(
        f"The {backend} {analysis} sensitivity analysis reported the objective as identically "
        f"zero, while the transient output of the same run shows it swinging "
        f"{spread:.4g} V. The sensitivity analysis produced nothing, so the gradients for "
        f"{listed} are not usable and SPIPE will not return them.{hint}",
        list(parameters), analysis, backend)


# ---------------------------------------------------------------------------------------
# rewriting a device card with a new parameter value
# ---------------------------------------------------------------------------------------

#: Devices whose *first positional* value is the parameter of the same name as the device
#: letter -- ``R1 a b 1k``, ``C1 a b 2f``, ``L1 a b 1u``.
_POSITIONAL = {'r': 'R', 'c': 'C', 'l': 'L'}


def substitute_device_value(line: str, device: str, name: str, value: float) -> Optional[str]:
    """Return *line* with device parameter *name* set to *value*, or ``None`` if it is not there.

    Handles the two spellings a SPICE card uses:

    * ``name=value`` anywhere on the card (``M1 d g s b nch W=16u L=0.5u``);
    * the single positional value of an ``R``/``C``/``L`` card (``R3 a b 1k``).

    The value is written in full ``repr`` precision and with no engineering suffix, which
    every SPICE dialect accepts, so a rewritten card is numerically exactly the tensor.
    """
    body = line.split('#')[0]
    tokens = body.split()
    if not tokens or tokens[0].lower() != device.lower():
        return None

    pattern = re.compile(r'(?i)(?<![A-Za-z0-9_])(' + re.escape(name) + r')\s*=\s*[^\s]+')
    if pattern.search(body):
        return pattern.sub(lambda m: f"{m.group(1)}={value!r}", body, count=1) + '\n'

    letter = tokens[0][0].lower()
    if _POSITIONAL.get(letter) == name.upper() and len(tokens) >= 4:
        tokens[3] = repr(value)
        return ' '.join(tokens) + '\n'
    return None


def rewrite_device_parameters(lines: Sequence[str],
                              values: Dict[DeviceParameter, 'torch.Tensor']) -> List[str]:
    """Apply every entry of *values* to the matching device card in *lines*.

    :raises RuntimeError: naming the parameter, if no card in *lines* carries it.  A
        ``.sensparam`` declaration that silently does nothing would make the gradient wrong
        in a way nothing else would catch.
    """
    out = list(lines)
    for (device, name), tensor in values.items():
        value = float(tensor.detach().reshape(()))
        for index, line in enumerate(out):
            replaced = substitute_device_value(line, device, name, value)
            if replaced is not None:
                out[index] = replaced
                break
        else:
            raise RuntimeError(
                f"'.sensparam {format_key(device, name)}' names a parameter SPIPE cannot "
                f"write back into the netlist: no card in the .electronic section starts "
                f"with {device!r} and carries '{name}=' (or is an R/C/L card whose "
                f"positional value is {name.upper()}). Give the parameter explicitly, e.g. "
                f"'{device} ... {name.lower()}=16u'.")
    return out


def read_device_parameter(lines: Sequence[str], device: str, name: str) -> float:
    """The value device parameter *name* has in *lines*.

    :raises RuntimeError: if the device or the parameter is not found, listing what the
        matching card does carry.
    """
    from spipe.utils import convert

    pattern = re.compile(r'(?i)(?<![A-Za-z0-9_])' + re.escape(name) + r'\s*=\s*([^\s]+)')
    for line in lines:
        body = line.split('#')[0]
        tokens = body.split()
        if not tokens or tokens[0].lower() != device.lower():
            continue
        match = pattern.search(body)
        if match:
            try:
                return convert(match.group(1))
            except ValueError as exc:
                raise RuntimeError(
                    f"'.sensparam {format_key(device, name)}': the card {body.strip()!r} "
                    f"sets {name}={match.group(1)!r}, which SPIPE cannot read as a number "
                    f"({exc}). A '.sensparam' parameter must be a literal value, not an "
                    f"expression.") from None
        letter = tokens[0][0].lower()
        if _POSITIONAL.get(letter) == name.upper() and len(tokens) >= 4:
            return convert(tokens[3])
        raise RuntimeError(
            f"'.sensparam {format_key(device, name)}': the card {body.strip()!r} does not "
            f"set '{name}='. Write the parameter explicitly on the card, e.g. "
            f"'{device} ... {name.lower()}=16u'.")
    raise RuntimeError(
        f"'.sensparam {format_key(device, name)}' names the device {device!r}, which does "
        f"not appear in the .electronic section.")


def warn_adjoint_override(declared: Sequence[DeviceParameter]) -> None:
    """Warn that ``use_adjoint`` is being overridden because device parameters are declared."""
    listed = ', '.join(format_key(d, p) for d, p in declared)
    warnings.warn(
        f"use_adjoint=True was requested, but this netlist declares device sensitivity "
        f"parameters ({listed}). Xyce's transient adjoint returns an identically zero "
        f"sensitivity block for device parameters (measured on 7.10), so SPIPE is forcing "
        f"'.options sensitivity direct=1 adjoint=0' for this deck. The direct method costs "
        f"one extra linear solve per parameter.",
        RuntimeWarning, stacklevel=3)
