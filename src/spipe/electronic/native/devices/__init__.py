"""Device model library for the native SPIPE circuit engine.

Every device type is a :class:`~.base.DeviceGroup` subclass registered in
:data:`REGISTRY` by its SPICE letter.  Adding a device means adding a class
here -- the parser, the MNA assembly and the solvers stay untouched.
"""

from __future__ import annotations

from .base import (DTYPE, Device, DeviceGroup, LoadContext, limexp, pnjlim)
from .passive import Capacitor, Inductor, Resistor
from .sources import CurrentSource, VoltageSource
from .controlled import CCCS, CCVS, VCCS, VCVS
from .diode import Diode
from .mosfet import Mosfet
from .bjt import BJT
from .switch import VCSwitch, CCSwitch

REGISTRY = {
    "r": Resistor,
    "c": Capacitor,
    "l": Inductor,
    "v": VoltageSource,
    "i": CurrentSource,
    "e": VCVS,
    "g": VCCS,
    "f": CCCS,
    "h": CCVS,
    "d": Diode,
    "m": Mosfet,
    "q": BJT,
    "s": VCSwitch,
    "w": CCSwitch,
}

__all__ = [
    "REGISTRY", "DTYPE", "Device", "DeviceGroup", "LoadContext",
    "Resistor", "Capacitor", "Inductor", "VoltageSource", "CurrentSource",
    "VCVS", "VCCS", "CCCS", "CCVS", "Diode", "Mosfet", "BJT",
    "VCSwitch", "CCSwitch", "limexp", "pnjlim",
]
