from typing import Tuple, Union, Dict, List, Any
import re
from math import pi

def extract(source: str, convert_numeric=False) -> Tuple[Union[str, Any], List[str], Dict[Any, Union[float,str]]]:
    '''
    Extract values from a string of the format

    'a b c d e xx=1.0khz yy=1ms zz = 1e-3m aa=1.hz'
    '.aa b c d e xx=1.0khz yy=1.0'
    '.aa xx=1.0khz yy=1e-3m'

    '''
    source = re.sub(r'\s*=\s*', '=', source)
    initial_string, *remain = source.split()
    kv_pair = dict()
    string = []
    for current_string in remain:
        if '=' in current_string:
            key, value = current_string.split('=')
            kv_pair[key] = convert(value) if convert_numeric else value
        else:
            string.append(current_string)

    return initial_string, string, kv_pair


def convert(value_str: str) -> float:
    # Unit suffixes, all lower case: the value is lower-cased before the lookup, so a key with a
    # capital letter ('fF', 'mA', ...) could never match -- those entries used to sit here unused.
    unit_multipliers = {
        '': 1,
        # SPICE scale factors
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12,
        # frequency
        'hz': 1, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9, 'thz': 1e12,
        # length (metres)
        'nm': 1e-9, 'um': 1e-6, 'mm': 1e-3, 'cm': 1e-2,
        # time
        'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1,
        # angle
        'pi': pi,
        # capacitance
        'ff': 1e-15, 'pf': 1e-12, 'nf': 1e-9, 'uf': 1e-6, 'mf': 1e-3,
        # voltage
        'kv': 1e3, 'v': 1.0, 'mv': 1e-3, 'uv': 1e-6, 'nv': 1e-9, 'pv': 1e-12, 'fv': 1e-15,
        'av': 1e-18,
        # current
        'ka': 1e3, 'a': 1, 'ma': 1e-3, 'ua': 1e-6, 'na': 1e-9, 'pa': 1e-12,
        # power
        'kw': 1e3, 'w': 1, 'mw': 1e-3, 'uw': 1e-6, 'nw': 1e-9, 'pw': 1e-12, 'fw': 1e-15,
        'aw': 1e-18,
    }

    # Strip any whitespace and convert to lower case
    value_str = value_str.strip().lower()

    # Regular expression to extract numeric part and unit part.
    # NOTE: the previous pattern was r"([+-]?[\d\.eE-]+)([a-zA-Z]*)", whose character
    # class allowed '-' but not '+', so a number with an explicit positive exponent
    # ('1.93e+14') parsed as '1.93e' and raised. Python's default float formatting
    # emits exactly that form, so any programmatically generated netlist hit it.
    match = re.match(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)", value_str)

    if not match:
        raise ValueError(f"Invalid format: {value_str}")

    numeric_value = float(match.group(1))
    unit_part = match.group(2).strip()

    if unit_part in unit_multipliers:
        return numeric_value * unit_multipliers[unit_part]
    if unit_part == 'm':
        # A bare 'm' is metres in physics and milli in SPICE -- and the .electronic section of the
        # same file is SPICE. Guessing either way is a silent factor of 1000, so refuse.
        raise ValueError(
            f"{value_str!r}: a bare 'm' is ambiguous here (metres, or SPICE's milli?). Write "
            f"the number out ('1e-3'), or use an explicit unit: 'mm', 'um', 'nm' for lengths.")
    raise ValueError(
        f"Unknown unit {unit_part!r} in {value_str!r}. Accepted suffixes: SPICE scale factors "
        f"f p n u k meg g t; and units such as nm um mm cm, fs ps ns us ms s, hz khz mhz ghz thz, "
        f"pi, mw w, ma a (case-insensitive).")
