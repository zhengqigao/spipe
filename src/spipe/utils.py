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
    # Define SI unit multipliers
    unit_multipliers = {
        '': 1,
        'hz': 1,
        'khz': 1e3,
        'mhz': 1e6,
        'ghz': 1e9,
        'thz': 1e12,


        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'cm': 1e-2,
        'm': 1,

        'fs': 1e-15,
        'ps': 1e-12,
        'ns': 1e-9,
        'us': 1e-6,
        'ms': 1e-3,
        's': 1,

        'p':1e-12,
        'u': 1e-6,
        'pi': pi,

        'fF':1e-15,
        'f':1e-15,
        'pF': 1e-12,
        'nF': 1e-9,
        'uF': 1e-6,
        'mF': 1e-3,
        'F': 1,


        'kv': 1e3,
        'v': 1.0,
        'mv': 1e-3,
        'uv': 1e-6,
        'nv': 1e-9,
        'av': 1e-10,
        'pv': 1e-12,
        'fv':1e-15,

        'kA': 1e3,
        'A': 1,
        'mA': 1e-3,
        'uA': 1e-6,
        'nA': 1e-9,
        'pA': 1e-12,

        'kw': 1e3,
        'w': 1,
        'mw': 1e-3,
        'uw': 1e-6,
        'nw': 1e-9,
        'pw': 1e-12,
        'fw': 1e-15,
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
    else:
        raise ValueError(f"Unknown unit: {unit_part} in {value_str}")
