import warnings
from collections import defaultdict
from typing import Optional, Callable, Dict, Any, List
import torch
from math import pi
import re
from spipe import config


FreeLightSpeed = 299792458.0


# def omega2freq(omega: torch.Tensor) -> torch.Tensor:
#     return omega / (2 * pi)
#
#
# def freq2omega(freq: torch.Tensor) -> torch.Tensor:
#     return freq * (2 * pi)
#
#
# def freq2wavelength(freq: torch.Tensor, neff: Optional[float] = 1.0) -> torch.Tensor:
#     return FreeLightSpeed / neff / freq
#
#
# def wavelength2freq(wavelength: torch.Tensor, neff: Optional[float] = 1.0) -> torch.Tensor:
#     return FreeLightSpeed / neff / wavelength


def neff(neff0: float, ng0: Optional[float] = None, wl0: Optional[float] = None) -> Callable:
    if wl0 == None or ng0 == None:  # If ng is not provided, neff is a constant independent of frequency
        neff_func = lambda omega: neff0 * torch.ones_like(omega)
    else:  # If ng is provided, neff is frequency dependent
        def neff_func(omega):
            wl_e = 2 * pi * FreeLightSpeed / omega
            return wl_e / wl0 * (neff0 - ng0) + ng0
    return neff_func


def taylor(coeff: List, x: torch.Tensor, x0: Optional[float] = 0) -> torch.Tensor:
    """
    Calculate the Taylor series expansion of a function at a given point x0.
    """
    delta_x = x - x0
    y = sum(c * delta_x ** i for i, c in enumerate(coeff))
    return y


def require_index_coeffs(kwargs: Dict, model: str) -> None:
    """Refuse a modulator line that gives no ``coeff0=, coeff1=, ...``.

    Those coefficients are the modulator's whole response. Without them ``collect_coeff``
    returns ``[0]``, the phase shift is identically zero, and the device silently never
    modulates -- a plain ``coeff=1e-4`` (no index) was read as exactly that.
    """
    if any(re.fullmatch(r'coeff\d+', str(k)) for k in kwargs):
        return
    hint = (" A plain 'coeff=' is not read: number them, e.g. 'coeff1=1e-4' for a term linear "
            "in the drive." if 'coeff' in kwargs else '')
    raise RuntimeError(
        f"{model}: no index-change coefficients given. Write them as coeff0=, coeff1=, "
        f"coeff2=, ... -- the relative effective-index change is "
        f"dn/neff = coeff0 + coeff1*V + coeff2*V^2 + ...; see docs/netlist.md.{hint}")


def collect_coeff(coeff_dict: Dict, key_string: Optional[str] = '', delete: Optional[bool] = False) -> List:
    if key_string:
        pattern = fr'^{key_string}(\d+)'
    else:
        pattern = r'^[A-Za-z]+(\d+)'

    max_term = 0
    term_map = defaultdict(int)
    delete_list = []
    for k, v in coeff_dict.items():
        match = re.match(pattern, k)
        if match is not None:
            max_term = max(max_term, int(match.group(1)))
            term_map[int(match.group(1))] = v
            delete_list.append(k)

    if len(term_map.keys()) == 0:
        warnings.warn(f"Given key string `{key_string}`, no coefficient detected from {coeff_dict.items()}.")

    res = [0] * (max_term + 1)
    for k, v in term_map.items():
        res[k] = v

    if delete:
        for k in delete_list:
            del coeff_dict[k]

    return res