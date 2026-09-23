import os
from .waveguide import *
from .modm import *
from .modp import *
from .mzm import *
from .pbum import *
from .pd_array import *
from .ps import *
from .mzi import *
from .splitter import *
from .wdm import *

__all__ = ['_json_path_', '_extra_model_', '_defined_model_']

_json_path_ = os.path.join(os.path.dirname(__file__), 'model.json')
_extra_model_ = dict()
_defined_model_ = ['WaveGuide', 'ModM', "ModP", "PBUm", "PS", "MZI", "Splitter1to4", "WDM1to4", "Splitter1to1", "WDM1to1", "WDM1to2", "Splitter1to2", "Splitter1to3", "MZM"]


def print_info():
    print("_extra_model_:", _extra_model_)
    print("_defined_model_:", _defined_model_)
    print("_json_path_:", _json_path_)
