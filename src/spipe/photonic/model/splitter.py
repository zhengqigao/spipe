from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict, Any
from spipe import config
import re
from ..func import collect_coeff, taylor

__all__ = ['Splitter1to4', 'Splitter1to1', 'Splitter1to2', 'Splitter1to3']

class Splitter1to1(Device):
    """the class definition of a 1-to-1 passive splitter device"""
    _name = 'splitter1to1'
    _required_attr = ['time', 'omega', 'neff', ]
    _optional_attr = {'ng': None, 'wl': None}
    _num_port = [1, 1]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 2, 2),
                                     dtype=config['complex_dtype'], device=config['device'])
        scatter_matrix[...,0,1] = 1.0
        scatter_matrix[...,1,0] = 1.0

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad

class Splitter1to2(Device):
    """the class definition of a 1-to-2 passive splitter device"""
    _name = 'splitter1to2'
    _required_attr = ['time', 'omega', 'neff', ]
    _optional_attr = {'ng': None, 'wl': None}
    _num_port = [1, 2]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For active device like modulator, the time axis makes a difference.
        # specifically, our key assumption is 'quasi steady state'.
        # We will calculate a scatter matrix for a specific time point:
        # scatter_matrix[i] depends only on param['time'][i]

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 3, 3),
                                     dtype=config['complex_dtype'], device=config['device'])
        scatter_matrix[..., 0, 1:] = 0.5 ** 0.5
        scatter_matrix[..., 1:, 0] = 0.5 ** 0.5

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad

class Splitter1to3(Device):
    """the class definition of a 1-to-3 passive splitter device"""
    _name = 'splitter1to3'
    _required_attr = ['time', 'omega', 'neff', ]
    _optional_attr = {'ng': None, 'wl': None}
    _num_port = [1, 3]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For active device like modulator, the time axis makes a difference.
        # specifically, our key assumption is 'quasi steady state'.
        # We will calculate a scatter matrix for a specific time point:
        # scatter_matrix[i] depends only on param['time'][i]

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 4, 4),
                                     dtype=config['complex_dtype'], device=config['device'])
        scatter_matrix[..., 0, 1:] = (1 / self._num_port[1]) ** 0.5
        scatter_matrix[..., 1:, 0] = (1/ self._num_port[1]) ** 0.5

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad

class Splitter1to4(Device):
    """the class definition of a 1-to-4 passive splitter device"""
    _name = 'splitter1to4'
    _required_attr = ['time', 'omega', 'neff', ]
    _optional_attr = {'ng': None, 'wl': None}
    _num_port = [1, 4]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For active device like modulator, the time axis makes a difference.
        # specifically, our key assumption is 'quasi steady state'.
        # We will calculate a scatter matrix for a specific time point:
        # scatter_matrix[i] depends only on param['time'][i]

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 5, 5),
                                     dtype=config['complex_dtype'], device=config['device'])
        scatter_matrix[..., 0, 1:] = 0.25 ** 0.5
        scatter_matrix[..., 1:, 0] = 0.25 ** 0.5

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
