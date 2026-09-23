from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict
from math import pi
from spipe import config

__all__ = ['MZI']


class MZI(Device):
    """The class definition of a passive MZI.


    """

    _name = 'mzi'
    _required_attr = ['time','omega', 'theta',  'neff', ]
    _optional_attr = {'ng': None, 'wl': None, 'alpha': 1.0, 'l': 0.0,}
    _num_port = [2, 2]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:

        # linearly approximate the effective index neff
        neff_func = neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = (self.params['omega'] * neff_func(self.params['omega']) / FreeLightSpeed)  # propagation constant

        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For passive device, the time axis doesn't make any difference.
        # Namely, the scatter matrix is broadcasted to the zero-th axis and param['time'] is not used.

        S1 = torch.zeros((2, 2), dtype = config['complex_dtype'], device = config['device'])
        S1[0, 0] = S1[1, 1] = 1.0 * torch.cos(self.params['theta'])
        S1[0, 1] = S1[1, 0] = 1.j * torch.sin(self.params['theta'])

        S = torch.zeros((4, 4), dtype = config['complex_dtype'], device = config['device'])
        S[2:, :2] = S1
        S[:2, 2:] = S1

        scatter_wg = (torch.exp(1.j * beta * self.params['l']) * (1.0 if self.params['l'] == 0 else self.params['alpha'])).to(config['complex_dtype'])

        scatter_matrix = (scatter_wg.reshape(-1, 1, 1) * S.unsqueeze(0)).unsqueeze(0)

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
