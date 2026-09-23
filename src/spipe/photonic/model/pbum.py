from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict
from math import pi
from spipe import config

__all__ = ['PBUm']


class PBUm(Device):
    """The class definition of a passive basic unit (pbu) model with phase shifts in the middle.

    This device model corresponds to the passive version of the tunable basic unit (tbu) used frequently in programmable
    photonics.
    """

    _name = 'pbum'
    _required_attr = ['time','omega', 'theta', 'phi', 'l', 'neff']
    _optional_attr = {'ng': None, 'wl': None, 'alpha': 1.0, 'cp_left': 0.25 * pi, 'cp_right': 0.25 * pi}
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
        S1[0, 0] = S1[1, 1] = 1.0 * torch.cos(self.params['cp_left'])
        S1[0, 1] = S1[1, 0] = 1.j * torch.sin(self.params['cp_left'])

        S2 = torch.zeros((2, 2), dtype = config['complex_dtype'], device = config['device'])
        S2[0, 0] = torch.exp(1.j * self.params['theta'])
        S2[1, 1] = torch.exp(1.j * self.params['phi'])

        S3 = torch.zeros((2, 2), dtype = config['complex_dtype'], device = config['device'])
        S3[0, 0] = S3[1, 1] = 1.0 * torch.cos(self.params['cp_right'])
        S3[0, 1] = S3[1, 0] = 1.j * torch.sin(self.params['cp_right'])

        S = torch.zeros((4, 4), dtype = config['complex_dtype'], device = config['device'])
        S[2:, :2] = S3 @ S2 @ S1
        S[:2, 2:] = S1 @ S2 @ S3

        scatter_wg = (torch.exp(1.j * beta * self.params['l']) * (1.0 if self.params['l'] == 0 else self.params['alpha']))

        scatter_matrix = (scatter_wg.reshape(-1, 1, 1) * S.unsqueeze(0)).unsqueeze(0).to(config['complex_dtype'])

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
