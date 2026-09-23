from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict, Any
from spipe import config
import re
from ..func import collect_coeff, taylor

__all__ = ['ModP']


class ModP(Device):
    """the class definition of a 1-in 1-out Phase shifter device"""
    _name = 'modp'
    _required_attr = ['time','omega', 'act', 'neff', 'an', 'coeff', 'act_l']
    _optional_attr = {'ng': None, 'wl': None, 'wg_l': 0.0, 'alpha': 1.0}
    _num_port = [1, 1]
    _active_port = 1

    def __init__(self, **kwargs):
        kwargs['coeff'] = collect_coeff(kwargs, 'coeff', delete=True)
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For active device like modulator, the time axis makes a difference.
        # specifically, our key assumption is 'quasi steady state'.
        # We will calculate a scatter matrix for a specific time point:
        # scatter_matrix[i] depends only on param['time'][i]

        # linearly approximate the effective index neff at any omega
        neff_func = neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = (self.params['omega'] * neff_func(self.params['omega']) / FreeLightSpeed)  # propagation constant

        # calculate the phase shift induced by the passive waveguide if included
        # By default, this value will be zero, corresponding to a Mod-phase shifter model without considering wg.
        transfer_wg = (1.0 if self.params['wg_l'] == 0 else self.params['alpha']) * torch.exp(1.j * beta * self.params['wg_l']).to(config['complex_dtype']).reshape(1, -1, 1, 1)

        # calculate the active phase shift induced by param['act'] at different param['omega']
        # ps = 2 * pi * act_l / lambda * (dneff/dv * delta_v) = neff * omega * act_l / c * (dneff/dv * delta_v)
        # We know beta = neff * omega / c
        # We assume (dneff/dv * delta_v) = param['coeff'] * param['act'], i.e., linear changes.
        # We can extend it to higher-order changes.

        ps = (beta * self.params['act_l']).unsqueeze(0) * \
              (taylor(self.params['coeff'], self.params['act'])).unsqueeze(1) # % (2 * torch.pi)

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 2, 2), dtype=config['complex_dtype'], device = config['device'])
        scatter_matrix[:, :, 0, 1] = scatter_matrix[:, :, 1, 0] = torch.exp(1.j * ps)
        scatter_matrix = scatter_matrix * transfer_wg

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
