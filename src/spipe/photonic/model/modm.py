from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict, Any
from spipe import config
import re
from ..func import collect_coeff, taylor

__all__ = ['ModM']

class ModM(Device):
    """the class definition of a 2-in 2-out active MZM device"""
    _name = 'modm'
    _required_attr = ['time', 'omega', 'act', 'neff', 'an', 'coeff']
    _optional_attr = {'ng': None, 'wl': None, 'act_l': 10e-6, 'wgu_l': 0.0, 'wgl_l': 0.0, 'alpha': 1.0}
    _num_port = [2, 2]
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

        # linearly approximate the effective index neff
        neff_func = neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = self.params['omega'] / FreeLightSpeed * neff_func(self.params['omega'])  # propagation constant

        # calculate the active phase shift induced by param['act'] at different param['omega']
        # ps = 2 * pi * act_l / lambda * (dneff/dp * delta_p) = neff * omega * act_l / c * (dneff/dp * delta_p)
        # We know beta = neff * omega / c
        # We assume (dneff/dp * delta_p) = param['coeff'] * param['act'], i.e., linear changes.
        # We can extend it to more complicated cases

        ps = (beta * self.params['act_l']).unsqueeze(0) * \
             (taylor(self.params['coeff'], self.params['act'])).unsqueeze(1) # % (2 * torch.pi)

        cos_ps, sin_ps = torch.cos(ps), torch.sin(ps)

        s_mzm = torch.zeros((len(self.params['time']), len(self.params['omega']), 2, 2), dtype=config['complex_dtype'], device = config['device'])
        s_mzm[..., 0, 0] = s_mzm[..., 1, 1] = cos_ps
        s_mzm[..., 0, 1] = s_mzm[..., 1, 0] = 1.j * sin_ps

        s_wg = torch.zeros((len(self.params['omega']), 2, 2), dtype=config['complex_dtype'], device = config['device'])
        s_wg[..., 0, 0] = (1.0 if self.params['wgu_l'] == 0 else self.params['alpha']) * torch.exp(1.j * beta * self.params['wgu_l'])
        s_wg[..., 1, 1] = (1.0 if self.params['wgl_l'] == 0 else self.params['alpha']) * torch.exp(1.j * beta * self.params['wgl_l'])

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 4, 4), dtype=config['complex_dtype'], device = config['device'])
        scatter_matrix[..., 2:, :2] = torch.matmul(s_mzm, s_wg)
        scatter_matrix[..., :2, 2:] = torch.matmul(s_wg, s_mzm)

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
