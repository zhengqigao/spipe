from spipe.photonic.model.base import Device, complex_tensor
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict
from spipe import config

__all__ = ['WaveGuide']


class WaveGuide(Device):
    """The class definition of a waveguide model."""

    _name = 'wg'
    _required_attr = ['time', 'omega', 'l', 'neff']
    _optional_attr = {'ng': None, 'wl': None, 'alpha': 1.0}
    _num_port = [1, 1]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:

        # The scatter_matrix should be (len(time), len(omega), _num_port.sum(), _num_port.sum())
        # For passive device, the time axis doesn't make any difference.
        # Namely, the scatter matrix is broadcasted to the zero-th axis and param['time'] is not used.

        neff_func = neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = self.params['omega'] * neff_func(self.params['omega']) / FreeLightSpeed
        scatter_matrix = ((1.0 if self.params['l'] == 0 else self.params['alpha']) *
                          torch.exp(1.j * beta * self.params['l']).to(config['complex_dtype']).reshape(-1, 1, 1)
                          * complex_tensor([[0.0, 1.0], [1.0, 0.0]]).unsqueeze(0)).unsqueeze(0)

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}

            # grad['alpha'] = (1.0 / self.params['alpha'] * scatter_matrix).to(config['dtype'])

            return scatter_matrix, grad
