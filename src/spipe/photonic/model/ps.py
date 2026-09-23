from spipe.photonic.model.base import Device
import torch
from typing import Optional, Tuple, Dict
from spipe import config

__all__ = ['PS']


class PS(Device):
    """The class definition of a phase shifter."""

    _name = 'ps'
    _required_attr = ['time', 'omega', 'ps']
    _optional_attr = {'neff': None, 'ng': None, 'wl': None, }
    _num_port = [1, 1]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 2, 2), dtype=config['complex_dtype'], device = config['device'])
        scatter_matrix[:, :, 1, 0] = scatter_matrix[:, :, 0, 1] = torch.exp(1.j * self.params['ps'])

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}

            return scatter_matrix, grad
