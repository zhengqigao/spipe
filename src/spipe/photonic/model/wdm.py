from spipe.photonic.model.base import Device
import torch
from spipe.photonic.func import FreeLightSpeed, neff
from typing import Optional, Tuple, Dict, Any
from spipe import config
import re
from ..func import collect_coeff, taylor

__all__ = ['WDM1to4', 'WDM1to1', 'WDM1to2']


def _check_channels(device: Device) -> int:
    """Validate that the simulation frequency grid matches the number of WDM output ports.

    A ``wdm1toN`` device is a wavelength (de)multiplexer: WDM channel ``k`` is routed to output
    port ``k + 1``.  The channels of the simulation are exactly the frequency points created by
    the ``.freq`` statement, so the number of frequency points *must* equal the number of output
    ports of the device.  Previously the frequency axis was indexed as if it were a port
    selector, which silently dropped light when ``len(omega) > n_ports`` and raised an obscure
    ``IndexError`` when ``len(omega) < n_ports``.

    :return: the (validated) number of channels.
    :raises RuntimeError: if the two do not agree.
    """
    num_channel = len(device.params['omega'])
    num_port = device._num_port[1]
    if num_channel != num_port:
        raise RuntimeError(
            f"Device model '{device._name}' ({device.__class__.__name__}) is a "
            f"1-to-{num_port} wavelength (de)multiplexer: WDM channel k is routed to output "
            f"port k+1, so the simulation must define exactly one frequency per output port. "
            f"Got len(omega) = {num_channel} frequency point(s) but the device has "
            f"{num_port} output port(s). Either change the '.freq' statement to use "
            f"{num_port} frequency points, or use a 'wdm1to{num_channel}' device instead.")
    return num_channel


class WDM1to1(Device):
    """the class definition of a passive WDM device.

    WDM channel ``k`` (i.e. the ``k``-th frequency point of the ``.freq`` grid) is routed to
    output port ``k + 1``.  **The ``.freq`` statement must therefore define exactly 1 frequency
    point**, one per output port; otherwise a ``RuntimeError`` is raised (see
    :func:`_check_channels`).
    """
    _name = 'wdm1to1'
    _required_attr = ['time', 'omega', 'neff', ]
    _optional_attr = {'ng': None, 'wl': None}
    _num_port = [1, 1]
    _active_port = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:

        _check_channels(self)

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


class WDM1to2(Device):
    """the class definition of a passive WDM device.

    WDM channel ``k`` (i.e. the ``k``-th frequency point of the ``.freq`` grid) is routed to
    output port ``k + 1``.  **The ``.freq`` statement must therefore define exactly 2 frequency
    points**, one per output port; otherwise a ``RuntimeError`` is raised (see
    :func:`_check_channels`).
    """
    _name = 'wdm1to2'
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

        _check_channels(self)

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 3, 3),
                                     dtype=config['complex_dtype'], device=config['device'])

        scatter_matrix[..., 0, 1, 0] = 1.0
        scatter_matrix[..., 1, 2, 0] = 1.0

        scatter_matrix[..., 0, 0, 1] = 1.0
        scatter_matrix[..., 1, 0, 2] = 1.0

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad


class WDM1to4(Device):
    """the class definition of a passive WDM device.

    WDM channel ``k`` (i.e. the ``k``-th frequency point of the ``.freq`` grid) is routed to
    output port ``k + 1``.  **The ``.freq`` statement must therefore define exactly 4 frequency
    points**, one per output port; otherwise a ``RuntimeError`` is raised (see
    :func:`_check_channels`).
    """
    _name = 'wdm1to4'
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

        _check_channels(self)

        scatter_matrix = torch.zeros((len(self.params['time']), len(self.params['omega']), 5, 5),
                                     dtype=config['complex_dtype'], device=config['device'])

        scatter_matrix[..., 0, 1, 0] = 1.0
        scatter_matrix[..., 1, 2, 0] = 1.0
        scatter_matrix[..., 2, 3, 0] = 1.0
        scatter_matrix[..., 3, 4, 0] = 1.0

        scatter_matrix[..., 0, 0, 1] = 1.0
        scatter_matrix[..., 1, 0, 2] = 1.0
        scatter_matrix[..., 2, 0, 3] = 1.0
        scatter_matrix[..., 3, 0, 4] = 1.0

        if vari_set is None:
            return scatter_matrix, {}
        else:
            # if possible, provide the gradient of the scatter matrix w.r.t. parameters here;
            # If unknown or hard, simply return an empty dictionary. We will perform automatic differentiation.
            grad = {}
            return scatter_matrix, grad
