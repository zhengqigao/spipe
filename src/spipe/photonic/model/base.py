import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type, Dict
from abc import abstractmethod, ABCMeta
import json
import warnings
from collections import OrderedDict
import importlib
import os
import torch
import torch.nn as nn
import itertools
from spipe import config
import numpy as np

__all__ = ['Device', 'complex_dtype', 'real_dtype', 'complex_zeros', 'complex_tensor', 'as_complex',
           'real_tensor', 'as_real']

_has_fc = False

try:
    from torch.nn.utils._stateless import functional_call

    _has_fc = True
except ImportError:
    pass

try:
    from torch.nn.utils.stateless import functional_call

    _has_fc = True
except ImportError:
    pass

try:
    from torch.func import functional_call

    _has_fc = True
except ImportError:
    pass


def complex_dtype() -> torch.dtype:
    """The complex dtype currently selected in :data:`spipe.config`.

    Every scatter matrix produced by a :class:`Device` must use this dtype.  Never hard-code
    ``torch.complex64`` / ``torch.complex128`` in a device model -- call this helper (or one of
    :func:`complex_zeros`, :func:`complex_tensor`, :func:`as_complex`) instead, so that the
    simulator honours ``config['complex_dtype']``.
    """
    return config['complex_dtype']


def real_dtype() -> torch.dtype:
    """The real dtype currently selected in :data:`spipe.config`."""
    return config['real_dtype']


def complex_zeros(*shape: Any) -> torch.Tensor:
    """``torch.zeros`` with the configured complex dtype and device."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    return torch.zeros(shape, dtype=config['complex_dtype'], device=config['device'])


def complex_tensor(data: Any) -> torch.Tensor:
    """``torch.tensor`` with the configured complex dtype and device."""
    return torch.tensor(data, dtype=config['complex_dtype'], device=config['device'])


def real_tensor(data: Any) -> torch.Tensor:
    """``torch.tensor`` with the configured *real* dtype and device.

    Counterpart of :func:`complex_tensor`.  Use it for anything that takes part in arithmetic --
    lengths, indices of refraction, phases, coupling angles, drive signals.  A bare
    ``torch.tensor(0.35)`` is float32, so ``torch.cos`` of it carries only ~1e-8 of accuracy no
    matter what ``config['complex_dtype']`` says: the container ends up double precision while the
    *value* in it is single precision.
    """
    return torch.tensor(data, dtype=config['real_dtype'], device=config['device'])


def as_real(tensor: Any) -> torch.Tensor:
    """Cast a real-valued tensor to exactly ``config['real_dtype']`` (and the configured device).

    Integer, boolean and complex tensors are passed through untouched.
    """
    if not isinstance(tensor, torch.Tensor):
        return real_tensor(tensor)
    if tensor.is_floating_point() and tensor.dtype != config['real_dtype']:
        tensor = tensor.to(config['real_dtype'])
    return tensor


def as_complex(tensor: Any) -> torch.Tensor:
    """Cast ``tensor`` to exactly ``config['complex_dtype']`` (and the configured device).

    This is the single choke point used by :meth:`Device.transfer`, so that a model which
    internally computes in a different precision can never leak a wrong dtype into the solver.
    """
    if not isinstance(tensor, torch.Tensor):
        return complex_tensor(tensor)
    if tensor.dtype != config['complex_dtype']:
        tensor = tensor.to(config['complex_dtype'])
    return tensor


def _param_wrap(param: Any) -> Any:
    """Wrap a device attribute as an ``nn.Parameter``.

    Real-valued attributes are materialised at ``config['real_dtype']``.  This is not cosmetic:
    ``torch.tensor(0.35)`` is float32, so ``torch.cos`` / ``torch.exp`` of such a parameter is only
    single-precision accurate, and casting the *result* to complex128 afterwards cannot recover the
    lost digits.  (A lossless MZI-waveguide-MZI chain used to violate energy conservation by
    ~4.4e-08 under ``complex_dtype = complex128`` for exactly this reason.)

    Integer, boolean, str, list, ndarray and ``None`` attributes keep their previous handling.
    """
    if isinstance(param, int):  # bool is a subclass of int; both keep the legacy behaviour
        return torch.nn.Parameter(torch.tensor(param)).to(config['device'])
    elif isinstance(param, float):
        return torch.nn.Parameter(real_tensor(param)).to(config['device'])
    elif isinstance(param, (str, list, np.ndarray)):
        return param
    elif isinstance(param, torch.Tensor):
        # a float32 tensor handed in from outside (drive signals, time grids) is promoted too
        return torch.nn.Parameter(as_real(param)).to(config['device'])
    elif param is None:
        return None
    else:
        raise NotImplementedError(f"The provided parameter type of {param} is not supported.")


#: Attributes that give the length of a device's propagation section, the section `alpha` applies to.
_SECTION_LENGTHS = ('l', 'wg_l', 'wgu_l', 'wgl_l')


def _check_alpha(model: str, kwargs: Dict) -> None:
    """Warn about the two ways `alpha=` silently does not do what it looks like.

    `alpha` is the field transmission of the device's propagation section -- the part of length
    `l` (or `wg_l`, `wgu_l`, `wgl_l`). With every such length zero there is no section, so it has
    no effect: `wg0 a b l=0 alpha=0.5` transmits everything. That is by design (a zero-length
    connection loses nothing), but it used to happen without a word. And `alpha > 1` is gain,
    which a passive device cannot have.
    """
    if 'alpha' not in kwargs:
        return
    try:
        alpha = float(kwargs['alpha'])
    except (TypeError, ValueError):
        return
    if alpha < 0.0:
        raise ValueError(f"{model}: alpha={alpha:g} is negative; alpha is the field transmission of "
                         f"the device's propagation section, between 0 (opaque) and 1 (lossless).")
    if alpha > 1.0:
        warnings.warn(f"{model}: alpha={alpha:g} > 1 is optical gain; alpha is a field "
                      f"transmission and a passive device has alpha <= 1.", stacklevel=4)
    lengths = [k for k in _SECTION_LENGTHS if k in kwargs]
    if alpha != 1.0 and lengths:
        try:
            all_zero = all(float(kwargs[k]) == 0.0 for k in lengths)
        except (TypeError, ValueError):
            all_zero = False
        if all_zero:
            warnings.warn(
                f"{model}: alpha={alpha:g} has no effect because {', '.join(k + '=0' for k in lengths)}. "
                f"alpha is the field transmission of the device's propagation section, and a "
                f"zero-length section has none. For a lossy junction, add a short 'wg' with alpha.",
                stacklevel=4)


class Device(nn.Module):
    _name = ''
    _required_attr = []
    _optional_attr = {}
    _num_port = [None, None]
    _active_port = None

    def __init__(self, **kwargs):
        super(Device, self).__init__()

        # A real check, not an assert: `python -O` strips asserts, which let a misspelled
        # parameter through; and the old message dumped every attribute instead of naming the
        # one that was wrong.
        allowed = set(self._required_attr + list(self._optional_attr.keys()) + ['ln', 'rn'])
        unknown = sorted(set(kwargs.keys()) - allowed)
        if unknown:
            import difflib
            user_facing = sorted(allowed - {'time', 'omega', 'act', 'an', 'ln', 'rn'})
            hints = []
            for key in unknown:
                close = difflib.get_close_matches(key, user_facing, n=1)
                hints.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            raise TypeError(
                f"{getattr(self, '_name', '') or self.__class__.__name__}: unknown parameter "
                f"{', '.join(hints)}. Valid parameters: {', '.join(user_facing)}.")

        # a length the line leaves out takes the model's default -- for mzi that is l=0, and its
        # alpha used to be ignored without the warning wg and pbum give
        _check_alpha(getattr(self, '_name', '') or self.__class__.__name__,
                     {**{k: v for k, v in self._optional_attr.items()
                         if k in _SECTION_LENGTHS and v is not None}, **kwargs})
        # A negative length is a negative delay: in envelope mode the light arrived 100 ps
        # before the modulator switched, and nothing objected.
        for key in _SECTION_LENGTHS + ('act_l',):
            if key in kwargs and kwargs[key] is not None:
                try:
                    value = float(kwargs[key])
                except (TypeError, ValueError):
                    continue
                if not value >= 0.0:
                    raise ValueError(f"{getattr(self, '_name', '') or self.__class__.__name__}: "
                                     f"{key}={value:g}, but a length must be zero or positive.")

        self.params = nn.ParameterDict()

        for attr in self._required_attr:
            if attr in kwargs.keys():
                self.params[attr] = _param_wrap(kwargs[attr])
            else:
                raise RuntimeError("The value of attribute '%s' is not provided when initializing model '%s'."
                                   % (attr, self.__class__.__name__))

        for attr, default_value in self._optional_attr.items():
            self.params[attr] = _param_wrap(kwargs[attr] if attr in kwargs.keys() else default_value)

        if 'ng' not in kwargs.keys() or kwargs['ng'] is None:
            self.params['ng'] = self.params['neff']

        self._check_param_precision()

    def _check_param_precision(self) -> None:
        """Guard the *precision* of the parameters, not just the dtype of the finished matrix.

        :meth:`transfer` casts its result to ``config['complex_dtype']``, which catches a wrong
        container but is blind to arithmetic that ran at a lower precision before the cast.  This
        check closes that hole: every real-valued parameter must already carry
        ``config['real_dtype']`` when the model starts computing with it.
        """
        for name, value in self.params.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() \
                    and value.dtype != config['real_dtype']:
                raise RuntimeError(
                    f"Model '{self.__class__.__name__}' received parameter '{name}' with dtype "
                    f"{value.dtype}, but config['real_dtype'] is {config['real_dtype']}. Real-valued "
                    f"device attributes must be materialised at config['real_dtype'], otherwise the "
                    f"trigonometric/exponential arithmetic that builds the scatter matrix runs at "
                    f"the lower precision and casting the result to config['complex_dtype'] cannot "
                    f"recover it. Build such tensors with spipe.photonic.model.base.real_tensor().")

    def forward(self, vari_set: Optional[set] = None) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError(
            "The method 'transfer' is not implemented in the model '%s'." % self.__class__.__name__)


    def transfer(self, vari_set: Optional[Set] = None) -> Union[torch.Tensor, Tuple]:
        if vari_set is None:
            # as_complex() guarantees the solver always receives exactly config['complex_dtype'],
            # whatever precision the model happened to compute in.
            return as_complex(self.forward()[0])
        else:
            with torch.enable_grad():
                transfer_matrix, grad = self.forward(vari_set)
                auto_deri_set = vari_set - grad.keys()
                auto_grad = self._autodiff(transfer_matrix, auto_deri_set) if auto_deri_set else {}
            return as_complex(transfer_matrix), {**{k: as_complex(v) for k, v in grad.items()},
                                                 **{k: as_complex(v) for k, v in auto_grad.items()}}

    def _autodiff(self, transfer_matrix: torch.Tensor, auto_deri_set: Set) -> Dict:
        # Automatic differentiation based on:
        #   https://github.com/pytorch/pytorch/issues/49171
        #   https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
        # A naive implementation is using three for-loops (i,j,k) to iterate through transfer_matrix:
        # ... torch.autograd.grad(transfer_matrix[i, j, k].real, self.params[vari], create_graph=True)[0] ...
        # However, this is very inefficient compared to the following vectorized implementation.
        # Note that for efficient matrix multiplication outside this function in our simulator, we chose a slight
        # different gradient layout than Pytorch.

        if _has_fc:
            auto_deri_list = list(auto_deri_set)

            tuple_param = tuple([v for k, v in self.params.items() if k in auto_deri_list])
            param_name = list(n for n, _ in self.named_parameters() if n[len('params.'):] in auto_deri_list)

            def transfer_wrap(*params):
                out, _ = functional_call(self, {n: p for n, p in zip(param_name, params)}, None)
                return out.real, out.imag

            jac_real_list, jac_imag_list = torch.autograd.functional.jacobian(transfer_wrap, tuple_param,
                                                                              strategy="forward-mode", vectorize=True)

            auto_grad = dict()
            for i in range(len(auto_deri_list)):
                per_ind = list(range(transfer_matrix.ndim, jac_real_list[i].ndim)) + list(range(transfer_matrix.ndim))
                auto_grad[auto_deri_list[i]] = (jac_real_list[i] + 1.j * jac_imag_list[i]).permute(per_ind)

        else:

            auto_grad = {}
            for vari in auto_deri_set:
                jac_tensor = complex_zeros((*self.params[vari].shape, *transfer_matrix.shape))
                indices = itertools.product(*(range(dim) for dim in transfer_matrix.shape))

                for index in indices:
                    # Use the index to get the real and imaginary parts of the transfer_matrix
                    jac_real = torch.autograd.grad(transfer_matrix[index].real, self.params[vari], create_graph=True)[0]
                    jac_imag = torch.autograd.grad(transfer_matrix[index].imag, self.params[vari], create_graph=True)[0]

                    # Combine real and imaginary parts
                    jac = jac_real + 1.j * jac_imag

                    # Assign the result to the appropriate location in the jac_matrix
                    jac_index = (slice(None),) * len(self.params[vari].shape) + index
                    jac_tensor[jac_index] = jac

                auto_grad[vari] = jac_tensor
        return auto_grad

    @classmethod
    def _collect_info(cls) -> dict:

        info_dict = OrderedDict()
        info_dict['model_name'] = cls._name
        info_dict['class_name'] = cls.__name__
        info_dict['required_attr'] = [attr for attr in cls._required_attr]
        info_dict['optional_attr'] = [attr for attr in cls._optional_attr]
        info_dict['num_port'] = cls._num_port
        info_dict['active_port'] = cls._active_port
        return info_dict
