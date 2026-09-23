from typing import Type
import inspect
import json
import importlib
from spipe.photonic.model.base import Device
from .model import _defined_model_, _json_path_, _extra_model_

__all__ = ['register', 'reset']

_model_module_name_ = "spipe.photonic.model"


def _builtin_classes() -> dict:
    """Map ``_name`` -> class for every model shipped with SPIPE."""
    module = importlib.import_module(_model_module_name_)
    return {getattr(module, class_name)._name: getattr(module, class_name) for class_name in _defined_model_}


def _dump(info: dict) -> None:
    with open(_json_path_, 'w') as f:
        json.dump(info, f, indent=4, separators=(',', ':'))


def register(model: Type[Device]) -> None:
    """Register a user-defined model to the model.json file.

    Since the provided netlist will be parsed according to the model.json file. This function will correctly register
    the user-defined circuit model, so that during parsing, it could be recognized.

    The registration takes effect immediately: a netlist parsed after this call may use the new
    model.  :func:`reset` undoes every registration made so far.

    :param model: The class (e.g., WaveGuide, PBUm) of the user-defined model.
    :raises RuntimeError: if ``model`` is not a :class:`Device` subclass, if ``model._name`` is
        empty or not lower case, or if ``model._name`` collides with an already defined model.
    """
    if not (inspect.isclass(model) and issubclass(model, Device)):
        raise RuntimeError("register() expects a subclass of spipe.photonic.model.base.Device, "
                           "but got %r." % (model,))

    if not model._name:
        raise RuntimeError("The user-defined model '%s' has an empty _name. Please give it a "
                           "lower-case name used as the netlist prefix." % (model.__name__,))

    if not model._name.islower():
        raise RuntimeError("The user-defined model has _name = '%s'. Please use lower case." % (model._name))

    # check if the class name or the _name is used by other already defined models
    defined_classes = {**_builtin_classes(), **_extra_model_}

    for defined_class in defined_classes.values():
        if model._name.lower().startswith(defined_class._name.lower()) or \
                defined_class._name.lower().startswith(model._name.lower()):
            raise RuntimeError("The model name '%s' is already used in '%s'." % (model._name, defined_class._name))

    with open(_json_path_, 'r') as f:
        info = json.load(f)

    info[model._name] = model._collect_info()
    _dump(info)

    # NOTE: mutate the dict in place; `spipe.photonic.photonic` holds a reference to this very
    # object, so rebinding the name here would silently disconnect the two.
    _extra_model_[model._name] = model


def reset() -> None:
    """Reset the model.json file and clear _extra_model_"""
    info = {}
    for class_ in _builtin_classes().values():
        info[class_._name] = class_._collect_info()

    _dump(info)

    # in-place, for the same reason as in register()
    _extra_model_.clear()
