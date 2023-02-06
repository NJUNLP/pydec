import torch
import pydec
from torch.nn import Module
from . import Tracer
from torch.autograd.grad_mode import _DecoratorContextManager

from .library import (
    _is_support_module,
    _is_unsupport_module,
    _is_pydec_module,
    _is_customized_module,
    _pydec_builtin_module_dict,
    _pydec_builtin_api_dict,
    _pydec_functional_dict,
    _customized_api_dict,
    _customized_functional_dict,
    _customized_module_dict,
    _init,
)
from typing import Any


def compile(model: Module) -> Tracer:
    _init()  # the library init should be invoked at runtime to avoid circular import
    # static stage
    _static_compile(model)
    # do runtime stage in runtime context
    return Tracer(model, RuntimeContextManager())


def _static_compile(model: Module) -> None:
    for name, module in model._modules.items():
        if _is_pydec_module(module):
            continue
        # search in customized lib first
        if _is_customized_module(module):
            pydec_module: pydec.nn.DecModule = _customized_module_dict[
                module._get_name()
            ]
            setattr(model, name, pydec_module.convert_from(module))
        elif _is_support_module(module):
            # TODO(bug): Container's submodules aren't compiled
            pydec_module: pydec.nn.DecModule = _pydec_builtin_module_dict[
                module._get_name()
            ]
            setattr(model, name, pydec_module.convert_from(module))
        elif not _is_unsupport_module(module):
            _static_compile(module)
        else:
            raise RuntimeError(
                f"PyDec does not currently support automatic tracking of Module ({module._get_name()})."
            )


class RuntimeContextManager(_DecoratorContextManager):
    r"""Context-manager that convert torch API to pydec API automatically."""

    def __init__(self) -> None:
        super().__init__()
        self.api_prev = {}
        self.functional_prev = {}

    def __enter__(self) -> None:
        torch_api_dict = vars(torch)
        torch_functional_dict = vars(torch.nn.functional)
        for name, func in _pydec_builtin_api_dict.items():
            self.api_prev[name] = torch_api_dict[name]
            setattr(torch, name, func)
        for name, func in _pydec_functional_dict.items():
            self.functional_prev[name] = torch_functional_dict[name]
            setattr(torch.nn.functional, name, func)
        for name, func in _customized_api_dict.items():
            self.api_prev[name] = torch_api_dict[name]
            setattr(torch, name, func)
        for name, func in _customized_functional_dict.items():
            self.functional_prev[name] = torch_functional_dict[name]
            setattr(torch.nn.functional, name, func)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for name, func in self.api_prev.items():
            setattr(torch, name, func)
        for name, func in self.functional_prev.items():
            setattr(torch.nn.functional, name, func)
