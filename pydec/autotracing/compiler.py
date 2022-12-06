import torch
import pydec
from pydec import Composition
from torch.nn import Module
from . import Tracer
from torch.autograd.grad_mode import _DecoratorContextManager
from functools import wraps
from ..exception_utils import args_error_not_in_tracing
from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)

_torch_builtin_modules = []
_pydec_builtin_modules = []
_pydec_builtin_module_names = []
_pydec_builtin_api_dict = {}
_pydec_functional_dict = {}


def compile(model: Module) -> Tracer:
    _init()
    # static stage
    _static_compile(model)
    # do runtime stage in runtime context
    return Tracer(model, RuntimeContextManager())


def _static_compile(model: Module) -> None:
    for name, module in model._modules.items():
        if _is_pydec_module(module):
            continue
        if _is_support_module(module):
            pydec_module: pydec.nn.DecModule = vars(pydec.nn)[module._get_name()]
            setattr(model, name, pydec_module.convert_from(module))
        elif not _is_unsupport_module(module):
            _static_compile(module)
        else:
            raise RuntimeError(
                f"PyDec does not currently support automatic tracking of Module ({module._get_name()})."
            )


def _is_input_composition(*args, **kwargs) -> bool:
    for arg in args:
        if isinstance(arg, Composition):
            return True
    for arg in kwargs.values():
        if isinstance(arg, Composition):
            return True
    return False


def _api_wrapper(torch_api: Callable, pydec_api: Callable) -> Callable:
    @wraps(torch_api)
    def warpped_func(*args, **kwargs):
        c_mode = _is_input_composition(*args, **kwargs)
        if c_mode and not pydec.is_tracing_enabled():
            # Can only throw exceptions since not in the runtime compiler context.
            raise args_error_not_in_tracing(*args, **kwargs)
        if c_mode:
            return pydec_api(*args, **kwargs)
        else:
            return torch_api(*args, **kwargs)

    return warpped_func


def _init():
    def _init_builtin_modules():
        _torch_builtin_modules.clear()
        for m in vars(torch.nn).values():
            if isinstance(m, type) and m.__module__.startswith("torch.nn.modules"):
                _torch_builtin_modules.append(m)
        _pydec_builtin_modules.clear()
        for m in vars(pydec.nn).values():
            if isinstance(m, type) and m.__module__.startswith("pydec.nn.modules"):
                _pydec_builtin_modules.append(m)

    def _init_pydec_module_names():
        _pydec_builtin_module_names.clear()
        for m in vars(pydec.nn).values():
            if isinstance(m, type) and m.__module__.startswith("pydec.nn.modules"):
                _pydec_builtin_module_names.append(m.__name__)

    def _init_api_dict():
        _pydec_builtin_api_dict.clear()
        torch_api_dict = vars(torch)
        for name, var in vars(pydec).items():
            if type(var).__name__ == "function":
                if (
                    name.startswith("c_")
                    or name.startswith("_")
                    or name not in torch_api_dict
                    or name == "obj"
                ):
                    continue
                _pydec_builtin_api_dict[name] = _api_wrapper(torch_api_dict[name], var)

        _pydec_functional_dict.clear()
        torch_functional_dict = vars(torch.nn.functional)
        for name, var in vars(pydec.nn.functional).items():
            if type(var).__name__ == "function":
                if (
                    name.startswith("c_")
                    or name.startswith("_")
                    or name not in torch_functional_dict
                    or name == "obj"
                ):
                    continue
                _pydec_functional_dict[name] = _api_wrapper(
                    torch_functional_dict[name], var
                )

    _init_builtin_modules()
    _init_pydec_module_names()
    _init_api_dict()


def _is_support_module(module: Module):
    return module._get_name() in _pydec_builtin_module_names


def _is_unsupport_module(module: Module):
    if type(module) in _torch_builtin_modules:
        return True
    return False


def _is_pydec_module(module: Module):
    return type(module) in _pydec_builtin_modules


class RuntimeContextManager(_DecoratorContextManager):
    r"""Context-manager that convert torch API to pydec API automatically."""

    def __init__(self) -> None:
        super().__init__()
        self.api_prev = {}
        self.functional_prev = {}

    def __enter__(self) -> None:
        torch_api_dict = vars(torch)
        for name, func in _pydec_builtin_api_dict.items():
            self.api_prev[name] = torch_api_dict[name]
            setattr(torch, name, func)
        torch_functional_dict = vars(torch.nn.functional)
        for name, func in _pydec_functional_dict.items():
            self.functional_prev[name] = torch_functional_dict[name]
            setattr(torch.nn.functional, name, func)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for name, func in self.api_prev.items():
            setattr(torch, name, func)
        for name, func in self.functional_prev.items():
            setattr(torch.nn.functional, name, func)
