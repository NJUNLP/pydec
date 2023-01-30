import torch
import pydec
from pydec import Composition
from functools import wraps
from ..exception_utils import args_error_not_in_tracing
from torch.nn import Module
from typing import Callable


_torch_builtin_modules = []
_pydec_builtin_module_dict = {}
_pydec_builtin_api_dict = {}
_pydec_functional_dict = {}
_customized_api_dict = {}
_customized_functional_dict = {}
_customized_module_dict = {}


def _init() -> None:
    # only initialize once
    if len(_torch_builtin_modules) != 0:
        return

    # do initialization
    def _init_builtin_modules():
        _torch_builtin_modules.clear()
        for m in vars(torch.nn).values():
            if isinstance(m, type) and m.__module__.startswith("torch.nn.modules"):
                _torch_builtin_modules.append(m)

    def _init_pydec_module_dict():
        _pydec_builtin_module_dict.clear()
        for name, var in vars(pydec.nn).items():
            if isinstance(var, type) and var.__module__.startswith("pydec.nn.modules"):
                _pydec_builtin_module_dict[name] = var

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
    _init_pydec_module_dict()
    _init_api_dict()


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


def register_api(name: str) -> None:
    def register_api_with_name(func):
        torch_api_dict = vars(torch)
        if name in _customized_api_dict:
            import warnings

            warnings.warn(
                "override registered function ({})".format(name), stacklevel=2
            )

        _customized_api_dict[name] = _api_wrapper(torch_api_dict[name], func)
        return func

    return register_api_with_name


def register_functional_api(name: str) -> None:
    def register_functional_api_with_name(func):
        torch_functional_dict = vars(torch.nn.functional)
        if name in _customized_functional_dict:
            import warnings

            warnings.warn(
                "override registered function ({})".format(name), stacklevel=2
            )

        _customized_functional_dict[name] = _api_wrapper(
            torch_functional_dict[name], func
        )
        return func

    return register_functional_api_with_name


def register_module(name: str) -> None:
    def register_module_with_name(module: pydec.nn.DecModule):
        if name in _customized_module_dict:
            import warnings

            warnings.warn("override registered module ({})".format(name), stacklevel=2)

        _customized_module_dict[name] = module
        return module

    return register_module_with_name


def register_cmethod(name: str) -> None:
    def register_cmethod_with_name(func: Callable):
        pydec.composition._c_register_method(name, func)

    return register_cmethod_with_name


def _is_support_module(module: Module) -> bool:
    return module._get_name() in _pydec_builtin_module_dict


def _is_unsupport_module(module: Module) -> bool:
    if type(module) in _torch_builtin_modules:
        return True
    return False


def _is_pydec_module(module: Module) -> bool:
    return type(module) in _pydec_builtin_module_dict.values()


def _is_customized_api() -> bool:
    pass


def _is_customized_module(module: Module) -> bool:
    return module._get_name() in _customized_module_dict
