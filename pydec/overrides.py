import torch
import pydec
import warnings
import torch.overrides as overrides
import functools
from typing import (
    Any,
    Dict,
    Union,
    List,
    Tuple,
    Sequence,
    Optional,
    Callable,
    TypeVar,
    overload,
)

T = TypeVar("T")

_HANDLED_BUILTIN_FUNCTIONS = {}
_HANDLED_CUSTOMIZED_FUNCTIONS = {}
_HANDLED_COMPOSITION_METHODS = {}


_namespaces_map = {
    "pydec._composition": torch.Tensor,
    "pydec.variable_functions": torch,
    "pydec.nn": torch.nn,
    "pydec.nn.functional": torch.nn.functional,
}

"""
Functions that are not in the list of overridable functions, but can still
be dispatched via `__torch_function__`.
"""
_whitelist_torch_functions = [
    torch.detach_,
    torch.round_,
    torch.abs_,
    torch.sigmoid_,
    torch.tanh_,
    torch.nn.functional.relu_,
    torch.nn.functional.leaky_relu_,
]


def _register_function(torch_function: T, builtin: bool = False) -> Callable[[T], T]:
    # checking
    if (
        torch_function not in overrides.get_testing_overrides()
        and torch_function not in _whitelist_torch_functions
    ):
        msg = (
            f"function {torch_function} is not overridable."
            " Invoke `torch.overrides.get_overridable_functions()` to see all overridable torch functions."
        )
        raise TypeError(msg)

    def _decorator(func: Callable):
        if builtin:
            _HANDLED_BUILTIN_FUNCTIONS[torch_function] = func
        else:
            overridable_functions = overrides.get_overridable_functions()
            if torch_function in overridable_functions[torch.Tensor]:
                _HANDLED_COMPOSITION_METHODS[torch_function.__name__] = func
                if hasattr(pydec.Composition, torch_function.__name__):
                    warnings.warn(
                        f"pydec.Composition already has built-in method ({torch_function.__name__}), and will be overrided by {func}.",
                        stacklevel=2,
                    )
            # else:
            # pydec_modules = [pydec, pydec.nn, pydec.nn.functional]
            # module_name = ".".join(
            #     ["pydec"] + overrides.resolve_name(torch_function).split(".")[1:-1]
            # )
            # if module_name in [m.__name__ for m in pydec_modules]:
            #     for module in pydec_modules:
            #         if module.__name__ == module_name:
            #             setattr(module, torch_function.__name__, func)
            #             break
            # else:
            #     warnings.warn(
            #         f"function {torch_function} will not append to pydec cause the module {module_name} doesn't exist now.",
            #         stacklevel=2,
            #     )
            _HANDLED_CUSTOMIZED_FUNCTIONS[torch_function] = func
        return func

    return _decorator


def _register_builtin_function(torch_function: T) -> Callable[[T], T]:
    return _register_function(torch_function, builtin=True)


def register_torch_function(torch_function: T) -> Callable[[T], T]:
    """Register a torch function override for Composition"""
    return _register_function(torch_function, builtin=False)


def _auto_registration(func: T) -> T:
    assert hasattr(func, "__module__")
    assert func.__module__ in _namespaces_map
    torch_namespace = _namespaces_map[func.__module__]
    if not hasattr(torch_namespace, func.__name__):
        msg = (
            f"function ({func.__name__}) is not overridable."
            " Invoke `torch.overrides.get_overridable_functions()` to see all overridable torch functions."
        )
        raise TypeError(msg)
    torch_function = getattr(torch_namespace, func.__name__)

    return _register_function(torch_function, builtin=True)(func)


def _hadle_self_is_tensor(func: T, inplace=False) -> T:
    # TODO: inplace warning

    @functools.wraps(func)
    def wrapped_func(self, *args, **kwargs):
        if isinstance(self, torch.Tensor):
            composition_index = None
            for i in range(len(args)):
                if isinstance(args[i], pydec.Composition):
                    composition_index = i
                    break
            if composition_index is not None:
                new_args = list(args)
                self, new_args[composition_index] = new_args[composition_index], self
                return func(self, *new_args, **kwargs)
            composition_key = None
            for k, v in kwargs.items():
                if isinstance(v, pydec.Composition):
                    composition_key = k
                    break
            if composition_key is not None:
                self, kwargs[composition_key] = kwargs[composition_key], self
                return func(self, *args, **kwargs)
            else:
                raise RuntimeError(
                    f"there is no composition arguments for the invoking of function {func}."
                )
        else:
            return func(self, *args, **kwargs)

    return wrapped_func


def _hadle_self_is_tensor_inplace(func: T) -> T:
    def wrapped_hadle_self_is_tensor(func: Callable):
        return _hadle_self_is_tensor(func, inplace=True)

    return wrapped_hadle_self_is_tensor(func)


def is_registered(torch_function: Callable) -> bool:
    return (
        torch_function in _HANDLED_BUILTIN_FUNCTIONS
        or torch_function in _HANDLED_CUSTOMIZED_FUNCTIONS
    )


def get_customized_composition_methods() -> Dict[str, Callable]:
    return _HANDLED_COMPOSITION_METHODS


def dispatch_torch_function(torch_function: Callable) -> Callable:
    # Customized functions have higher priority
    if torch_function in _HANDLED_CUSTOMIZED_FUNCTIONS:
        return _HANDLED_CUSTOMIZED_FUNCTIONS[torch_function]
    elif torch_function in _HANDLED_BUILTIN_FUNCTIONS:
        return _HANDLED_BUILTIN_FUNCTIONS[torch_function]
    else:
        return None