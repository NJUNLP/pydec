import functools
import inspect

from torch.autograd.grad_mode import _DecoratorContextManager

from typing import Dict, Tuple, Union, Any, Callable, Optional, TYPE_CHECKING

from ..composition import Composition

from torch.types import (
    _int,
    _float,
    _bool,
    Number,
    _dtype,
    _device,
    _qscheme,
    _size,
    _layout,
    SymInt,
)

_DECOMPOSITION_FUNC_REGISTRY = {}


class _DecompositionState:
    decomposition_name: str = None
    decomposition_args: Dict[str, Any] = {}


def register_decomposition_func(name: str):
    """
    TODO: docs need update
    New decomposition_func can be added with the :func:`register_decomposition_func`
    function decorator.

    For example::

        @register_model('forward_norm_decomposition')
        def abs_decomposition(input: Composition, func: Callable[[Tensor], Tensor], *, ref: Optional[Tensor] = None, inplace: _bool = False,):
            (...)

    Args:
        name (str): the name of the funcion
    """

    def register_func(func):
        if name in _DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        # if name == "none":
        #     raise ValueError(
        #         'Cannot register function ({}), the name "none" is reserved.'.format(
        #             name
        #         )
        #     )

        @functools.wraps(func)
        def warp_decomposition_func(*args, **kwargs):
            dec_args = _DecompositionState.decomposition_args.copy()
            dec_args.update(kwargs)
            argspec = inspect.getfullargspec(func)
            if argspec.varkw is None:
                ignore_keys = []
                for key in dec_args.keys():
                    if key not in (argspec.args + argspec.kwonlyargs):
                        ignore_keys.append(key)
                for key in ignore_keys:
                    dec_args.pop(key)
            return func(*args, **dec_args)

        _DECOMPOSITION_FUNC_REGISTRY[name] = warp_decomposition_func

        return warp_decomposition_func

    return register_func


def set_decomposition_func(name: str) -> None:
    if name not in _DECOMPOSITION_FUNC_REGISTRY:
        raise ValueError("decomposition function ({}) is not registered".format(name))
    _DecompositionState.decomposition_name = name


def get_decomposition_name() -> str:
    return _DecompositionState.decomposition_name


def get_decomposition_func() -> Callable[..., Composition]:
    if _DecompositionState.decomposition_name not in _DECOMPOSITION_FUNC_REGISTRY:
        return None
    else:
        current_decomposition_func = _DECOMPOSITION_FUNC_REGISTRY[
            _DecompositionState.decomposition_name
        ]
        return current_decomposition_func


class using_decomposition_func(_DecoratorContextManager):
    def __init__(self, name: str) -> None:
        if name not in _DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError(
                "Decomposition function ({}) is not registered".format(name)
            )
        self.prev = None
        self.using_name = name

    def __enter__(self):
        self.prev = _DecompositionState.decomposition_name
        _DecompositionState.decomposition_name = self.using_name

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _DecompositionState.decomposition_name = self.prev

    def clone(self):
        return self.__class__(self.using_name)


class no_decomposition(_DecoratorContextManager):
    def __init__(
        self,
    ) -> None:
        self.prev = None

    def __enter__(self):
        self.prev = _DecompositionState.decomposition_name
        _DecompositionState.decomposition_name = "none"

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _DecompositionState.decomposition_name = self.prev


def set_decomposition_args(update: _bool = True, **kwargs) -> None:
    if update:
        _DecompositionState.decomposition_args.update(kwargs)
    else:
        _DecompositionState.decomposition_args = kwargs


def get_decomposition_args() -> Dict[str, Any]:
    return _DecompositionState.decomposition_args


class using_decomposition_args(_DecoratorContextManager):
    def __init__(self, update: _bool = True, **kwargs) -> None:
        self.update = update
        self.prev = None
        self.using_args = kwargs

    def __enter__(self):
        self.prev = _DecompositionState.decomposition_args.copy()
        if self.update:
            _DecompositionState.decomposition_args.update(self.using_args)
        else:
            _DecompositionState.decomposition_args = self.using_args

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _DecompositionState.decomposition_args = self.prev

    def clone(self):
        return self.__class__(self.update, **self.using_args)
