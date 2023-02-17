"""
Decomposition of One Variable Functions.
"""

from __future__ import annotations
import torch
import pydec
from torch import Tensor
import functools

from typing import Dict, Tuple, Union, Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..._composition import Composition


from .states import (
    get_decomposition_func,
    get_decomposition_name,
)

from pydec.exception_utils import none_decomposition_func_error

__all__ = [
    "relu",
    "relu_",
    "leaky_relu",
    "leaky_relu_",
    "gelu",
    "tanh",
    "tanh_",
    "sigmoid",
    "sigmoid_",
]


def relu(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        # TODO: inplace arg overwrite
        out = decomposition_func(input=input, func=torch.nn.functional.relu, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def relu_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        # TODO: inplace arg overwrite
        out = decomposition_func(
            input=input, func=torch.nn.functional.relu_, inplace=True, ref=ref
        )
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def leaky_relu(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.nn.functional.leaky_relu, ref=ref
        )
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def leaky_relu_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.nn.functional.leaky_relu, ref=ref, inplace=True
        )
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def gelu(
    input: Composition, approximate: str = "none", *, ref: Optional[Tensor] = None
) -> Composition:
    torch_gelu = functools.partial(torch.nn.functional.gelu, approximate=approximate)
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch_gelu, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def tanh(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.tanh, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def tanh_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.tanh_, inplace=True, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def sigmoid(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.sigmoid, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def sigmoid_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.sigmoid_, inplace=True, ref=ref
        )
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())
