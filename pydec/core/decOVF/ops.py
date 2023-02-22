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
    "reciprocal",
    "reciprocal_",
    "exp",
    "exp_",
    "biased_exp",
    "sqrt",
    "sqrt_",
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


def tanh(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.tanh, out=out, ref=ref)
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


def reciprocal(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.reciprocal, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def reciprocal_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.reciprocal_, inplace=True, ref=ref
        )
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def exp(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.exp, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def exp_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.exp_, inplace=True, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def biased_exp(
    input: Composition,
    bias: Tensor,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    if ref is None:
        ref = input.c_sum()
    new_input = pydec._from_replce(input.components, input.residual + bias)
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=new_input, func=torch.exp, ref=ref + bias)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def sqrt(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.sqrt, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def sqrt_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.sqrt_, ref=ref)
        assert isinstance(out, pydec.Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())
