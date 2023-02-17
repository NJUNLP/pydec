"""
Decomposition of Multi-Variable Functions.
"""

from __future__ import annotations
import torch
import pydec
from torch import Tensor
import functools

from ..utils import parse_args

from typing import Dict, Tuple, Union, Any, Callable, Optional, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from .._composition import Composition

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

from pydec.exception_utils import args_error


def _decomposeMVF_grad(
    input: Composition, out_cbody: Tensor, out_residual: Tensor, grad: Tensor
) -> Composition:
    if (
        not isinstance(input, Composition)
        or not isinstance(out_cbody, Tensor)
        or not isinstance(out_residual, Tensor)
        or not isinstance(grad, Tensor)
    ):
        raise args_error(decomposeMVF.__name__, input, out_cbody, out_residual, grad)
    out_components = torch.matmul(input.components, grad.transpose(-1, -2))
    multiplier = out_cbody / out_components.sum(dim=0)
    out_components *= multiplier
    return pydec._from_replce(out_components, out_residual)


def _decomposeMVF_func(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    ref: Optional[Tensor] = None,
    grad: Optional[Tensor] = None,
):
    if (
        not isinstance(input, Composition)
        or (ref is not None and not isinstance(ref, Tensor))
        or (grad is not None and not isinstance(grad, Tensor))
    ):
        raise args_error(decomposeMVF.__name__, input, func, ref, grad)
    if ref is None:
        ref = input.c_sum()
    if grad is not None:
        with torch.no_grad():
            c_out = func(ref)
            out_residual = func(input.residual)
            out_cbody = c_out - out_residual
            out_dim = c_out.size(-1)
            in_dim = input.size(-1)
            if grad.dim() < 2 or grad.size()[-2:] != (out_dim, in_dim):
                raise RuntimeError  # TODO: add msg
    else:
        input_grad_holder = ref.clone().detach_()
        input_grad_holder.requires_grad_(True)
        with torch.enable_grad():
            c_out = func(input_grad_holder)
            grad = []
            for i in range(c_out.size(-1)):
                ...
            # loss = c_out.sum(dim=0,...,-2)
            # TODO
    return _decomposeMVF_grad(input, out_cbody, out_residual, grad, inplace=inplace)


@overload
def decomposeMVF(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    ref: Optional[Tensor] = None,
    grad: Optional[Tensor] = None,
) -> Composition:
    ...


@overload
def decomposeMVF(
    input: Composition, out_cbody: Tensor, out_residual: Tensor, grad: Tensor
) -> Composition:
    ...


def decomposeMVF(*args, **kwargs) -> Composition:
    mode = "func"
    if len(args) > 1:
        if isinstance(args[1], Tensor):
            mode = "grad"
    elif "cbody" in kwargs:
        mode = "grad"
    if mode == "func":
        parse_args(args, ["input", "func", "ref", "grad"], kwargs)
        return _decomposeMVF_func(**kwargs)
    else:
        parse_args(args, ["input", "out_cbody", "out_residual", "grad"], kwargs)
        return _decomposeMVF_grad(**kwargs)


def hybrid_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    threshold: _float = 0.15,
    inplace: _bool = False,
) -> Composition:
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref
    recovery_out = func(recovery)
    residual_out = func(input._residual_tensor)

    decompose_out = recovery_out - residual_out

    composition = input._component_tensor
    sum_composition = composition.sum(dim=0)
    abs_composition = composition.abs()
    abs_sum_composition = abs_composition.sum(dim=0, keepdim=True)
    instability_ratio = sum_composition.abs() / abs_sum_composition
    mask = (instability_ratio < threshold).expand_as(composition)

    if not inplace:
        composition = composition.clone()

    composition[mask] = composition[mask].abs()

    multiplier = decompose_out / composition.sum(dim=0)

    if inplace:
        input._component_tensor *= multiplier
        input._residual_tensor = residual_out
        return input
    else:
        out_component_tensor = composition * multiplier
        out_residual_tensor = residual_out
        return pydec._from_replce(out_component_tensor, out_residual_tensor)
