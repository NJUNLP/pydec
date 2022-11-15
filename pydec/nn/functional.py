r"""Functional interface"""

import torch
from torch import Tensor
import torch.nn.functional as F
from ..composition import Composition
from ..bias_decomposition import (
    get_bias_decomposition_func,
    get_bias_decomposition_name,
)
from ..variable_functions import _from_replce
from ..exception_utils import none_bias_decomposition_func_error

from typing import Optional, Callable


def _non_linear_decompose(
    input: Composition, func: Callable[[Tensor], Tensor]
) -> Composition:
    """
    Non-linear decomposition for **point-wise opertations**.
    """
    recovery = input.c_sum()
    residual = input._residual_tensor
    recovery_out = func(recovery)
    residual_out = func(residual)

    decompose_tensor = recovery_out - residual_out

    decomposition_func = get_bias_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(bias=decompose_tensor, context=input)
        out._residual_tensor = out._residual_tensor + residual_out
        return out
    else:
        raise none_bias_decomposition_func_error(get_bias_decomposition_name())


def relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    return _non_linear_decompose(input, func=F.relu)


def linear(input: Composition, weight: Tensor, bias: Tensor = None) -> Composition:
    out = input @ weight.t()
    if bias is not None:
        out._residual_tensor = out._residual_tensor + bias
        return out
    else:
        return out


def layer_norm_1d(
    input: Composition,
    ref: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Composition:
    r"""Applies Layer Normalization for last dimension."""
    input_mean = input.mean(dim=-1, keepdim=True)
    if ref is None:
        ref = input.c_sum()
    input_std = torch.sqrt(torch.var(ref, dim=-1, unbiased=False, keepdim=True) + eps)
    out = (input - input_mean) * weight / input_std

    if bias is not None:
        out._residual_tensor = out._residual_tensor + bias
        return out
    else:
        return out


def legacy_relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    # return legacy_relu(input, ref)
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    isolated_relu_out = torch.nn.functional.relu(input._residual_tensor)

    out = input.masked_fill(zero_mask, 0.0)
    relu_out = out._residual_tensor

    delta_relu_out = relu_out - isolated_relu_out
    decomposition_func = get_bias_decomposition_func()
    if decomposition_func is not None:
        bias_composition = decomposition_func(bias=delta_relu_out, context=input)
        out._residual_tensor = isolated_relu_out
        out = out + bias_composition
        return out
    else:
        raise none_bias_decomposition_func_error(get_bias_decomposition_name())
