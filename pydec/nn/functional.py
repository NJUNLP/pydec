r"""Functional interface"""

import torch
from torch import Tensor
from ..composition import Composition
from ..bias_decomposition import (
    get_bias_decomposition_func,
    get_bias_decomposition_name,
)
from ..exception_utils import none_bias_decomposition_func_error

from typing import Optional


def relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    out = input.masked_fill(zero_mask, 0.0)
    return out


def linear(input: Composition, weight: Tensor, bias: Tensor = None) -> Composition:
    out = input @ weight.t()
    if bias is not None:
        # TODO: replace by add
        decomposition_func = get_bias_decomposition_func()
        if decomposition_func is not None:
            bias_composition = decomposition_func(bias, context=out)
            out = out + bias_composition
            return out
        else:
            raise none_bias_decomposition_func_error(get_bias_decomposition_name())
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
        # TODO: replace by add
        decomposition_func = get_bias_decomposition_func()
        if decomposition_func is not None:
            bias_composition = decomposition_func(bias, context=out)
            out = out + bias_composition
            return out
        else:
            raise none_bias_decomposition_func_error(get_bias_decomposition_name())
    else:
        return out
