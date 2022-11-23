r"""Functional interface"""

import torch
from torch import Tensor
import torch.nn.functional as F
from ..composition import Composition
from ..decomposition import (
    get_decomposition_func,
    get_decomposition_name,
)
from ..exception_utils import none_decomposition_func_error
from typing import Optional


def relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.nn.functional.relu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def gelu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.nn.functional.gelu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


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
    def hybrid_decomposition(
        bias,
        context: Composition,
        *,
        threshold=0.15,
        eps=1e-6,
    ) -> Composition:
        def ratio_map(ratio: Tensor):
            zero_map = ratio < threshold
            ratio[zero_map] = 0
            ratio[~zero_map] = 1

        compositions = context._composition_tensor
        sum_compositions = compositions.sum(dim=0, keepdim=True)
        abs_compositions = compositions.abs()
        abs_sum_compositions = abs_compositions.sum(dim=0, keepdim=True)
        ratio = sum_compositions.abs() / abs_sum_compositions

        sum_compositions[sum_compositions == 0] = eps
        abs_sum_compositions[abs_sum_compositions == 0] = eps

        ratio_map(ratio)

        weights = ratio * compositions / sum_compositions
        abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

        bias_composition_tensor = weights * bias + abs_weights * bias
        from ..variable_functions import _from_replce

        out = _from_replce(bias_composition_tensor)
        return out

    # return legacy_relu(input, ref)
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    isolated_relu_out = torch.nn.functional.relu(input._residual_tensor)

    out = input.masked_fill(zero_mask, 0.0)
    relu_out = out._residual_tensor

    delta_relu_out = relu_out - isolated_relu_out
    decomposition_func = hybrid_decomposition
    bias_composition = decomposition_func(bias=delta_relu_out, context=input)
    out._residual_tensor = isolated_relu_out
    out = out + bias_composition
    return out
