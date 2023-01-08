from __future__ import annotations
import torch
from torch import Tensor

from typing import Dict, Tuple, Union, Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
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


from ..variable_functions import _from_replce, zeros_like
from .states import register_decomposition_func, set_decomposition_func

# def _non_linear_decompose(
#     input: Composition, func: Callable[[Tensor], Tensor]
# ) -> Composition:
#     """
#     Non-linear decomposition for **point-wise opertations**.
#     """
#     recovery = input.c_sum()
#     residual = input._residual_tensor
#     recovery_out = func(recovery)
#     residual_out = func(residual)

#     decompose_tensor = recovery_out - residual_out

#     decomposition_func = get_decomposition_func()
#     if decomposition_func is not None:
#         out = decomposition_func(bias=decompose_tensor, context=input)
#         out._residual_tensor = out._residual_tensor + residual_out
#         return out
#     else:
#         raise none_decomposition_func_error(get_decomposition_name())


# def _base_decomposition(
#     input: Tensor,
#     residual: Tensor,
#     func: Callable[[Tensor], Tensor],
#     *,
#     inplace: _bool = False,
# ) -> Tensor:
#     """
#     Note: if inplace is True, the func must be a inplace function.
#     """
#     out = func(input)
#     residual_out = func(residual)

#     decompose_tensor = recovery_out - residual_out
#     # TODO: use composition add
#     if inplace:
#         input._residual_tensor = recovery_out
#         return input, decompose_tensor
#     else:
#         out_composition_tensor = input._composition_tensor.clone()
#         out_residual_tensor = residual_out
#         out = _from_replce(out_composition_tensor, out_residual_tensor)
#         return out, decompose_tensor


@register_decomposition_func("none")
def _none_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    inplace: _bool = False,
) -> Composition:
    r"""
    Note: since Pydec TODO(add a version), this algorithm is no longer the default
    decomposition algorithm for pydec, as the results it obtains do not make any sense.

    A trivial decomposition algorithm. Just add the output to residual.
    """
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref
    recovery_out = func(recovery)

    if inplace:
        input._composition_tensor[:] = 0
        input._residual_tensor = recovery_out
        return input
    else:
        out = zeros_like(input)
        out._residual_tensor += recovery_out
        return out


@register_decomposition_func("abs_decomposition")
def abs_decomposition(
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

    composition = input._composition_tensor
    abs_composition = composition.abs()

    multiplier = decompose_out / abs_composition.sum(dim=0)

    if inplace:
        input._composition_tensor.abs_()
        input._composition_tensor *= multiplier
        input._residual_tensor = residual_out
        return input
    else:
        out_composition_tensor = abs_composition * multiplier
        out_residual_tensor = residual_out
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_decomposition_func("hybrid_decomposition")
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

    composition = input._composition_tensor
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
        input._composition_tensor *= multiplier
        input._residual_tensor = residual_out
        return input
    else:
        out_composition_tensor = composition * multiplier
        out_residual_tensor = residual_out
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_decomposition_func("sampled_shaply")
def sampled_shaply(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    inplace: _bool = False,
) -> Composition:
    """
    not work.
    """
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref

    composition = input._composition_tensor
    residual = input._residual_tensor
    recovery_out = func(recovery)
    residual_out = func(residual)

    delta = recovery_out - residual_out

    sampled_input = (
        composition.sum(dim=0, keepdim=True).expand_as(composition).contiguous()
    )
    sampled_input -= composition
    sampled_input = (
        sampled_input[None].expand((input.numc(),) + sampled_input.size()).contiguous()
    )
    multiplier = torch.linspace(0, 1, steps=input.numc(), device=composition.device)
    multiplier = multiplier.view((input.numc(),) + composition.dim() * (1,))
    sampled_input *= multiplier
    exclusive_out = func(sampled_input + residual)
    inclusive_out = func(sampled_input + composition + residual)
    sampled_shaply_value = (inclusive_out - exclusive_out).mean(dim=0)

    if (sampled_shaply_value.sum(dim=0) - delta).abs().sum() > 0.1:
        fix_multiplier = delta / sampled_shaply_value.sum(dim=0)
        sampled_shaply_value *= fix_multiplier

    if inplace:
        input._composition_tensor = sampled_shaply_value
        input._residual_tensor = residual_out
        return input
    else:
        out_composition_tensor = sampled_shaply_value
        out_residual_tensor = residual_out
        return _from_replce(out_composition_tensor, out_residual_tensor)


# TODO: the algorithms below are deprecated.

"""
@register_decomposition_func("abs_decomposition")
def abs_decomposition(
    bias: Union[Number, Tensor],
    context: Composition,
    *,
    eps=1e-6,
) -> Composition:
    compositions = context._composition_tensor
    abs_compositions = compositions.abs()
    sum_compositions = abs_compositions.sum(dim=0, keepdim=True)
    sum_compositions[sum_compositions == 0] = eps
    weights = abs_compositions / sum_compositions
    bias_composition_tensor = weights * bias
    out = _from_replce(bias_composition_tensor)
    return out


@register_decomposition_func("hybrid_decomposition")
def hybrid_decomposition(
    bias: Union[Number, Tensor],
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
    out = _from_replce(bias_composition_tensor)
    return out


@register_decomposition_func("sign_decomposition")
def sign_decomposition(
    bias: Union[Number, Tensor],
    context: Composition,
    *,
    threshold=0.4,
    eps=1e-6,
) -> Composition:
    def ratio_map(ratio: Tensor):
        zero_map = ratio < threshold
        ratio[zero_map] = 0
        ratio[~zero_map] = 1

    compositions = context._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    abs_sum_compositions = compositions.abs().sum(dim=0, keepdim=True)
    ratio = sum_compositions.abs() / abs_sum_compositions

    sum_compositions[sum_compositions == 0] = eps

    ratio_map(ratio)
    weights = ratio * compositions / sum_compositions
    bias_composition_tensor = weights * bias
    bias_residula_tensor = (1 - weights.sum(dim=0)) * bias

    out = _from_replce(bias_composition_tensor, bias_residula_tensor)
    return out


@register_decomposition_func("sign_decomposition_threshold")
def sign_decomposition_threshold(
    bias: Union[Number, Tensor],
    context: Composition,
    *,
    threshold=0.4,
    eps=1e-6,
) -> Composition:
    compositions = context._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    ratio = (sum_compositions.abs() > threshold).to(torch.float)

    sum_compositions[sum_compositions == 0] = eps

    weights = ratio * compositions / sum_compositions

    bias_composition_tensor = weights * bias
    bias_residula_tensor = (1 - weights.sum(dim=0)) * bias
    out = _from_replce(bias_composition_tensor, bias_residula_tensor)
    return out


@register_decomposition_func("hybrid_decomposition_threshold")
def hybrid_decomposition_threshold(
    bias: Union[Number, Tensor],
    context: Composition,
    *,
    threshold=0.15,
    eps=1e-6,
) -> Composition:
    compositions = context._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    abs_compositions = compositions.abs()
    abs_sum_compositions = abs_compositions.sum(dim=0, keepdim=True)

    ratio = (sum_compositions.abs() > threshold).to(torch.float)

    sum_compositions[sum_compositions == 0] = eps
    abs_sum_compositions[abs_sum_compositions == 0] = eps

    weights = ratio * compositions / sum_compositions
    abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

    bias_composition_tensor = weights * bias + abs_weights * bias
    out = _from_replce(bias_composition_tensor)
    return out


@register_decomposition_func("norm_decomposition")
def norm_decomposition(
    bias: Union[Number, Tensor],
    context: Composition,
    *,
    p=float("inf"),  # 2,
    eps=1e-6,
) -> Composition:
    compositions = context._composition_tensor
    norm_compositions = torch.norm(compositions, p=p, dim=-1, keepdim=True)
    sum_compositions = norm_compositions.sum(dim=0, keepdim=True)
    sum_compositions[sum_compositions == 0] = eps

    weights = norm_compositions / sum_compositions

    bias_composition_tensor = weights * bias
    return _from_replce(bias_composition_tensor)

"""

# @register_decomposition_func("sparse_abs_decomposition")
# def forward_sparse_abs_decomposition(x: Tensor, eps=1e-6):
#     # import pdb
#     # pdb.set_trace()
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     abs_compositions = compositions.abs()

#     mask_index = abs_compositions.topk(
#         k=int(abs_compositions.size(1) * 5 / 10), dim=1, largest=False
#     )[1]
#     abs_compositions.scatter_(1, mask_index, 0.0)
#     sum_compositions = abs_compositions.sum(dim=1, keepdim=True)
#     sum_compositions[sum_compositions == 0] = eps
#     weights = abs_compositions / sum_compositions
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("sparse_norm_decomposition")
# def forward_sparse_norm_decomposition(x: Tensor, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     norm_compositions = torch.norm(
#         compositions, p=float("inf"), dim=-1, keepdim=True
#     )  # T x CT x B x 1

#     mask_index = norm_compositions.topk(
#         k=int(norm_compositions.size(1) * 1 / 3), dim=1, largest=False
#     )[1]
#     norm_compositions.scatter_(1, mask_index, 0.0)
#     sum_compositions = norm_compositions.sum(dim=1, keepdim=True)
#     sum_compositions[sum_compositions == 0] = eps
#     weights = norm_compositions / sum_compositions  # T x CT x B x 1
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("sparse_hybrid_decomposition")
# def forward_sparse_hybrid_decomposition(x, eps=1e-6):
#     def ratio_map(ratio: Tensor):
#         zero_map = ratio < 0.3
#         ratio[zero_map] = 0
#         ratio[~zero_map] = 1

#         # zero_map = ratio < 0.1
#         # one_map = ratio > 0.2
#         # ratio[zero_map] = 0
#         # ratio[one_map] = 1
#         # ratio[~zero_map & ~one_map] = 10 * (ratio[~zero_map & ~one_map] - 0.1)

#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]

#     abs_compositions = compositions.abs()

#     mask_index = abs_compositions.topk(
#         k=int(abs_compositions.size(1) * 3 / 10), dim=1, largest=False
#     )[1]
#     compositions = compositions.scatter(1, mask_index, 0.0)
#     abs_compositions.scatter_(1, mask_index, 0.0)

#     sum_compositions = compositions.sum(dim=1, keepdim=True)
#     abs_sum_compositions = abs_compositions.sum(dim=1, keepdim=True)
#     ratio = sum_compositions.abs() / abs_sum_compositions

#     sum_compositions[sum_compositions == 0] = eps
#     abs_sum_compositions[abs_sum_compositions == 0] = eps

#     ratio_map(ratio)

#     weights = ratio * compositions / sum_compositions
#     abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

#     x[:, 1:] += weights * residual + abs_weights * residual
#     # x[:, 1:] += abs_weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("positive_decomposition")
# def forward_sign_decomposition(x, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     compositions
#     sum_compositions = compositions.sum(dim=1, keepdim=True)
#     sum_compositions[sum_compositions == 0] = eps
#     weights = compositions / sum_compositions
#     overflow_indices = sum_compositions.abs() < 0.4
#     weights.masked_fill_(overflow_indices, 0.0)
#     x[:, 1:] += weights * residual
#     x[:, 0:1].masked_fill_(~overflow_indices, 0.0)
#     x = forward_abs_decomposition(x)
#     return x


# @register_decomposition_func("hybrid_norm_decomposition")
# def forward_hybrid_norm_decomposition(x: Tensor, power_factor=1, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]  # T x CT x B x C
#     product = torch.matmul(residual.unsqueeze(-2), compositions.unsqueeze(-1)).squeeze(
#         -1
#     )  # T x CT x B x 1
#     sum_product = product.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     abs_product = product.abs()
#     abs_sum_product = abs_product.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     ratio = sum_product.abs() / abs_sum_product  # T x 1 x B x 1

#     sum_product[sum_product == 0] = eps
#     abs_sum_product[abs_sum_product == 0] = eps

#     ratio = ratio**power_factor
#     # ratio = 1

#     weights = ratio * product / sum_product  # T x CT x B x 1
#     abs_weights = (1 - ratio) * abs_product / abs_sum_product  # T x CT x B x 1
#     x[:, 1:] += weights * residual + abs_weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("softmax_decomposition")
# def forward_softmax_decomposition(x, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     norm_compositions = torch.norm(
#         compositions, p=float("inf"), dim=-1, keepdim=True
#     )  # T x CT x B x 1
#     weights = torch.softmax(norm_compositions, dim=1)

#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("norm_softmax_decomposition")
# def forward_norm_softmax_decomposition(x, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     norm_compositions = torch.norm(
#         compositions, p=2, dim=-1, keepdim=True
#     )  # T x CT x B x 1
#     sum_compositions = norm_compositions.sum(dim=1, keepdim=True)  # T x 1 x B x 1
#     sum_compositions[sum_compositions == 0] = eps

#     weights = norm_compositions / sum_compositions  # T x CT x B x 1

#     weights = torch.softmax(weights, dim=1)  # T x CT x B x 1
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("average_decomposition")
# def forward_average_decomposition(x: Tensor, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     weights = torch.ones(compositions.size()[:-1]).to(compositions).unsqueeze(
#         -1
#     ) / compositions.size(1)
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_decomposition_func("sparse_norm_decomposition_sparsification")
# def forward_sparse_norm_decomposition_sparsification(x: Tensor, eps=1e-6):
#     """
#     To sparsificate the compositions, which will make the compositions to
#     be inconsistent with original hidden state
#     """
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     norm_compositions = torch.norm(
#         compositions, p=float("inf"), dim=-1, keepdim=True
#     )  # T x CT x B x 1

#     mask_index = norm_compositions.topk(
#         k=int(norm_compositions.size(1) * 9 / 10), dim=1, largest=False
#     )[1]

#     compositions.scatter_(1, mask_index, 0.0)

#     norm_compositions = torch.norm(
#         compositions, p=float("inf"), dim=-1, keepdim=True
#     )  # T x CT x B x 1

#     sum_compositions = norm_compositions.sum(dim=1, keepdim=True)
#     sum_compositions[sum_compositions == 0] = eps
#     weights = norm_compositions / sum_compositions  # T x CT x B x 1
#     x[:, 1:] = compositions
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x

"""
Initialization
"""
try:
    set_decomposition_func("hybrid_decomposition")
except ValueError:
    from . import states

    set_decomposition_func(
        states._DECOMPOSITION_FUNC_REGISTRY.keys().__iter__().__next__()
    )
