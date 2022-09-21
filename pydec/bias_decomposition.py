from __future__ import annotations

import torch
from torch import Tensor

from typing import Union, Any, Callable, ContextManager, TYPE_CHECKING

if TYPE_CHECKING:
    from .composition import Composition

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


from .variable_functions import _from_replce

_BIAS_DECOMPOSITION_FUNC_REGISTRY = {}


class _BiasDecompositionState:
    bias_decomposition_name: str = None


def register_bias_decomposition_func(name):
    """
    New bias_decomposition_func can be added with the :func:`register_bias_decomposition_func`
    function decorator.

    For example::

        @register_model('forward_norm_decomposition')
        def forward_abs_decomposition(c: Composition, bias: Union[Number, Tensor] = None, *, eps=1e-6):
            (...)

    Args:
        name (str): the name of the funcion
    """

    def register_func(func):
        if name in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        # if name == "none":
        #     raise ValueError(
        #         'Cannot register function ({}), the name "none" is reserved.'.format(
        #             name
        #         )
        #     )
        _BIAS_DECOMPOSITION_FUNC_REGISTRY[name] = func
        return func

    return register_func


def set_bias_decomposition_func(name: str) -> None:
    if name not in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
        raise ValueError(
            "Bias decomposition function ({}) is not registered".format(name)
        )
    _BiasDecompositionState.bias_decomposition_name = name


def get_bias_decomposition_name() -> str:
    return _BiasDecompositionState.bias_decomposition_name


def get_bias_decomposition_func(inplace: bool = False) -> Callable[..., Composition]:
    if (
        _BiasDecompositionState.bias_decomposition_name
        not in _BIAS_DECOMPOSITION_FUNC_REGISTRY
    ):
        return None
    else:
        current_bias_decomposition_func = _BIAS_DECOMPOSITION_FUNC_REGISTRY[
            _BiasDecompositionState.bias_decomposition_name
        ]
        if inplace:
            if _BiasDecompositionState.bias_decomposition_name.endswith("_"):
                return current_bias_decomposition_func
            else:
                inplace_func_name = (
                    _BiasDecompositionState.bias_decomposition_name + "_"
                )
                if inplace_func_name in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
                    return _BIAS_DECOMPOSITION_FUNC_REGISTRY[inplace_func_name]
                else:
                    return current_bias_decomposition_func
        else:
            return current_bias_decomposition_func


class using_bias_decomposition_func(ContextManager):
    def __init__(self, name: str) -> None:
        if name not in _BIAS_DECOMPOSITION_FUNC_REGISTRY:
            raise ValueError(
                "Bias decomposition function ({}) is not registered".format(name)
            )
        self.prev = None
        self.using_name = name

    def __enter__(self):
        self.prev = _BiasDecompositionState.bias_decomposition_name
        _BiasDecompositionState.bias_decomposition_name = self.using_name

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _BiasDecompositionState.bias_decomposition_name = self.prev


class no_bias_decomposition(ContextManager):
    def __init__(
        self,
    ) -> None:
        self.prev = None

    def __enter__(self):
        self.prev = _BiasDecompositionState.bias_decomposition_name
        _BiasDecompositionState.bias_decomposition_name = "none"

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _BiasDecompositionState.bias_decomposition_name = self.prev


@register_bias_decomposition_func("none")
def _none_decomposition(c: Composition, bias: Union[Number, Tensor] = None):
    r"""
    Default decomposition with no_bias_decomposition. Just add the bias to residual.
    """
    return _from_replce(c._composition_tensor, c._residual_tensor + bias)


@register_bias_decomposition_func("none_")
def _none_decomposition_(c: Composition, bias: Union[Number, Tensor] = None):
    c._residual_tensor += bias
    return c


@register_bias_decomposition_func("abs_decomposition")
def abs_decomposition(
    c: Composition, bias: Union[Number, Tensor] = None, *, eps=1e-6, inplace=False
) -> Composition:
    if bias is None:
        bias = c._residual_tensor
        out_residual_tensor = torch.zeros_like(c._residual_tensor).to(
            c._residual_tensor
        )
    else:
        out_residual_tensor = c._residual_tensor
    compositions = c._composition_tensor
    abs_compositions = compositions.abs()
    sum_compositions = abs_compositions.sum(dim=0, keepdim=True)
    sum_compositions[sum_compositions == 0] = eps
    weights = abs_compositions / sum_compositions
    if inplace:
        compositions += weights * bias
        c._residual_tensor = out_residual_tensor
        return c
    else:
        out_composition_tensor = compositions + weights * bias
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_bias_decomposition_func("abs_decomposition_")
def abs_decomposition_(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    eps=1e-6,
) -> Composition:
    return abs_decomposition(c, bias, eps=eps, inplace=True)


@register_bias_decomposition_func("hybrid_decomposition")
def hybrid_decomposition(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
    inplace=False,
) -> Composition:
    def ratio_map(ratio: Tensor):
        zero_map = ratio < threshold
        ratio[zero_map] = 0
        ratio[~zero_map] = 1

    if bias is None:
        bias = c._residual_tensor
        out_residual_tensor = torch.zeros_like(c._residual_tensor).to(
            c._residual_tensor
        )
    else:
        out_residual_tensor = c._residual_tensor

    compositions = c._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    abs_compositions = compositions.abs()
    abs_sum_compositions = abs_compositions.sum(dim=0, keepdim=True)
    ratio = sum_compositions.abs() / abs_sum_compositions

    sum_compositions[sum_compositions == 0] = eps
    abs_sum_compositions[abs_sum_compositions == 0] = eps

    ratio_map(ratio)

    weights = ratio * compositions / sum_compositions
    abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

    if inplace:
        compositions += weights * bias + abs_weights * bias
        c._residual_tensor = out_residual_tensor
        return c
    else:
        out_composition_tensor = compositions + weights * bias + abs_weights * bias
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_bias_decomposition_func("hybrid_decomposition_")
def hybrid_decomposition_(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
) -> Composition:
    return hybrid_decomposition(c, bias, threshold=threshold, eps=eps, inplace=True)


@register_bias_decomposition_func("sign_decomposition")
def sign_decomposition(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
    inplace=False,
) -> Composition:
    def ratio_map(ratio: Tensor):
        zero_map = ratio < threshold
        ratio[zero_map] = 0
        ratio[~zero_map] = 1

    if bias is None:
        bias = c._residual_tensor
        out_residual_tensor = torch.zeros_like(c._residual_tensor).to(
            c._residual_tensor
        )
    else:
        out_residual_tensor = c._residual_tensor

    compositions = c._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    abs_sum_compositions = compositions.abs().sum(dim=0, keepdim=True)
    ratio = sum_compositions.abs() / abs_sum_compositions

    sum_compositions[sum_compositions == 0] = eps

    ratio_map(ratio)

    weights = ratio * compositions / sum_compositions

    if inplace:
        compositions += weights * bias
        c._residual_tensor = out_residual_tensor + (1 - ratio[0]) * bias
        return c
    else:
        out_composition_tensor = compositions + weights * bias
        out_residual_tensor = out_residual_tensor + (1 - ratio[0]) * bias
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_bias_decomposition_func("sign_decomposition_")
def sign_decomposition_(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
) -> Composition:
    return sign_decomposition(c, bias, threshold=threshold, eps=eps, inplace=True)


@register_bias_decomposition_func("hybrid_decomposition_value_threshold")
def hybrid_decomposition_value_threshold(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
    inplace=False,
) -> Composition:
    if bias is None:
        bias = c._residual_tensor
        out_residual_tensor = torch.zeros_like(c._residual_tensor).to(
            c._residual_tensor
        )
    else:
        out_residual_tensor = c._residual_tensor

    compositions = c._composition_tensor
    sum_compositions = compositions.sum(dim=0, keepdim=True)
    abs_compositions = compositions.abs()
    abs_sum_compositions = abs_compositions.sum(dim=0, keepdim=True)

    ratio = (sum_compositions.abs() > threshold).to(torch.float)

    sum_compositions[sum_compositions == 0] = eps
    abs_sum_compositions[abs_sum_compositions == 0] = eps

    weights = ratio * compositions / sum_compositions
    abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

    if inplace:
        compositions += weights * bias + abs_weights * bias
        c._residual_tensor = out_residual_tensor
        return c
    else:
        out_composition_tensor = compositions + weights * bias + abs_weights * bias
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_bias_decomposition_func("hybrid_decomposition_value_threshold_")
def hybrid_decomposition_value_threshold_(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    threshold=0.4,
    eps=1e-6,
) -> Composition:
    return hybrid_decomposition_value_threshold(
        c.bias, threshold=threshold, eps=eps, inplace=True
    )


@register_bias_decomposition_func("norm_decomposition")
def norm_decomposition(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    p=2,
    eps=1e-6,
    inplace=False,
) -> Composition:
    if bias is None:
        bias = c._residual_tensor
        out_residual_tensor = torch.zeros_like(c._residual_tensor).to(
            c._residual_tensor
        )
    else:
        out_residual_tensor = c._residual_tensor

    compositions = c._composition_tensor
    norm_compositions = torch.norm(compositions, p=p, dim=-1, keepdim=True)
    sum_compositions = norm_compositions.sum(dim=0, keepdim=True)
    sum_compositions[sum_compositions == 0] = eps

    weights = norm_compositions / sum_compositions

    if inplace:
        compositions += weights * bias
        c._residual_tensor = out_residual_tensor
        return c
    else:
        out_composition_tensor = compositions + weights * bias
        return _from_replce(out_composition_tensor, out_residual_tensor)


@register_bias_decomposition_func("norm_decomposition_")
def norm_decomposition_(
    c: Composition,
    bias: Union[Number, Tensor] = None,
    *,
    p=2,
    eps=1e-6,
) -> Composition:
    return norm_decomposition(c, bias, p=2, eps=eps, inplace=True)


# @register_bias_decomposition_func("sparse_abs_decomposition")
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


# @register_bias_decomposition_func("sparse_norm_decomposition")
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


# @register_bias_decomposition_func("sparse_hybrid_decomposition")
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


# @register_bias_decomposition_func("positive_decomposition")
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


# @register_bias_decomposition_func("hybrid_norm_decomposition")
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


# @register_bias_decomposition_func("softmax_decomposition")
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


# @register_bias_decomposition_func("norm_softmax_decomposition")
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


# @register_bias_decomposition_func("average_decomposition")
# def forward_average_decomposition(x: Tensor, eps=1e-6):
#     residual = x[:, 0:1, :, :]  # T x 1 x B x C
#     compositions = x[:, 1:]
#     weights = torch.ones(compositions.size()[:-1]).to(compositions).unsqueeze(
#         -1
#     ) / compositions.size(1)
#     x[:, 1:] += weights * residual
#     x[:, 0] = 0.0
#     return x


# @register_bias_decomposition_func("sparse_norm_decomposition_sparsification")
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
    set_bias_decomposition_func("none")
except ValueError:
    set_bias_decomposition_func(
        _BIAS_DECOMPOSITION_FUNC_REGISTRY.keys().__iter__().__next__()
    )
