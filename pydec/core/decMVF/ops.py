"""
Decomposition of Multi-Variable Functions.
"""

from __future__ import annotations
import torch
import pydec
from torch import Tensor

from ...utils import parse_args

from typing import (
    Dict,
    List,
    Tuple,
    Union,
    Any,
    Callable,
    Optional,
    overload,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ..._composition import Composition

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
)

import functools

from pydec.exception_utils import args_error

# TODO: disable grad tracing in ops


# def _decomposeMVF_grad(
#     input: Composition, out_cbody: Tensor, out_residual: Tensor, grad: Tensor
# ) -> Composition:
#     if (
#         not isinstance(input, Composition)
#         or not isinstance(out_cbody, Tensor)
#         or not isinstance(out_residual, Tensor)
#         or not isinstance(grad, Tensor)
#     ):
#         raise args_error(decomposeMVF.__name__, input, out_cbody, out_residual, grad)
#     out_components = torch.matmul(input.components, grad.transpose(-1, -2))
#     multiplier = out_cbody / out_components.sum(dim=0)
#     out_components *= multiplier
#     return pydec.as_composition(out_components, out_residual)


# def _decomposeMVF_func(
#     input: Composition,
#     func: Callable[[Tensor], Tensor],
#     ref: Optional[Tensor] = None,
#     grad: Optional[Tensor] = None,
# ):
#     if (
#         not isinstance(input, Composition)
#         or (ref is not None and not isinstance(ref, Tensor))
#         or (grad is not None and not isinstance(grad, Tensor))
#     ):
#         raise args_error(decomposeMVF.__name__, input, func, ref, grad)
#     if ref is None:
#         ref = input.c_sum()
#     if grad is not None:
#         with torch.no_grad():
#             c_out = func(ref)
#             out_residual = func(input.residual)
#             out_cbody = c_out - out_residual
#             out_dim = c_out.size(-1)
#             in_dim = input.size(-1)
#             if grad.dim() < 2 or grad.size()[-2:] != (out_dim, in_dim):
#                 raise RuntimeError  # TODO: add msg
#     else:
#         input_grad_holder = ref.clone().detach_()
#         input_grad_holder.requires_grad_(True)
#         with torch.enable_grad():
#             c_out = func(input_grad_holder)
#             grad = []
#             for i in range(c_out.size(-1)):
#                 ...
#             # loss = c_out.sum(dim=0,...,-2)
#             # TODO
#     return _decomposeMVF_grad(input, out_cbody, out_residual, grad, inplace=inplace)


# @overload
# def decomposeMVF(
#     input: Composition,
#     func: Callable[[Tensor], Tensor],
#     ref: Optional[Tensor] = None,
#     grad: Optional[Tensor] = None,
# ) -> Composition:
#     ...


# @overload
# def decomposeMVF(
#     input: Composition, out_cbody: Tensor, out_residual: Tensor, grad: Tensor
# ) -> Composition:
#     ...


# def decomposeMVF(*args, **kwargs) -> Composition:
#     mode = "func"
#     if len(args) > 1:
#         if isinstance(args[1], Tensor):
#             mode = "grad"
#     elif "cbody" in kwargs:
#         mode = "grad"
#     if mode == "func":
#         parse_args(args, ["input", "func", "ref", "grad"], kwargs)
#         return _decomposeMVF_func(**kwargs)
#     else:
#         parse_args(args, ["input", "out_cbody", "out_residual", "grad"], kwargs)
#         return _decomposeMVF_grad(**kwargs)


__all__ = [
    "cc_mul",
    "cc_mul_",
    "cc_div",
    "cc_div_composite",
    "cc_matmul",
    "cc_bmm",
    "var",
]


def cc_mul(
    input: Composition,
    other: Composition,
    *,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    """
    f(x1,x2) = x1*x2 = (c1+b1)*(c2+b2) = fc1 + fc2
    fc1 = 0.5*x1*x2 - 0.5*b1*x2 + 0.5*x1*b2 - 0.5*b1*b2
    fc2 = 0.5*x1*x2 - 0.5*x1*b2 + 0.5*b1*x2 - 0.5*b1*b2
    """
    if ref_input is None:
        ref_input = input.c_sum()
    if ref_other is None:
        ref_other = other.c_sum()
    out_residual = torch.mul(input.residual, other.residual)
    ref_out = torch.mul(ref_input, ref_other)
    shapley_bias = -torch.mul(input.residual, ref_other) + torch.mul(
        ref_input, other.residual
    )  # -b1*x2 + x1*b2

    c_body = ref_out - out_residual
    out_c1 = 0.5 * (c_body + shapley_bias)
    out_c2 = 0.5 * (c_body - shapley_bias)
    # out_c1 = 0.5 * c_body
    # out_c2 = out_c1

    # TODO: bug: for 0-dim tensor, should append a dim to composition (components tensor should have at least 2 dim)
    multiplier1 = out_c1 / input.components.sum(dim=0, keepdim=True)
    multiplier2 = out_c2 / other.components.sum(dim=0, keepdim=True)
    multiplier1.nan_to_num_(0, 0, 0)
    multiplier2.nan_to_num_(0, 0, 0)
    out_component_tensor = (
        multiplier1 * input._component_tensor + multiplier2 * other._component_tensor
    )
    if out is not None:
        out.residual[:] = out_residual
        out.components[:] = out_component_tensor
    return pydec.as_composition(out_component_tensor, out_residual)


def cc_mul_(
    input: Composition,
    other: Composition,
    *,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    raise NotImplementedError


def cc_div(
    input: Composition,
    other: Composition,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    # return cc_div_composite(
    #     input, other, out=out, ref_input=ref_input, ref_other=ref_other
    # )
    other = pydec.reciprocal(other, ref=ref_other)
    return cc_mul(
        input,
        other,
        out=out,
        ref_input=ref_input,
        ref_other=ref_other,
    )


def cc_div_composite(
    input: Composition,
    other: Composition,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    """
    f(x1,x2) = x1/x2 = (c1+b1)/(c2+b2) = fc1 + fc2
    fc1 = 0.5(x1/x2 - b1/x2 + x1/b2 - b1/b2)
    = 0.5[(x1-b1)/x2 + (x1-b1)/b2]
    fc2 = 0.5(x1/x2 - x1/b2 + b1/x2 - b1/b2)
    = 0.5[(x1+b1)/x2 - (x1+b1)/b2]

    TODO: not stable. (1+0)/(2+0) vs. (1+0)/(2+0.01)
    """
    if ref_input is None:
        ref_input = input.c_sum()
    if ref_other is None:
        ref_other = other.c_sum()
    out_residual = torch.div(
        input.residual, other.residual, rounding_mode=rounding_mode
    ).nan_to_num_(0, 0, 0)
    term00 = torch.div(
        ref_input - input.residual, ref_other, rounding_mode=rounding_mode
    ).nan_to_num_(0, 0, 0)
    term01 = torch.div(
        ref_input - input.residual, other.residual, rounding_mode=rounding_mode
    ).nan_to_num_(0, 0, 0)
    term10 = torch.div(
        ref_input + input.residual, ref_other, rounding_mode=rounding_mode
    ).nan_to_num_(0, 0, 0)
    term11 = torch.div(
        ref_input + input.residual, other.residual, rounding_mode=rounding_mode
    ).nan_to_num_(0, 0, 0)

    out_c1 = 0.5 * (term00 + term01)
    out_c2 = 0.5 * (term10 - term11)
    multiplier1 = out_c1 / input.components.sum(dim=0, keepdim=True)
    multiplier2 = out_c2 / other.components.sum(dim=0, keepdim=True)

    multiplier1.nan_to_num_(0, 0, 0)
    multiplier2.nan_to_num_(0, 0, 0)
    out_component_tensor = (
        multiplier1 * input._component_tensor + multiplier2 * other._component_tensor
    )
    if out is not None:
        out.residual[:] = out_residual
        out.components[:] = out_component_tensor
    return pydec.as_composition(out_component_tensor, out_residual)


def cc_matmul(
    input: Composition,
    other: Composition,
    *,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    if input.dim() == 1:
        return cc_matmul(
            input.unsqueeze(0), other, out=out, ref_input=ref_input, ref_other=ref_other
        ).squeeze_(-2)
    if other.dim() == 1:
        return cc_matmul(
            input,
            other.unsqueeze(-1),
            out=out,
            ref_input=ref_input,
            ref_other=ref_other,
        ).squeeze_(-1)
    input = input.unsqueeze(-2)
    other = other.unsqueeze(-3).transpose(-1, -2)
    c_out = pydec.mul(input, other, ref_input=ref_input, ref_other=ref_other).sum(-1)
    if out is not None:
        out.components[:] = c_out.components
        out.residual[:] = c_out.residual
    return c_out


def var(
    input: Composition,
    dim: Union[_int, _size],
    unbiased: _bool = True,
    keepdim: _bool = False,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    # TODO: apply linearization
    if ref is None:
        ref = input.c_sum()
    input_mean = input.mean(dim=dim, keepdim=True)
    ref_mean = ref.mean(dim=dim, keepdim=True)

    input_diff = input - input_mean
    ref_diff = ref - ref_mean
    input_vars = pydec.square(input_diff, ref=ref_diff)
    if unbiased:
        if isinstance(dim, _int):
            dim = (dim,)
        unbiased_num = (
            functools.reduce(lambda x, y: x * y, [ref.size(d) for d in dim]) - 1
        )
        return pydec.div(
            pydec.sum(input_vars, dim=dim, keepdim=keepdim), unbiased_num, out=out
        )
    else:
        return pydec.mean(input_vars, dim=dim, keepdim=keepdim, out=out)


def cc_bmm(
    input: Composition,
    mat2: Composition,
    *,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_mat2: Optional[Tensor] = None,
) -> Composition:
    input = input.unsqueeze(-2)  # b x n x 1 x m
    mat2 = mat2.unsqueeze(-3).transpose(-1, -2)  # b x 1 x p x m
    c_out = pydec.mul(input, mat2, ref_input=ref_input, ref_other=ref_mat2).sum(-1)
    if out is not None:
        out.components[:] = c_out.components
        out.residual[:] = c_out.residual
    return c_out


# deprecated
def layer_norm_loo(
    input: Composition,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    normalized_dims = tuple(range(-len(normalized_shape), 0))
    input_mean = input.mean(dim=normalized_dims, keepdim=True)
    input = input - input_mean

    if ref is None:
        ref = input.c_sum()
    else:
        ref = ref - ref.mean(dim=normalized_dims, keepdim=True)

    def central_norm(
        input: Tensor,
        normalized_dims: Tuple[int, ...],
        eps: float = 1e-5,
    ):
        # return torch.nn.functional.layer_norm(input, [input.size(-1)])
        return input / torch.sqrt(
            torch.var(input, dim=normalized_dims, unbiased=False, keepdim=True) + eps
        )

    ref_out = central_norm(ref, normalized_dims, eps=eps)
    out_residual = central_norm(input.residual, normalized_dims, eps=eps)

    out_components = torch.zeros_like(input.components)
    for i in range(input.numc()):
        masked_input: Composition = pydec.as_composition(
            input.components.clone(), input.residual
        )
        masked_input()[i] = 0
        out_components[i] = ref_out - central_norm(
            masked_input.c_sum(), normalized_dims, eps=eps
        )
    scaler = (ref_out - out_residual) / out_components.sum(dim=0)
    out_components *= scaler

    out = pydec.as_composition(out_components, out_residual)
    if weight is not None:
        out *= weight
    if bias is not None:
        out += bias
    return out


# deprecated
def layer_norm_grad(
    input: Composition,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    """
    step1: input = input - input.mean()
    step2: do mask

    diff = \frac{termb - terma}{terma*termb}x_i\\
    terma = \sqrt{\frac{1}{m}\sum_ix_i^2}\\
    termb = \sqrt{\frac{1}{m}\sum_ix_i^2-\frac{1}{m}x_j^2}
    """
    normalized_dims = tuple(range(-len(normalized_shape), 0))
    batch_shape = input.size()[: (input.ndim - len(normalized_shape))]
    input = input.view(batch_shape + (-1,))
    input = input - input.mean(dim=-1, keepdim=True)

    normalized_num = input.size(-1)

    if ref is None:
        ref = input.c_sum()
    else:
        ref.view(batch_shape + (-1,))
        ref = ref - ref.mean(dim=-1, keepdim=True)

    def central_norm(
        input: Tensor,
        normalized_dims: Tuple[int, ...],
        eps: float = 1e-5,
    ):
        # return torch.nn.functional.layer_norm(input, normalized_shape, eps=eps)
        return input / torch.sqrt(
            input.square().sum(dim=normalized_dims, keepdim=True) / input.size(-1) + eps
        )

    ref_out = central_norm(ref, (-1,), eps=eps)
    out_residual = central_norm(input.residual, (-1,), eps=eps)

    square_ref = ref.square()
    terma = square_ref.sum(dim=-1, keepdim=True)
    termb = terma - square_ref
    terma /= normalized_num
    terma.sqrt_()
    termb /= normalized_num
    termb.sqrt_()

    tw = ref.unsqueeze(-1) * ((termb - terma) / (termb * terma)).unsqueeze(-2)
    tw = tw.diagonal_scatter(ref_out, 0, -2, -1)

    # weight = torch.zeros((ref.size(0), ref.size(0)))
    # for i in range(len(weight)):
    #     new_ref = ref.clone()
    #     new_ref[i] = 0
    #     weight[:, i] = ref_out - central_norm(new_ref, [-1])

    # scaler = (ref_out - out_residual) / weight.sum(dim=1)
    # weight *= scaler[:, None]
    # weight /= input.components.sum(dim=0)
    # out_components = input.components @ weight.transpose(-1, -2)

    scaler = (ref_out - out_residual) / tw.sum(dim=-1)
    tw *= scaler.unsqueeze(-1)

    tw /= input.components.sum(dim=0).unsqueeze(-2)
    out_components = input.components @ tw.transpose(-1, -2)
    out = pydec.as_composition(out_components, out_residual)
    out = out.view(batch_shape + tuple(normalized_shape))

    if weight is not None:
        out *= weight
    if bias is not None:
        out += bias
    return out
