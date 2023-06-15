from __future__ import annotations
import pydec
import torch
from torch import Tensor
from typing import Any, Union, List, Tuple, Optional, Callable, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from pydec import Composition


# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
# In the future, this will become useful if mypy is introduced into pydec
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

from pydec.exception_utils import component_num_error

__all__ = [
    "cc_add",
    "cc_add_",
    "ct_add",
    "ct_add_",
    "tc_add",
    "cc_sub",
    "cc_sub_",
    "ct_sub",
    "ct_sub_",
    "tc_sub",
    "ct_mul",
    "ct_mul_",
    "tc_mul",
    "ct_div",
    "ct_div_",
    "ct_matmul",
    "tc_matmul",
    "ct_mv",
    "tc_mv",
    "ct_mm",
    "tc_mm",
    "ct_bmm",
    "tc_bmm",
]


def cc_add(
    input: Composition,
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if input.numc() != other.numc():
        raise component_num_error(input.numc(), other.numc())
    out_component_tensor = torch.add(
        input._component_tensor,
        other._component_tensor,
        alpha=alpha,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.add(
        input._residual_tensor,
        other._residual_tensor,
        alpha=alpha,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def cc_add_(
    input: Composition,
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
) -> Composition:
    if input.numc() != other.numc():
        raise component_num_error(input.numc(), other.numc())
    out_component_tensor = input._component_tensor.add_(
        other._component_tensor,
        alpha=alpha,
    )
    out_residual_tensor = input._residual_tensor.add_(
        other._residual_tensor,
        alpha=alpha,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_add(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        out._component_tensor[:] = input._component_tensor
        out_component_tensor = out._component_tensor
    else:
        out_component_tensor = input._component_tensor.clone()
    out_residual_tensor = torch.add(
        input._residual_tensor,
        other,
        alpha=alpha,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_add_(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
) -> Composition:
    out_component_tensor = input._component_tensor
    out_residual_tensor = input._residual_tensor.add_(
        other,
        alpha=alpha,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_add(
    input: Union[Tensor, Number],
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    return ct_add(other, input, alpha=alpha, out=out)


def cc_sub(
    input: Composition,
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if input.numc() != other.numc():
        raise component_num_error(input.numc(), other.numc())
    out_component_tensor = torch.sub(
        input._component_tensor,
        other._component_tensor,
        alpha=alpha,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.sub(
        input._residual_tensor,
        other._residual_tensor,
        alpha=alpha,
        out=out.out_residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def cc_sub_(
    input: Composition,
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
) -> Composition:
    if input.numc() != other.numc():
        raise component_num_error(input.numc(), other.numc())
    out_component_tensor = input._component_tensor.sub_(
        other._component_tensor,
        alpha=alpha,
    )
    out_residual_tensor = input._residual_tensor.sub_(
        other._residual_tensor,
        alpha=alpha,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_sub(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        out._component_tensor[:] = input._component_tensor
        out_component_tensor = out._component_tensor
    else:
        out_component_tensor = input._component_tensor.clone()
    out_residual_tensor = torch.sub(
        input._residual_tensor,
        other,
        alpha=alpha,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_sub_(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
) -> Composition:
    out_component_tensor = input._component_tensor
    out_residual_tensor = input._residual_tensor.sub_(
        other,
        alpha=alpha,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_sub(
    input: Union[Tensor, Number],
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        out._component_tensor[:] = other._component_tensor
        out_component_tensor = out._component_tensor
    else:
        out_component_tensor = input._component_tensor.clone()
    out_residual_tensor = torch.sub(
        input,
        other._residual_tensor,
        alpha=alpha,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_mul(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(other, Tensor) and other.dim() > input.dim():
        # handle broadcast
        new_size = (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
        out_component_tensor = torch.mul(
            input._component_tensor.view(new_size),
            other,
            out=out._component_tensor if out is not None else None,
        )
    else:
        out_component_tensor = torch.mul(
            input._component_tensor,
            other,
            out=out._component_tensor if out is not None else None,
        )
    out_residual_tensor = torch.mul(
        input._residual_tensor,
        other,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_mul_(input: Composition, other: Union[Tensor, Number]) -> Composition:
    if isinstance(other, Tensor) and other.dim() > input.dim():
        # handle broadcast
        new_size = (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
        input._component_tensor = input._component_tensor.view(new_size).mul_(other)
    else:
        input._component_tensor = input._component_tensor.mul_(other)
    input._residual_tensor = input._residual_tensor.mul_(other)
    return input


def tc_mul(
    input: Union[Tensor, Number],
    other: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    return ct_mul(other, input, out=out)


def ct_div(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(other, Tensor) and other.dim() > input.dim():
        # handle broadcast
        new_size = (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
        out_component_tensor = torch.div(
            input._component_tensor.view(new_size),
            other,
            rounding_mode=rounding_mode,
            out=out._component_tensor if out is not None else None,
        )
    else:
        out_component_tensor = torch.div(
            input._component_tensor,
            other,
            rounding_mode=rounding_mode,
            out=out._component_tensor if out is not None else None,
        )
    out_residual_tensor = torch.div(
        input._residual_tensor,
        other,
        rounding_mode=rounding_mode,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_div_(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    rounding_mode: Optional[str] = None,
) -> Composition:
    if isinstance(other, Tensor) and other.dim() > input.dim():
        # handle broadcast
        new_size = (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
        input._component_tensor = input._component_tensor.view(new_size).div_(
            other, rounding_mode=rounding_mode
        )
    else:
        input._component_tensor = input._component_tensor.div_(
            other,
            rounding_mode == rounding_mode,
        )
    input._residual_tensor = input._residual_tensor.div_(
        other,
        rounding_mode=rounding_mode,
    )
    return input


def ct_matmul(
    input: Composition, other: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_component_tensor = torch.matmul(
        input._component_tensor,
        other,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.matmul(
        input._residual_tensor,
        other,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_matmul(
    input: Tensor, other: Composition, *, out: Optional[Composition] = None
) -> Composition:
    if other.dim() == 1:
        # if the component_tensor's ndim is 2, the component dim
        # will be incorrectly included in the multiplication
        out_component_tensor = torch.matmul(
            input,
            other._component_tensor.unsqueeze(-1),
            out=out._component_tensor.unsqueeze_(-1) if out is not None else None,
        )
        out_component_tensor.squeeze_(-1)
        if out is not None:
            out._component_tensor.squeeze_(-1)
    else:
        out_component_tensor = torch.matmul(
            input,
            other._component_tensor,
            out=out._component_tensor if out is not None else None,
        )
    out_residual_tensor = torch.matmul(
        input,
        other._residual_tensor,
        out=out._residual_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_mv(
    input: Composition,
    vec: Tensor,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.mv(
        input._residual_tensor,
        vec,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input._component_tensor,
        vec,
        out=out._component_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_mv(
    input: Tensor,
    vec: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.mv(
        input,
        vec._residual_tensor,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input,
        vec._component_tensor.unsqueeze(-1),
        out=out._component_tensor.unsqueeze_(-1) if out is not None else None,
    )
    out_component_tensor.squeeze_(-1)
    if out is not None:
        out._component_tensor.squeeze_(-1)
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_mm(
    input: Composition,
    mat2: Tensor,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.mm(
        input._residual_tensor,
        mat2,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input._component_tensor,
        mat2,
        out=out._component_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_mm(
    input: Tensor,
    mat2: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.mm(
        input,
        mat2._residual_tensor,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input,
        mat2._component_tensor,
        out=out._component_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def ct_bmm(
    input: Composition,
    mat2: Tensor,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.bmm(
        input._residual_tensor,
        mat2,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input._component_tensor,
        mat2,
        out=out._component_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def tc_bmm(
    input: Tensor,
    mat2: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_residual_tensor = torch.bmm(
        input,
        mat2._residual_tensor,
        out=out._residual_tensor if out is not None else None,
    )
    out_component_tensor = torch.matmul(
        input,
        mat2._component_tensor,
        out=out._component_tensor if out is not None else None,
    )
    return pydec.as_composition(out_component_tensor, out_residual_tensor)
