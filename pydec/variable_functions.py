import torch
from torch import Tensor
from torch._C import memory_format

from pydec._composition import Composition
from .overrides import _auto_registration

from typing import Any, Union, List, Tuple, Optional, Callable, overload

# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
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

from torch import strided

from pydec.utils import _shift_dim, _shift_dims
from .decomposition import get_decomposition_func, get_decomposition_name
from pydec.exception_utils import (
    arg_value_error,
    none_decomposition_func_error,
    component_num_error,
    unsupported_operand_error,
    args_error,
)
import builtins


def void(
    *,
    dtype: _dtype = None,
    device: Union[_device, str, None] = None,
    requires_grad: _bool = False,
) -> Composition:
    return Composition(torch.zeros([]))


def _from_replce(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    out = void()
    out._composition_tensor = composition_tensor
    if residual_tensor is None:
        residual_tensor = torch.zeros(composition_tensor.size()[1:]).to(
            composition_tensor
        )
    out._residual_tensor = residual_tensor
    return out


@_auto_registration
def cat(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    dim: _int = 0,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    c_tensors = tuple(c._composition_tensor for c in compositions)
    out_composition_tensor = torch.cat(
        c_tensors,
        _shift_dim(dim),
        out=out._composition_tensor if out is not None else None,
    )
    r_tensors = tuple(c._residual_tensor for c in compositions)
    out_residual_tensor = torch.cat(
        r_tensors, dim, out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def concat(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    dim: _int = 0,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    return cat(compositions=compositions, dim=dim, out=out)


def concatenate(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    dim: _int = 0,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    return cat(compositions=compositions, dim=dim, out=out)


def c_cat(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    *,
    sum_residual: _bool = False,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(compositions)):
        if compositions[i].size() != compositions[0].size():
            raise arg_value_error(
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for component number {i} in the list."
            )

    c_tensors = tuple(c._composition_tensor for c in compositions)
    out_composition_tensor = torch.cat(
        c_tensors, 0, out=out._composition_tensor if out is not None else None,
    )
    out_residual_tensor = None
    if sum_residual:
        r_tensors = tuple(c._residual_tensor for c in compositions)
        out_residual_tensor = builtins.sum(r_tensors)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def stack(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    dim: _int = 0,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(compositions)):
        if compositions[i].size() != compositions[0].size():
            raise arg_value_error(
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for component number {i} in the list."
            )
    c_tensors = tuple(c._composition_tensor for c in compositions)
    out_composition_tensor = torch.stack(
        c_tensors,
        _shift_dim(dim),
        out=out._composition_tensor if out is not None else None,
    )
    r_tensors = tuple(c._residual_tensor for c in compositions)
    out_residual_tensor = torch.stack(
        r_tensors, dim, out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


def c_stack(
    components: Union[Tuple[Tensor, ...], List[Tensor]],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(components)):
        if components[i].size() != components[0].size():
            raise arg_value_error(
                f"Sizes of components must match. Expected size [{components[0].size()}] but got size [{components[i].size()}] for component number {i} in the list."
            )

    out_composition_tensor = torch.stack(
        components, 0, out=out._composition_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor)


def diagonal_init(
    input: Composition, src: Tensor, dim: _int, offset: _int = 0
) -> Composition:
    permute_dims = list(range(src.dim()))
    dim = (dim + src.dim()) % src.dim()  # Converted to a positive number
    permute_dims.remove(dim)
    permute_dims.append(dim)
    src = src.permute(permute_dims)
    if (
        torch.__version__ < "1.11.0"
    ):  # for versions < 1.11.0, 'diagonal_scatter' does not exist.
        out_composition_tensor = input._composition_tensor.clone()
        diag_view = out_composition_tensor.diagonal(
            offset=offset, dim1=0, dim2=_shift_dim(dim)
        )
        diag_view = src
    else:
        out_composition_tensor = input._composition_tensor.diagonal_scatter(
            src, offset=offset, dim1=0, dim2=_shift_dim(dim)
        )
    out_residual_tensor = input._residual_tensor.clone()
    return _from_replce(out_composition_tensor, out_residual_tensor)


def c_apply(input: Composition, callable: Callable[..., Tensor]) -> Composition:
    out_composition_tensor = callable(input._composition_tensor)
    out_residual_tensor = callable(input._residual_tensor)
    return _from_replce(out_composition_tensor, out_residual_tensor)


def c_map(
    input, composition: Composition, callable: Callable[..., Tensor]
) -> Composition:
    out_composition_tensor = callable(
        input._composition_tensor, composition._composition_tensor
    )
    out_residual_tensor = callable(input._residual_tensor, composition._residual_tensor)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def numel(input: Composition) -> _int:
    return torch.numel(input._residual_tensor)


def c_numel(input: Composition, count_residual=False) -> _int:
    if count_residual:
        return input._composition_tensor.numel() + input._residual_tensor.numel()
    else:
        return input._composition_tensor.numel()


def numc(input: Composition) -> _int:
    return len(input)


@_auto_registration
def clone(
    input: Composition, *, memory_format: Optional[memory_format] = None
) -> Composition:
    out_composition_tensor = torch.clone(
        input._composition_tensor, memory_format=memory_format
    )
    out_residual_tensor = torch.clone(
        input._residual_tensor, memory_format=memory_format
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def detach(input: Composition) -> Composition:
    out_composition_tensor = torch.detach(input._composition_tensor)
    out_residual_tensor = torch.detach(input._residual_tensor)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def detach_(input: Composition) -> Composition:
    torch.detach_(input._composition_tensor)
    torch.detach_(input._residual_tensor)
    return input


@_auto_registration
def add(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(other, Composition):
        if input.numc() != other.numc():
            raise component_num_error(input.numc(), other.numc())
        if out is None:
            out_composition_tensor = input._composition_tensor.add(
                other._composition_tensor, alpha=alpha
            )
            out_residual_tensor = input._residual_tensor.add(
                other._residual_tensor, alpha=alpha
            )
        else:
            out_composition_tensor = input._composition_tensor.add(
                other._composition_tensor, alpha=alpha, out=out._composition_tensor
            )
            out_residual_tensor = input._residual_tensor.add(
                other._residual_tensor, alpha=alpha, out=out._residual_tensor
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)
    elif isinstance(other, (_int, _float, _bool, Tensor)):
        out_composition_tensor = input._composition_tensor.clone()
        out_residual_tensor = input._residual_tensor.add(
            other,
            alpha=alpha,
            # TODO out=out._residual_tensor if out is not None else None,
        )
        if out is not None:
            out._composition_tensor[:] = out_composition_tensor
        return _from_replce(out_composition_tensor, out_residual_tensor)
    else:
        raise unsupported_operand_error("add", type(input), type(other))


@_auto_registration
def sub(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    return add(-other, alpha=alpha, out=out)


@overload
def subtract(
    input: Composition,
    other: Tensor,
    *,
    alpha: Number = 1,
    out: Optional[Tensor] = None,
) -> Composition:
    ...


@overload
def subtract(input: Composition, other: Number, alpha: Number = 1) -> Composition:
    ...


@_auto_registration
def subtract(
    input: Composition, other: Any, *, alpha: Number = 1, out: Optional[Tensor] = None,
) -> Composition:
    return sub(input, other=other, alpha=alpha, out=out)


@_auto_registration
def mul(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(other, Composition):
        raise args_error(Composition.mul.__name__, input, other, out=out)
    if isinstance(other, Tensor):
        if other.dim() > input.dim():
            new_size = (
                (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
            )
            out_composition_tensor = input._composition_tensor.view(new_size).mul(
                other, out=out._composition_tensor
            )
        else:
            out_composition_tensor = input._composition_tensor.mul(
                other, out=out._composition_tensor
            )
        out_residual_tensor = input._residual_tensor.mul(
            other, out=out._residual_tensor
        )
    else:
        out_composition_tensor = input._composition_tensor * other
        out_residual_tensor = input._residual_tensor * other
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def multiply(
    input: Composition, other: Tensor, *, out: Optional[Tensor] = None
) -> Composition:
    ...


@overload
def multiply(input: Composition, other: Number) -> Composition:
    ...


@_auto_registration
def multiply(
    input: Composition, other: Any, *, out: Optional[Tensor] = None
) -> Composition:
    return mul(input, other=other, out=out)


@_auto_registration
def div(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    rounding_mode: Optional[str] = None,
) -> Tensor:
    if isinstance(other, Composition):
        raise args_error(
            Composition.div.__name__, input, other, rounding_mode=rounding_mode
        )
    if isinstance(other, Tensor):
        if other.dim() > input.dim():
            new_size = (
                (input.numc(),) + (1,) * (other.dim() - input.dim()) + input.size()
            )
            out_composition_tensor = input._composition_tensor.view(new_size).div(
                other, rounding_mode=rounding_mode
            )
        else:
            out_composition_tensor = input._composition_tensor.div(
                other, rounding_mode == rounding_mode
            )
        out_residual_tensor = input._residual_tensor.div(
            other, rounding_mode=rounding_mode
        )
    else:
        out_composition_tensor = input._composition_tensor.div(
            other, rounding_mode=rounding_mode
        )
        out_residual_tensor = input._residual_tensor.div(
            other, rounding_mode=rounding_mode
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def divide(input: Composition, other: Tensor,) -> Composition:
    ...


@overload
def divide(
    input: Composition, other: Tensor, *, rounding_mode: Optional[str],
) -> Composition:
    ...


@overload
def divide(
    input: Composition, other: Number, *, rounding_mode: Optional[str]
) -> Composition:
    ...


@overload
def divide(input: Composition, other: Number,) -> Composition:
    ...


@_auto_registration
def divide(input: Composition, other: Any, *, rounding_mode: Optional[str]):
    return div(input, other=other, rounding_mode=rounding_mode)


@_auto_registration
def mv(
    input: Union[Composition, Tensor],
    vec: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(input, Composition) and isinstance(vec, Composition):
        raise TypeError(
            "mv(): argument 'input' and argument 'vec' cannot both be Composition"
        )
    if isinstance(input, Composition):
        out_residual_tensor = torch.mv(
            input._residual_tensor,
            vec,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input._composition_tensor,
            vec,
            out=out._composition_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.mv(
            input,
            vec._residual_tensor,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input,
            vec._composition_tensor,
            out=out._composition_tensor if out is not None else None,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def mm(
    input: Union[Composition, Tensor],
    mat2: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if isinstance(input, Composition) and isinstance(mat2, Composition):
        raise TypeError(
            "mm(): argument 'input' and argument 'mat2' cannot both be Composition"
        )
    if isinstance(input, Composition):
        out_residual_tensor = torch.mm(
            input._residual_tensor,
            mat2,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input._composition_tensor,
            mat2,
            out=out._composition_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.mm(
            input,
            mat2._residual_tensor,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input,
            mat2._composition_tensor,
            out=out._composition_tensor if out is not None else None,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def bmm(
    input: Union[Composition, Tensor],
    mat2: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Tensor:
    if isinstance(input, Composition) and isinstance(mat2, Composition):
        raise TypeError(
            "bmm(): argument 'input' and argument 'mat2' cannot both be Composition"
        )
    if isinstance(input, Composition):
        out_residual_tensor = torch.bmm(
            input._residual_tensor,
            mat2,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input._composition_tensor,
            mat2,
            out=out._composition_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.bmm(
            input,
            mat2._residual_tensor,
            out=out._residual_tensor if out is not None else None,
        )
        out_composition_tensor = torch.matmul(
            input,
            mat2._composition_tensor,
            out=out._composition_tensor if out is not None else None,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def any(input: Composition, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def any(
    input: Composition,
    dim: _int,
    keepdim: _bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    ...


@_auto_registration
def any(input: Composition, *args: Any, **kwargs: Any) -> Tensor:
    return torch.any(input.c_sum(), *args, **kwargs)


@overload
def all(input: Composition, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def all(
    input: Composition,
    dim: _int,
    keepdim: _bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    ...


@_auto_registration
def all(input: Composition, *args: Any, **kwargs: Any) -> Tensor:
    return torch.all(input.c_sum(), *args, **kwargs)


@_auto_registration
def unsqueeze(input: Composition, dim: _int) -> Composition:
    out_residual_tensor = input._residual_tensor.unsqueeze(dim)
    out_composition_tensor = input._composition_tensor.unsqueeze(_shift_dim(dim))
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def squeeze(input: Composition) -> Composition:
    ...


@overload
def squeeze(input: Composition, dim: _int) -> Composition:
    ...


@_auto_registration
def squeeze(input: Composition, dim: _int = None) -> Composition:
    if dim is None:
        out_residual_tensor = input._residual_tensor.squeeze()
        out_composition_tensor = input._composition_tensor.squeeze()
        if input.numc() == 1:
            out_composition_tensor = out_composition_tensor.unsqueeze(0)
    else:
        out_residual_tensor = input._residual_tensor.squeeze(dim)
        out_composition_tensor = input._composition_tensor.squeeze(_shift_dim(dim))
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def transpose(input: Composition, dim0: _int, dim1: _int) -> Composition:
    out_residual_tensor = input._residual_tensor.transpose(dim0, dim1)
    out_composition_tensor = input._composition_tensor.transpose(
        _shift_dim(dim0), _shift_dim(dim1)
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def permute(input: Composition, dims: _size) -> Composition:
    out_residual_tensor = input._residual_tensor.permute(dims)
    out_composition_tensor = input._composition_tensor.permute((0,) + _shift_dims(dims))
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def sum(input: Composition, *, dtype: Optional[_dtype] = None) -> Composition:
    ...


@overload
def sum(
    input: Composition,
    dim: Union[_int, _size],
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    ...


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def sum(
    input: Composition,
    dim=None,
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    if dim is None:
        dim = tuple(range(1, input._composition_tensor.dim()))
        out_composition_tensor = torch.sum(
            input._composition_tensor,
            dim=dim,
            dtype=dtype,
            out=out._composition_tensor if out is not None else None,
        )
        out_residual_tensor = torch.sum(
            input._residual_tensor,
            dtype=dtype,
            out=out._residual_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.sum(
            input._residual_tensor,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            out=out._residual_tensor if out is not None else None,
        )
        if isinstance(dim, _int):
            dim = (dim,)
        out_composition_tensor = torch.sum(
            input._composition_tensor,
            dim=_shift_dims(dim),
            keepdim=keepdim,
            dtype=dtype,
            out=out._composition_tensor if out is not None else None,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


def c_sum(input: Composition, *, dtype: Optional[_dtype] = None) -> Tensor:
    return input._composition_tensor.sum(
        dim=0, dtype=dtype
    ) + input._residual_tensor.to(dtype=dtype)


@overload
def mean(input: Composition, *, dtype: Optional[_dtype] = None) -> Composition:
    ...


@overload
def mean(
    input: Composition,
    dim: Union[_int, _size],
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    ...


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def mean(
    input: Composition,
    dim: Union[_int, _size] = None,
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        raise arg_value_error(
            f"{mean.__name__}() dees not support keyword 'out' currently."
        )
    if dim is None:
        dim = tuple(range(1, input._composition_tensor.dim()))
        out_composition_tensor = torch.mean(
            input._composition_tensor,
            dim=dim,
            dtype=dtype,
            out=out._composition_tensor if out is not None else None,
        )
        out_residual_tensor = torch.mean(
            input._residual_tensor,
            dtype=dtype,
            out=out._residual_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.mean(
            input._residual_tensor,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            out=out._residual_tensor if out is not None else None,
        )
        if isinstance(dim, _int):
            dim = (dim,)
        out_composition_tensor = torch.mean(
            input._composition_tensor,
            dim=_shift_dims(dim),
            keepdim=keepdim,
            dtype=dtype,
            out=out._composition_tensor if out is not None else None,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def reshape(input: Composition, shape: _size) -> Composition:
    out_composition_tensor = input._composition_tensor.view((input.numc(),) + shape)
    out_residual_tensor = input._residual_tensor.view(shape)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def masked_fill(input: Composition, mask: Tensor, value: Tensor) -> Composition:
    ...


@overload
def masked_fill(input: Composition, mask: Tensor, value: Number) -> Composition:
    ...


@_auto_registration
def masked_fill(input: Composition, mask: Tensor, value: Any) -> Composition:
    out_composition_tensor = input._composition_tensor.masked_fill(mask[None], value)
    out_residual_tensor = input._residual_tensor.masked_fill(mask, value)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def c_masked_fill(input: Composition, mask: Tensor, value: Tensor) -> Composition:
    ...


@overload
def c_masked_fill(input: Composition, mask: Tensor, value: Number) -> Composition:
    ...


def c_masked_fill(input: Composition, mask: Tensor, value: Any) -> Composition:
    if mask.dim() == 1:
        if len(mask) != input.numc():
            raise arg_value_error(
                f"the length of mask ({len(mask)}) should match component number ({input.numc()})"
            )
        mask_size = (input.numc(),) + (1,) * input.dim()
        mask = mask.view(mask_size)
    out_composition_tensor = input._composition_tensor.masked_fill(mask, value)
    out_residual_tensor = input._residual_tensor.clone()
    return _from_replce(out_composition_tensor, out_residual_tensor)


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def masked_select(
    input: Composition, mask: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_composition_tensor = torch.masked_select(
        input._composition_tensor,
        mask[None],
        out=out._composition_tensor if out is not None else None,
    ).reshape(input.numc(), -1)
    if out is not None:
        out._composition_tensor = out._composition_tensor.reshape(input.numc(), -1)
    out_residual_tensor = torch.masked_select(
        input._residual_tensor,
        mask,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def masked_scatter(input: Composition, mask: Tensor, source: Tensor) -> Composition:
    out_composition_tensor = input._composition_tensor.masked_scatter(
        mask[None], source
    )
    out_residual_tensor = input._residual_tensor.masked_scatter(mask, source)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def gather(
    input: Composition,
    dim: _int,
    index: Tensor,
    *,
    sparse_grad: _bool = False,
    out: Optional[Composition] = None,
) -> Composition:
    ...


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def gather(
    input: Composition,
    dim: Any,
    index: Tensor,
    *,
    sparse_grad: _bool = False,
    out: Optional[Composition] = None,
) -> Composition:
    c_index = index[None].expand((input.numc(),) + (-1,) * index.dim())
    out_composition_tensor = torch.gather(
        input._composition_tensor,
        _shift_dim(dim),
        c_index,
        sparse_grad=sparse_grad,
        out=out._composition_tensor if out is not None else None,
    )
    out_residual_tensor = torch.gather(
        input._residual_tensor,
        dim,
        index,
        sparse_grad=sparse_grad,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def scatter(
    input: Composition,
    dim: _int,
    index: Tensor,
    src: Tensor,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def scatter(
    input: Composition,
    dim: _int,
    index: Tensor,
    src: Tensor,
    *,
    reduce: str,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def scatter(
    input: Composition,
    dim: _int,
    index: Tensor,
    value: Number,
    *,
    reduce: str,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def scatter(
    input: Composition,
    dim: _int,
    index: Tensor,
    value: Number,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def scatter(
    input: Composition,
    dim: Any,
    index: Tensor,
    src: Any = None,
    value: Any = None,
    *,
    reduce: str = None,
    out: Optional[Composition] = None,
) -> Composition:
    r"""
    Unsafe.
    Safe when reduce is not None.
    """
    if src is None:
        src = value
    if reduce == "add":
        holder = torch.zeros_like(input._residual_tensor).to(input._residual_tensor)
        holder = holder.scatter(dim, index, src, reduce=reduce)
        c_out = input + holder
        if out is not None:
            # TODO: use the out argument of `torch.add` raises an error
            out._composition_tensor[:] = c_out._composition_tensor
            out._residual_tensor[:] = c_out._residual_tensor
        return c_out
    else:
        c_index = index[None].expand((input.numc(),) + (-1,) * index.dim())
        if isinstance(src, Tensor):
            c_src = src[None].expand((input.numc(),) + (-1,) * src.dim())
        else:
            c_src = src
        if reduce is None:
            out_composition_tensor = torch.scatter(
                input._composition_tensor,
                _shift_dim(dim),
                c_index,
                c_src,
                out=out._composition_tensor if out is not None else None,
            )
            out_residual_tensor = torch.scatter(
                input._residual_tensor,
                dim,
                index,
                src,
                out=out._residual_tensor if out is not None else None,
            )
        else:
            out_composition_tensor = torch.scatter(
                input._composition_tensor,
                _shift_dim(dim),
                c_index,
                c_src,
                reduce=reduce,
                out=out._composition_tensor if out is not None else None,
            )
            out_residual_tensor = torch.scatter(
                input._residual_tensor,
                dim,
                index,
                src,
                reduce=reduce,
                out=out._residual_tensor if out is not None else None,
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def diagonal_scatter(
    input: Composition, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1
) -> Composition:
    if (
        torch.__version__ < "1.11.0"
    ):  # for versions < 1.11.0, 'diagonal_scatter' does not exist.
        raise RuntimeError("`diagonal_scatter` requires a torch version >= 1.11.0.")
    c_src = src[None].expand((input.numc(),) + (-1,) * src.dim())
    out_composition_tensor = input._composition_tensor.diagonal_scatter(
        c_src, offset, _shift_dim(dim1), _shift_dim(dim2)
    )
    out_residual_tensor = input._residual_tensor.diagonal_scatter(
        src, offset, dim1, dim2
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def index_select(
    input: Composition, dim: _int, index: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    ...


@_auto_registration
def index_select(
    input: Composition, dim: _int, index: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_composition_tensor = torch.index_select(
        input._composition_tensor,
        dim=_shift_dim(dim),
        index=index,
        out=out._composition_tensor if out is not None else None,
    )
    out_residual_tensor = torch.index_select(
        input._residual_tensor,
        dim=dim,
        index=index,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def c_index_select(
    input: Composition,
    index: Tensor,
    with_residual: _bool = True,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


def c_index_select(
    input: Composition,
    index: Tensor,
    with_residual: _bool = True,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    out_composition_tensor = torch.index_select(
        input._composition_tensor,
        dim=0,
        index=index,
        out=out._composition_tensor if out is not None else None,
    )
    if with_residual:
        if out is not None:
            out._residual_tensor = out._residual_tensor.reshape_as(
                input._residual_tensor
            )
            out._residual_tensor[:] = input._residual_tensor
        out_residual_tensor = input._residual_tensor.clone()
    else:
        out_residual_tensor = torch.zeros_like(input._residual_tensor).to(
            input._residual_tensor
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def masked_select(
    input: Composition, mask: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_composition_tensor = torch.masked_select(
        input._composition_tensor,
        mask=mask[None],
        out=out._composition_tensor if out is not None else None,
    ).reshape(input.numc(), -1)
    out_residual_tensor = torch.masked_select(
        input._residual_tensor,
        mask=mask,
        out=out._residual_tensor if out is not None else None,
    )
    if out is not None:
        out._composition_tensor = out._composition_tensor.reshape(input.numc(), -1)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def index_fill(
    input: Composition, dim: _int, index: Tensor, value: Tensor
) -> Composition:
    ...


@overload
def index_fill(
    input: Composition, dim: _int, index: Tensor, value: Number
) -> Composition:
    ...


@_auto_registration
def index_fill(input: Composition, dim: _int, index: Tensor, value: Any) -> Composition:
    out_composition_tensor = input._composition_tensor.index_fill(
        dim=_shift_dim(dim), index=index, value=value
    )
    out_residual_tensor = input._residual_tensor.index_fill(
        dim=dim, index=index, value=value
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def round(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    ...


@overload
def round(
    input: Composition, *, decimals: _int, out: Optional[Composition] = None
) -> Composition:
    ...


@_auto_registration
def round(
    input: Composition, *, decimals: _int = None, out: Optional[Composition] = None
) -> Composition:
    if decimals is not None:
        out_composition_tensor = torch.round(
            input._composition_tensor,
            decimals=decimals,
            out=out._composition_tensor if out is not None else None,
        )
        out_residual_tensor = torch.round(
            input._residual_tensor,
            decimals=decimals,
            out=out._residual_tensor if out is not None else None,
        )
    else:
        out_composition_tensor = torch.round(
            input._composition_tensor, decimals=decimals,
        )
        out_residual_tensor = torch.round(input._residual_tensor, decimals=decimals,)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def round_(input: Composition) -> Composition:
    ...


@overload
def round_(input: Composition, *, decimals: _int) -> Composition:
    ...


@_auto_registration
def round_(input: Composition, *, decimals: _int = None) -> Composition:
    if decimals is not None:
        input._composition_tensor.round_(decimals=decimals)
        input._residual_tensor.round_(decimals=decimals)
    else:
        input._composition_tensor.round_()
        input._residual_tensor.round_()
    return input


@overload
def zeros(
    size: _size,
    c_num: _int,
    *,
    out: Optional[Tensor] = None,
    dtype: _dtype = None,
    layout: Optional[_layout] = strided,
    device: Union[_device, str, None] = None,
    pin_memory: _bool = False,
    requires_grad: _bool = False,
) -> Composition:
    ...


@overload
def zeros(
    *size: _int,
    c_num: _int,
    out: Optional[Tensor] = None,
    dtype: _dtype = None,
    layout: Optional[_layout] = strided,
    device: Union[_device, str, None] = None,
    pin_memory: _bool = False,
    requires_grad: _bool = False,
) -> Composition:
    ...


def zeros(*args: Any, **kwargs: Any):
    def _zeros(
        *,
        size: _size,
        c_num: _int,
        out: Optional[Composition] = None,
        dtype: _dtype = None,
        layout: Optional[_layout] = strided,
        device: Union[_device, str, None] = None,
        pin_memory: _bool = False,
        requires_grad: _bool = False,
    ):
        out_composition_tensor = torch.zeros(
            (c_num,) + size,
            out=out._composition_tensor if out is not None else None,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        out_residual_tensor = torch.zeros(
            size,
            out=out._residual_tensor if out is not None else None,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    # parse args
    if len(args) > 0:
        if isinstance(args[0], _int):
            kwargs["size"] = args
        else:
            for i in range(len(args)):
                kwargs[["size", "c_num"][i]] = args[i]
    return _zeros(**kwargs)


@_auto_registration
def zeros_like(
    input: Composition,
    *,
    memory_format: Optional[memory_format] = None,
    dtype: _dtype = None,
    layout: Optional[_layout] = strided,
    device: Union[_device, str, None] = None,
    pin_memory: _bool = False,
    requires_grad: _bool = False,
) -> Composition:
    # TODO: fix bug. Default: if None, defaults to the dtype of input.
    out_composition_tensor = torch.zeros_like(
        input._composition_tensor,
        memory_format=memory_format,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    out_residual_tensor = torch.zeros_like(
        input._residual_tensor,
        memory_format=memory_format,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def abs(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    out_composition_tensor = torch.abs(
        input._composition_tensor, out=out._composition_tensor
    )
    out_residual_tensor = torch.abs(input._residual_tensor, out=out._residual_tensor)
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def abs_(input: Composition) -> Composition:
    torch.abs_(input._composition_tensor)
    torch.abs_(input._residual_tensor)
    return input


@_auto_registration
def relu(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        # TODO: inplace arg overwrite
        out = decomposition_func(input=input, func=torch.nn.functional.relu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def relu_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        # TODO: inplace arg overwrite
        out = decomposition_func(
            input=input, func=torch.nn.functional.relu_, inplace=True, ref=ref
        )
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def tanh(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.tanh, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def tanh_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.tanh_, inplace=True, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def sigmoid(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.sigmoid, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def sigmoid_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.sigmoid_, inplace=True, ref=ref
        )
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())

