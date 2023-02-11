import torch
import pydec
from torch import Tensor
from torch.nn.parameter import Parameter
from torch._C import memory_format

from pydec._composition import Composition, IndexComposition
from .overrides import _auto_registration, _register_builtin_function

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


def void() -> Composition:
    return Composition()


def _from_replce(
    component_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    out = void()
    out._component_tensor = component_tensor
    if residual_tensor is None:
        residual_tensor = torch.zeros(component_tensor.size()[1:]).to(component_tensor)
    out._residual_tensor = residual_tensor
    return out


@_auto_registration
def cat(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    dim: _int = 0,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    c_tensors = tuple(c._component_tensor for c in compositions)
    out_component_tensor = torch.cat(
        c_tensors,
        _shift_dim(dim),
        out=out._component_tensor if out is not None else None,
    )
    r_tensors = tuple(c._residual_tensor for c in compositions)
    out_residual_tensor = torch.cat(
        r_tensors, dim, out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    sum_residual: _bool = True,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(compositions)):
        if compositions[i].size() != compositions[0].size():
            raise arg_value_error(
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for component number {i} in the list"
            )

    c_tensors = tuple(c._component_tensor for c in compositions)
    out_component_tensor = torch.cat(
        c_tensors, 0, out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = None
    if sum_residual:
        r_tensors = tuple(c._residual_tensor for c in compositions)
        out_residual_tensor = builtins.sum(r_tensors)
    return _from_replce(out_component_tensor, out_residual_tensor)


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
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for component number {i} in the list"
            )
    c_tensors = tuple(c._component_tensor for c in compositions)
    out_component_tensor = torch.stack(
        c_tensors,
        _shift_dim(dim),
        out=out._component_tensor if out is not None else None,
    )
    r_tensors = tuple(c._residual_tensor for c in compositions)
    out_residual_tensor = torch.stack(
        r_tensors, dim, out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


def c_stack(
    components: Union[Tuple[Tensor, ...], List[Tensor]],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(components)):
        if components[i].size() != components[0].size():
            raise arg_value_error(
                f"Sizes of components must match. Expected size [{components[0].size()}] but got size [{components[i].size()}] for component number {i} in the list"
            )

    out_component_tensor = torch.stack(
        components, 0, out=out._component_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor)


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
        out_component_tensor = input._component_tensor.clone()
        diag_view = out_component_tensor.diagonal(
            offset=offset, dim1=0, dim2=_shift_dim(dim)
        )
        diag_view[:] = src
    else:
        out_component_tensor = input._component_tensor.diagonal_scatter(
            src, offset=offset, dim1=0, dim2=_shift_dim(dim)
        )
    out_residual_tensor = input._residual_tensor.clone()

    return _from_replce(out_component_tensor, out_residual_tensor)


def c_apply(input: Composition, callable: Callable[..., Tensor]) -> Composition:
    out_component_tensor = callable(input._component_tensor)
    out_residual_tensor = callable(input._residual_tensor)
    return _from_replce(out_component_tensor, out_residual_tensor)


def c_map(
    input, composition: Composition, callable: Callable[..., Tensor]
) -> Composition:
    out_component_tensor = callable(
        input._component_tensor, composition._component_tensor
    )
    out_residual_tensor = callable(input._residual_tensor, composition._residual_tensor)
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def numel(input: Composition) -> _int:
    return torch.numel(input._residual_tensor)


def c_numel(input: Composition, count_residual=False) -> _int:
    if count_residual:
        return input._component_tensor.numel() + input._residual_tensor.numel()
    else:
        return input._component_tensor.numel()


def numc(input: Composition) -> _int:
    return len(input)


@_auto_registration
def clone(
    input: Composition, *, memory_format: Optional[memory_format] = None
) -> Composition:
    out_component_tensor = torch.clone(
        input._component_tensor, memory_format=memory_format
    )
    out_residual_tensor = torch.clone(
        input._residual_tensor, memory_format=memory_format
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def detach(input: Composition) -> Composition:
    out_component_tensor = torch.detach(input._component_tensor)
    out_residual_tensor = torch.detach(input._residual_tensor)
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def detach_(input: Composition) -> Composition:
    torch.detach_(input._component_tensor)
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
            out_component_tensor = input._component_tensor.add(
                other._component_tensor, alpha=alpha
            )
            out_residual_tensor = input._residual_tensor.add(
                other._residual_tensor, alpha=alpha
            )
        else:
            out_component_tensor = input._component_tensor.add(
                other._component_tensor, alpha=alpha, out=out._component_tensor
            )
            out_residual_tensor = input._residual_tensor.add(
                other._residual_tensor, alpha=alpha, out=out._residual_tensor
            )
        return _from_replce(out_component_tensor, out_residual_tensor)
    elif isinstance(other, (_int, _float, _bool, Tensor)):
        out_component_tensor = input._component_tensor.clone()
        out_residual_tensor = input._residual_tensor.add(
            other,
            alpha=alpha,
            # TODO out=out._residual_tensor if out is not None else None,
        )
        if out is not None:
            out._component_tensor[:] = out_component_tensor
        return _from_replce(out_component_tensor, out_residual_tensor)
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
            out_component_tensor = input._component_tensor.view(new_size).mul(
                other, out=out._component_tensor
            )
        else:
            out_component_tensor = input._component_tensor.mul(
                other, out=out._component_tensor
            )
        out_residual_tensor = input._residual_tensor.mul(
            other, out=out._residual_tensor
        )
    else:
        out_component_tensor = input._component_tensor * other
        out_residual_tensor = input._residual_tensor * other
    return _from_replce(out_component_tensor, out_residual_tensor)


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
            out_component_tensor = input._component_tensor.view(new_size).div(
                other, rounding_mode=rounding_mode
            )
        else:
            out_component_tensor = input._component_tensor.div(
                other, rounding_mode == rounding_mode
            )
        out_residual_tensor = input._residual_tensor.div(
            other, rounding_mode=rounding_mode
        )
    else:
        out_component_tensor = input._component_tensor.div(
            other, rounding_mode=rounding_mode
        )
        out_residual_tensor = input._residual_tensor.div(
            other, rounding_mode=rounding_mode
        )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        out_component_tensor = torch.matmul(
            input._component_tensor,
            vec,
            out=out._component_tensor if out is not None else None,
        )
    else:
        out_residual_tensor = torch.mv(
            input,
            vec._residual_tensor,
            out=out._residual_tensor if out is not None else None,
        )
        out_component_tensor = torch.matmul(
            input,
            vec._component_tensor,
            out=out._component_tensor if out is not None else None,
        )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        out_component_tensor = torch.matmul(
            input._component_tensor,
            mat2,
            out=out._component_tensor if out is not None else None,
        )
    else:
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
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        out_component_tensor = torch.matmul(
            input._component_tensor,
            mat2,
            out=out._component_tensor if out is not None else None,
        )
    else:
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
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    out_component_tensor = input._component_tensor.unsqueeze(_shift_dim(dim))
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        out_component_tensor = input._component_tensor.squeeze()
        if input.numc() == 1:
            out_component_tensor = out_component_tensor.unsqueeze(0)
    else:
        out_residual_tensor = input._residual_tensor.squeeze(dim)
        out_component_tensor = input._component_tensor.squeeze(_shift_dim(dim))
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def transpose(input: Composition, dim0: _int, dim1: _int) -> Composition:
    out_residual_tensor = input._residual_tensor.transpose(dim0, dim1)
    out_component_tensor = input._component_tensor.transpose(
        _shift_dim(dim0), _shift_dim(dim1)
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def permute(input: Composition, dims: _size) -> Composition:
    out_residual_tensor = input._residual_tensor.permute(dims)
    out_component_tensor = input._component_tensor.permute((0,) + _shift_dims(dims))
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        dim = tuple(range(1, input._component_tensor.dim()))
        out_component_tensor = torch.sum(
            input._component_tensor,
            dim=dim,
            dtype=dtype,
            out=out._component_tensor if out is not None else None,
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
        out_component_tensor = torch.sum(
            input._component_tensor,
            dim=_shift_dims(dim),
            keepdim=keepdim,
            dtype=dtype,
            out=out._component_tensor if out is not None else None,
        )
    return _from_replce(out_component_tensor, out_residual_tensor)


def c_sum(input: Composition, *, dtype: Optional[_dtype] = None) -> Tensor:
    return input._component_tensor.sum(dim=0, dtype=dtype) + input._residual_tensor.to(
        dtype=dtype
    )


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
            f"{mean.__name__}() dees not support keyword 'out' currently"
        )
    if dim is None:
        dim = tuple(range(1, input._component_tensor.dim()))
        out_component_tensor = torch.mean(
            input._component_tensor,
            dim=dim,
            dtype=dtype,
            out=out._component_tensor if out is not None else None,
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
        out_component_tensor = torch.mean(
            input._component_tensor,
            dim=_shift_dims(dim),
            keepdim=keepdim,
            dtype=dtype,
            out=out._component_tensor if out is not None else None,
        )
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def reshape(input: Composition, shape: _size) -> Composition:
    out_component_tensor = input._component_tensor.view((input.numc(),) + shape)
    out_residual_tensor = input._residual_tensor.view(shape)
    return _from_replce(out_component_tensor, out_residual_tensor)


@overload
def masked_fill(input: Composition, mask: Tensor, value: Tensor) -> Composition:
    ...


@overload
def masked_fill(input: Composition, mask: Tensor, value: Number) -> Composition:
    ...


@_auto_registration
def masked_fill(input: Composition, mask: Tensor, value: Any) -> Composition:
    out_component_tensor = input._component_tensor.masked_fill(mask[None], value)
    out_residual_tensor = input._residual_tensor.masked_fill(mask, value)
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    out_component_tensor = input._component_tensor.masked_fill(mask, value)
    out_residual_tensor = input._residual_tensor.clone()
    return _from_replce(out_component_tensor, out_residual_tensor)


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def masked_select(
    input: Composition, mask: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_component_tensor = torch.masked_select(
        input._component_tensor,
        mask[None],
        out=out._component_tensor if out is not None else None,
    ).reshape(input.numc(), -1)
    if out is not None:
        out._component_tensor = out._component_tensor.reshape(input.numc(), -1)
    out_residual_tensor = torch.masked_select(
        input._residual_tensor,
        mask,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def masked_scatter(input: Composition, mask: Tensor, source: Tensor) -> Composition:
    out_component_tensor = input._component_tensor.masked_scatter(mask[None], source)
    out_residual_tensor = input._residual_tensor.masked_scatter(mask, source)
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    out_component_tensor = torch.gather(
        input._component_tensor,
        _shift_dim(dim),
        c_index,
        sparse_grad=sparse_grad,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.gather(
        input._residual_tensor,
        dim,
        index,
        sparse_grad=sparse_grad,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
            out._component_tensor[:] = c_out._component_tensor
            out._residual_tensor[:] = c_out._residual_tensor
        return c_out
    else:
        c_index = index[None].expand((input.numc(),) + (-1,) * index.dim())
        if isinstance(src, Tensor):
            c_src = src[None].expand((input.numc(),) + (-1,) * src.dim())
        else:
            c_src = src
        if reduce is None:
            out_component_tensor = torch.scatter(
                input._component_tensor,
                _shift_dim(dim),
                c_index,
                c_src,
                out=out._component_tensor if out is not None else None,
            )
            out_residual_tensor = torch.scatter(
                input._residual_tensor,
                dim,
                index,
                src,
                out=out._residual_tensor if out is not None else None,
            )
        else:
            out_component_tensor = torch.scatter(
                input._component_tensor,
                _shift_dim(dim),
                c_index,
                c_src,
                reduce=reduce,
                out=out._component_tensor if out is not None else None,
            )
            out_residual_tensor = torch.scatter(
                input._residual_tensor,
                dim,
                index,
                src,
                reduce=reduce,
                out=out._residual_tensor if out is not None else None,
            )
        return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def diagonal_scatter(
    input: Composition, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1
) -> Composition:
    if (
        torch.__version__ < "1.11.0"
    ):  # for versions < 1.11.0, 'diagonal_scatter' does not exist.
        raise RuntimeError("`diagonal_scatter` requires a torch version >= 1.11.0")
    c_src = src[None].expand((input.numc(),) + (-1,) * src.dim())
    out_component_tensor = input._component_tensor.diagonal_scatter(
        c_src, offset, _shift_dim(dim1), _shift_dim(dim2)
    )
    out_residual_tensor = input._residual_tensor.diagonal_scatter(
        src, offset, dim1, dim2
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


@overload
def index_select(
    input: Composition, dim: _int, index: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    ...


@_auto_registration
def index_select(
    input: Composition, dim: _int, index: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_component_tensor = torch.index_select(
        input._component_tensor,
        dim=_shift_dim(dim),
        index=index,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.index_select(
        input._residual_tensor,
        dim=dim,
        index=index,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    out_component_tensor = torch.index_select(
        input._component_tensor,
        dim=0,
        index=index,
        out=out._component_tensor if out is not None else None,
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
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def masked_select(
    input: Composition, mask: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    out_component_tensor = torch.masked_select(
        input._component_tensor,
        mask=mask[None],
        out=out._component_tensor if out is not None else None,
    ).reshape(input.numc(), -1)
    out_residual_tensor = torch.masked_select(
        input._residual_tensor,
        mask=mask,
        out=out._residual_tensor if out is not None else None,
    )
    if out is not None:
        out._component_tensor = out._component_tensor.reshape(input.numc(), -1)
    return _from_replce(out_component_tensor, out_residual_tensor)


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
    out_component_tensor = input._component_tensor.index_fill(
        dim=_shift_dim(dim), index=index, value=value
    )
    out_residual_tensor = input._residual_tensor.index_fill(
        dim=dim, index=index, value=value
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


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
        out_component_tensor = torch.round(
            input._component_tensor,
            decimals=decimals,
            out=out._component_tensor if out is not None else None,
        )
        out_residual_tensor = torch.round(
            input._residual_tensor,
            decimals=decimals,
            out=out._residual_tensor if out is not None else None,
        )
    else:
        out_component_tensor = torch.round(input._component_tensor, decimals=decimals,)
        out_residual_tensor = torch.round(input._residual_tensor, decimals=decimals,)
    return _from_replce(out_component_tensor, out_residual_tensor)


@overload
def round_(input: Composition) -> Composition:
    ...


@overload
def round_(input: Composition, *, decimals: _int) -> Composition:
    ...


@_auto_registration
def round_(input: Composition, *, decimals: _int = None) -> Composition:
    if decimals is not None:
        input._component_tensor.round_(decimals=decimals)
        input._residual_tensor.round_(decimals=decimals)
    else:
        input._component_tensor.round_()
        input._residual_tensor.round_()
    return input


@overload
def zeros(
    size: _size,
    c_num: _int,
    *,
    out: Optional[Composition] = None,
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
    out: Optional[Composition] = None,
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
        out_component_tensor = torch.zeros(
            (c_num,) + size,
            out=out._component_tensor if out is not None else None,
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
        return _from_replce(out_component_tensor, out_residual_tensor)

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
    out_component_tensor = torch.zeros_like(
        input._component_tensor,
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
    return _from_replce(out_component_tensor, out_residual_tensor)


def empty_index_composition(
    size: _size,
    c_num: _int,
    *,
    dtype: _dtype = torch.long,
    device: Union[_device, str, None] = None,
) -> IndexComposition:
    ...
    # TODO
    if dtype not in [torch.long, torch.int]:
        raise RuntimeError(
            "Expected argument 'dtype' to have one of the following scalar types: {}; but got {} instead".format(
                [torch.long, torch.int], dtype
            )
        )

    out_component_tensor = torch.zeros(size=(c_num,) + size, dtype=dtype, device=device)
    out_residual_tensor = torch.zeros(size=size, dtype=dtype, device=device)
    out_component_tensor[:] = IndexComposition.MASK_NUM
    out_residual_tensor[:] = IndexComposition.MASK_NUM
    return IndexComposition(out_component_tensor, out_residual_tensor)


@_auto_registration
def abs(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    out_component_tensor = torch.abs(
        input._component_tensor, out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.abs(
        input._residual_tensor, out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_component_tensor, out_residual_tensor)


@_auto_registration
def abs_(input: Composition) -> Composition:
    torch.abs_(input._component_tensor)
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


@_auto_registration
def _pack_padded_sequence(input: Composition, lengths: Tensor, batch_first: _bool):
    if batch_first:
        input = input.transpose(0, 1)
    packed_input = []
    batch_sizes = torch.zeros((input.size(0),), dtype=torch.long, device="cpu")
    pack_num = input.size(1)
    for i in range(input.size(0)):
        packed_input.append(input[i, :pack_num])
        batch_sizes[i] = pack_num
        pack_num -= (lengths == i + 1).sum().item()
    packed_input = pydec.cat(packed_input, dim=0)

    return packed_input, batch_sizes


@_auto_registration
def _pad_packed_sequence(
    packed_input: Composition,
    batch_sizes: Tensor,
    batch_first: _bool,
    padding_value: _float,
    max_seq_length: _int,
):
    padded_input = []
    batch_start = 0
    max_batch_size = batch_sizes[0]
    feature_size = packed_input.size(-1)
    lengths = torch.zeros(
        (max_batch_size,), dtype=torch.long, device=packed_input.device
    )
    for i in range(max_seq_length):
        batch_size = 0 if i >= len(batch_sizes) else batch_sizes[i]
        if batch_size > 0:
            padded_input.append(packed_input[batch_start : batch_start + batch_size])
            lengths[:batch_size] += 1
            batch_start += batch_size
        if max_batch_size - batch_size > 0:
            padding = pydec.zeros(
                (max_batch_size - batch_size, feature_size),
                c_num=packed_input.numc(),
                dtype=packed_input.dtype,
                device=packed_input.device,
            )
            if padding_value != 0:
                padding.residual[:] = padding_value
            padded_input.append(padding)
    padded_input = pydec.cat(padded_input, dim=0)
    padded_input = padded_input.view(-1, max_batch_size, feature_size)

    if batch_first:
        padded_input = padded_input.transpose_(0, 1)
    return padded_input, lengths


def _rnn_relu_cell(
    input: Composition,
    hx: Union[Tensor, Composition],
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter = None,
    bias_hh: Parameter = None,
    *,
    ref: Optional[Tensor] = None,
):
    return relu(
        pydec.nn.functional.linear(input, weight_ih, bias_ih)
        + pydec.nn.functional.linear(hx, weight_hh, bias_hh),
        ref=ref,
    )


def _rnn_relu_layer(
    input: Composition,
    batch_sizes: Optional[Tensor],
    hx: Tensor,
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter,
    bias_hh: Parameter,
    reverse: bool = False,
):
    """
    Could optimize performance by dropping completed samples
    """
    orig_hx = hx
    out = []
    out_hx = []
    batch_start = 0 if not reverse else input.size(0)
    if reverse:
        hx = None

    for i in (
        range(len(batch_sizes)) if not reverse else range(len(batch_sizes) - 1, -1, -1)
    ):
        batch_size = batch_sizes[i]
        if reverse:
            batch_size_new = batch_sizes[i]
            batch_size_old = 0 if i + 1 == len(batch_sizes) else batch_sizes[i + 1]
            if batch_size_new - batch_size_old > 0:
                if hx is None:
                    hx = orig_hx[batch_size_old:batch_size_new]
                else:
                    append_hx = pydec.zeros(
                        orig_hx[batch_size_old:batch_size_new].size(),
                        c_num=hx.numc(),
                        dtype=hx.dtype,
                        device=hx.device,
                    ).to(hx)
                    append_hx.residual[:] = orig_hx[batch_size_old:batch_size_new]
                    hx = pydec.cat([hx, append_hx], dim=0,)
        if not reverse:
            batch = input[batch_start : batch_start + batch_size]
            batch_start += batch_size
        else:
            batch = input[batch_start - batch_size : batch_start]
            batch_start -= batch_size
        hx = _rnn_relu_cell(batch, hx, weight_ih, weight_hh, bias_ih, bias_hh)
        out.append(hx)
        if not reverse:
            batch_size_new = batch_sizes[i + 1] if i + 1 < len(batch_sizes) else 0
            batch_size_old = batch_sizes[i]
            if batch_size_old - batch_size_new > 0:
                out_hx.append(hx[batch_size_new:batch_size_old])
                hx = hx[:batch_size_new]
    if reverse:
        out = out[::-1]
        out_hx = hx
    else:
        out_hx = out_hx[::-1]
        out_hx = pydec.cat(out_hx, dim=0)
    out = pydec.cat(out, dim=0)
    return out, out_hx


def _rnn_relu_packed(
    input: Composition,
    batch_sizes: Tensor,
    hx: Tensor,
    flat_weights: List[Parameter],
    bias: bool,
    num_layers: int,
    dropout: float,
    training: bool,
    bidirectional: bool,
):
    x = input
    weight_group_len = 2 * (2 if bias else 1) * (2 if bidirectional else 1)
    out_hx_list = []

    for i in range(num_layers):
        hx_layer, hx_layer_r = None, None
        weight_ih, weight_hh, bias_ih, bias_hh = None, None, None, None
        weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r = None, None, None, None
        weight_group = flat_weights[weight_group_len * i : weight_group_len * (i + 1)]

        weight_ih = weight_group[0]
        weight_hh = weight_group[1]
        if bias:
            bias_ih = weight_group[2]
            bias_hh = weight_group[3]
        if not bidirectional:
            hx_layer = hx[i]
            x, out_hx = _rnn_relu_layer(
                x, batch_sizes, hx_layer, weight_ih, weight_hh, bias_ih, bias_hh
            )
            out_hx_list.append(out_hx)
        else:
            weight_group = weight_group[len(weight_group) // 2 :]  # takes the back half
            hx_layer = hx[2 * i]
            hx_layer_r = hx[2 * i + 1]
            weight_ih_r = weight_group[0]
            weight_hh_r = weight_group[1]
            if bias:
                bias_ih_r = weight_group[2]
                bias_hh_r = weight_group[3]

            x_, out_hx = _rnn_relu_layer(
                x, batch_sizes, hx_layer, weight_ih, weight_hh, bias_ih, bias_hh
            )
            x_r, out_hx_r = _rnn_relu_layer(
                x,
                batch_sizes,
                hx_layer_r,
                weight_ih_r,
                weight_hh_r,
                bias_ih_r,
                bias_hh_r,
                True,
            )
            x = pydec.cat([x_, x_r], dim=-1)
            out_hx_list.append(out_hx)
            out_hx_list.append(out_hx_r)

    out = x
    out_hx = pydec.stack(out_hx_list, dim=0)
    return out, out_hx


@overload
def rnn_relu(
    input: Composition,
    batch_sizes: Tensor,
    hx: Tensor,
    flat_weights: List[Parameter],
    bias: bool,
    num_layers: int,
    dropout: float,
    training: bool,
    bidirectional: bool,
    /,
):
    ...


@overload
def rnn_relu(
    input: Composition,
    hx: Tensor,
    flat_weights: List[Parameter],
    bias: bool,
    num_layers: int,
    dropout: float,
    training: bool,
    bidirectional: bool,
    batch_first: bool,
    /,
):
    ...


@_auto_registration
def rnn_relu(*args,):
    if isinstance(args[3], bool):
        (
            input,
            hx,
            flat_weights,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        ) = args
        batch_sizes = None
        if batch_first:
            input: Composition = input.transpose(0, 1)
        seq_len = input.size(0)
        batch_size = input.size(1)
        batch_sizes = torch.tensor(
            [batch_size] * input.size(0), dtype=torch.long, device="cpu"
        )
        input = input.view(seq_len * batch_size, -1)
        out, out_hx = _rnn_relu_packed(
            input,
            batch_sizes,
            hx,
            flat_weights,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
        )
        out = out.view(seq_len, batch_size, -1)
        if batch_first:
            out.transpose_(0, 1)
        return out, out_hx
    else:
        (
            input,
            batch_sizes,
            hx,
            flat_weights,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
        ) = args
        return _rnn_relu_packed(
            input,
            batch_sizes,
            hx,
            flat_weights,
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
        )
