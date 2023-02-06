import torch
from torch import Tensor
from torch._C import memory_format

from pydec.composition import Composition

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
from pydec.exception_utils import arg_value_error
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
        r_tensors,
        dim,
        out=out._residual_tensor if out is not None else None,
    )
    return _from_replce(out_composition_tensor, out_residual_tensor)


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
        c_tensors,
        0,
        out=out._composition_tensor if out is not None else None,
    )
    out_residual_tensor = None
    if sum_residual:
        r_tensors = tuple(c._residual_tensor for c in compositions)
        out_residual_tensor = builtins.sum(r_tensors)
    return _from_replce(out_composition_tensor, out_residual_tensor)


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
        r_tensors,
        dim,
        out=out._residual_tensor if out is not None else None,
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
        components,
        0,
        out=out._composition_tensor if out is not None else None,
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


def numel(input: Composition) -> _int:
    return input.numel()


def c_numel(input: Composition, count_residual=False) -> _int:
    return input.c_numel(count_residual=count_residual)


def numc(input: Composition) -> _int:
    return input.numc()


def clone(
    input: Composition, *, memory_format: Optional[memory_format] = None
) -> Composition:
    return input.clone()


def detach(input: Composition) -> Composition:
    return input.detach()


def detach_(input: Composition) -> Composition:
    return input.detach_()


def add(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
    **kwargs,
) -> Composition:
    return input.add(other, alpha=alpha, out=out, **kwargs)


def sub(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    return input.sub(other, alpha=alpha, out=out)


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


def subtract(
    input: Composition,
    other: Any,
    *,
    alpha: Number = 1,
    out: Optional[Tensor] = None,
) -> Composition:
    return sub(input, other=other, alpha=alpha, out=out)


def mul(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    return input.mul(other, out=out)


@overload
def multiply(
    input: Composition, other: Tensor, *, out: Optional[Tensor] = None
) -> Composition:
    ...


@overload
def multiply(input: Composition, other: Number) -> Composition:
    ...


def multiply(
    input: Composition, other: Any, *, out: Optional[Tensor] = None
) -> Composition:
    return mul(input, other=other, out=out)


def div(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    rounding_mode: Optional[str] = None,
) -> Tensor:
    return input.div(other, rounding_mode=rounding_mode)


@overload
def divide(
    input: Composition,
    other: Tensor,
) -> Composition:
    ...


@overload
def divide(
    input: Composition,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
) -> Composition:
    ...


@overload
def divide(
    input: Composition, other: Number, *, rounding_mode: Optional[str]
) -> Composition:
    ...


@overload
def divide(
    input: Composition,
    other: Number,
) -> Composition:
    ...


def divide(input: Composition, other: Any, *, rounding_mode: Optional[str]):
    return div(input, other=other, rounding_mode=rounding_mode)


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
        out_residual_tensor = torch.mv(
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


def any(input: Composition, *args: Any, **kwargs: Any) -> Tensor:
    torch.any(input.c_sum(), *args, **kwargs)


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


def all(input: Composition, *args: Any, **kwargs: Any) -> Tensor:
    torch.all(input.c_sum(), *args, **kwargs)


def unsqueeze(input: Composition, dim: _int) -> Composition:
    return input.unsqueeze(dim)


@overload
def squeeze(input: Composition) -> Composition:
    ...


@overload
def squeeze(input: Composition, dim: _int) -> Composition:
    ...


def squeeze(input: Composition, dim: _int = None) -> Composition:
    return input.squeeze(dim)


def transpose(input: Composition, dim0: _int, dim1: _int) -> Composition:
    return input.transpose(dim0, dim1)


def permute(input: Composition, dims: _size) -> Composition:
    return input.permute(dims=dims)


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
def sum(
    input: Composition,
    dim=None,
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        raise arg_value_error(
            f"{sum.__name__}() dees not support keyword 'out' currently."
        )
    return input.sum(dim=dim, keepdim=keepdim, dtype=dtype)


def c_sum(input: Composition, *, dtype: Optional[_dtype] = None) -> Tensor:
    return input.c_sum(dtype=dtype)


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
    return input.mean(dim, keepdim, dtype=dtype)


def reshape(input: Composition, shape: _size) -> Composition:
    return input.reshape(shape=shape)


@overload
def masked_fill(input: Composition, mask: Tensor, value: Tensor) -> Composition:
    ...


@overload
def masked_fill(input: Composition, mask: Tensor, value: Number) -> Composition:
    ...


def masked_fill(input: Composition, mask: Tensor, value: Any) -> Composition:
    return input.masked_fill(mask, value)


@overload
def c_masked_fill(input: Composition, mask: Tensor, value: Tensor) -> Composition:
    ...


@overload
def c_masked_fill(input: Composition, mask: Tensor, value: Number) -> Composition:
    ...


def c_masked_fill(input: Composition, mask: Tensor, value: Any) -> Composition:
    return input.c_masked_fill(mask, value)


# TODO: to support 'out: Optional[Composition] = None'
def masked_select(
    input: Composition, mask: Tensor, *, out: Optional[Tensor] = None
) -> Composition:
    if out is not None:
        raise arg_value_error(
            f"{masked_select.__name__}() dees not support keyword 'out' currently."
        )
    return input.masked_select(mask)


def masked_scatter(input: Composition, mask: Tensor, source: Tensor) -> Composition:
    return input.masked_scatter(mask, source)


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
def gather(
    input: Composition,
    dim: Any,
    index: Tensor,
    *,
    sparse_grad: _bool = False,
    out: Optional[Composition] = None,
) -> Composition:
    if out is not None:
        raise arg_value_error(
            f"{gather.__name__}() dees not support keyword 'out' currently."
        )
    return input.gather(dim, index, sparse_grad=sparse_grad)


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


# TODO: to support 'out: Optional[Composition] = None'
def scatter(
    input: Composition,
    dim: Any,
    index: Tensor,
    src: Any = None,
    value: Any = None,
    *,
    reduce: str = None,
    out=None,
) -> Composition:
    if out is not None:
        raise arg_value_error(
            f"{scatter.__name__}() dees not support keyword 'out' currently."
        )
    return input.scatter(dim, index, src, value, reduce=reduce)


def diagonal_scatter(
    input: Composition, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1
) -> Composition:
    return input.diagonal_scatter(src, offset, dim1, dim2)


@overload
def index_select(
    input: Composition, dim: _int, index: Tensor, *, out: Optional[Composition] = None
) -> Composition:
    ...


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


def index_fill(input: Composition, dim: _int, index: Tensor, value: Any) -> Composition:
    return input.index_fill(dim=dim, index=index, value=value)


@overload
def round(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    ...


@overload
def round(
    input: Composition, *, decimals: _int, out: Optional[Composition] = None
) -> Composition:
    ...


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
            input._composition_tensor,
            decimals=decimals,
        )
        out_residual_tensor = torch.round(
            input._residual_tensor,
            decimals=decimals,
        )
    return _from_replce(out_composition_tensor, out_residual_tensor)


@overload
def round_(input: Composition) -> Composition:
    ...


@overload
def round_(input: Composition, *, decimals: _int) -> Composition:
    ...


def round_(input: Composition, *, decimals: _int = None) -> Composition:
    return input.round_(decimals=decimals)


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


def abs(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    out_composition_tensor = torch.abs(
        input._composition_tensor, out=out._composition_tensor
    )
    out_residual_tensor = torch.abs(input._residual_tensor, out=out._residual_tensor)
    return _from_replce(out_composition_tensor, out_residual_tensor)


def abs_(input: Composition) -> Composition:
    torch.abs_(input._composition_tensor)
    torch.abs_(input._residual_tensor)
    return input
