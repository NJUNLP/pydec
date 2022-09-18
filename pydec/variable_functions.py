import torch
from torch import Tensor
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

from pydec.utils import _shift_dim, _shift_dims
from pydec.exception_utils import arg_value_error


def void() -> Composition:
    return Composition(tuple(), 0)


def _from_replce(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    out = void()
    out._composition_tensor = composition_tensor
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


def c_cat(
    compositions: Union[Tuple[Composition, ...], List[Composition]],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    for i in range(1, len(compositions)):
        if compositions[i].size() != compositions[0].size():
            raise arg_value_error(
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for composition number {i} in the list."
            )

    c_tensors = tuple(c._composition_tensor for c in compositions)
    out_composition_tensor = torch.cat(
        c_tensors,
        0,
        out=out._composition_tensor if out is not None else None,
    )
    r_tensors = tuple(c._residual_tensor for c in compositions)
    out_residual_tensor = sum(r_tensors)
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
                f"Sizes of compositions must match except the number of composition. Expected size [{compositions[0].size()}] but got size [{compositions[i].size()}] for composition number {i} in the list."
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


def diagonal_init(
    input: Composition, src: Tensor, dim: _int, offset: _int = 0
) -> Composition:
    out_composition_tensor = input._composition_tensor.diagonal_scatter(
        src, offset=offset, dim1=0, dim2=dim
    )
    out_residual_tensor = input._residual_tensor.clone()
    return _from_replce(out_composition_tensor, out_residual_tensor)


def call_torch_function(c: Composition, func_name: str, **kwargs) -> Composition:
    out_composition_tensor = getattr(torch, func_name)(c._composition_tensor, **kwargs)
    out_residual_tensor = getattr(torch, func_name)(c._residual_tensor, **kwargs)
    return _from_replce(out_composition_tensor, out_residual_tensor)
