import torch
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


def _shift_dim(dim: _int) -> _int:
    if dim >= 0:
        dim += 1
    return dim


@overload
def _shift_dims(dims: _size) -> torch.Size:
    ...


@overload
def _shift_dims(*dims: _int) -> torch.Size:
    ...


def _shift_dims(*dims) -> torch.Size:
    if len(dims) == 1:
        dims = dims[0]
    out = list(dims)
    for i in range(len(dims)):
        out[i] = _shift_dim(dims[i])
    return torch.Size(out)
