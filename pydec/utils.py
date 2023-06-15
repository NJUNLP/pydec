import torch
from typing import (
    Any,
    Union,
    List,
    Dict,
    Tuple,
    Optional,
    Callable,
    overload,
    Sequence,
    Iterable,
)

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


def _shift_dim(dim: _int) -> _int:
    if dim >= 0:
        dim += 1
    return dim


@overload
def _shift_dims(dims: _size, /) -> torch.Size:
    ...


@overload
def _shift_dims(*dims: _int) -> torch.Size:
    ...


def _shift_dims(*dims: Any) -> torch.Size:
    if len(dims) == 1:
        dims = dims[0]
    out = list(dims)
    for i in range(len(dims)):
        out[i] = _shift_dim(dims[i])
    return torch.Size(out)


def parse_args(args: Iterable[Any], key_list: Iterable[str], kwargs: Dict[str, Any]):
    for key, arg in zip(key_list, args):
        kwargs[key] = arg
