import torch
import pydec
from torch import Tensor
from torch.nn.parameter import Parameter
from torch._C import memory_format

import pydec.core as core
from pydec._composition import Composition, IndexComposition
from .overrides import _auto_registration, _register_builtin_function

from typing import (
    Any,
    Union,
    List,
    Tuple,
    Optional,
    Callable,
    overload,
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

from torch import strided

from pydec.utils import _shift_dim, _shift_dims
from pydec.exception_utils import (
    arg_value_error,
    none_decomposition_func_error,
    unsupported_operand_error,
)
import builtins


def void() -> Composition:
    return Composition()


@overload
def as_composition(components: Tensor, residual: Tensor = None) -> Composition:
    ...


def as_composition(
    components: Tensor, residual: Tensor = None, check: _bool = True
) -> Composition:
    # TODO: implement `check`, check shape, dtype and device
    out = void()
    out._component_tensor = components
    if residual is None:
        residual = torch.zeros(components.size()[1:]).to(components)
    out._residual_tensor = residual
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
        r_tensors,
        dim,
        out=out._residual_tensor if out is not None else None,
    )
    return as_composition(out_component_tensor, out_residual_tensor)


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
        c_tensors,
        0,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = None
    if sum_residual:
        r_tensors = tuple(c._residual_tensor for c in compositions)
        out_residual_tensor = builtins.sum(r_tensors)
    return as_composition(out_component_tensor, out_residual_tensor)


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
        r_tensors,
        dim,
        out=out._residual_tensor if out is not None else None,
    )
    return as_composition(out_component_tensor, out_residual_tensor)


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
        components,
        0,
        out=out._component_tensor if out is not None else None,
    )
    return as_composition(out_component_tensor)


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

    return as_composition(out_component_tensor, out_residual_tensor)


def c_apply(input: Composition, callable: Callable[..., Tensor]) -> Composition:
    # TODO: check shape, dtype and device of result
    out_component_tensor = callable(input._component_tensor)
    out_residual_tensor = callable(input._residual_tensor)
    if not isinstance(out_component_tensor, Tensor) or not isinstance(
        out_residual_tensor, Tensor
    ):
        raise RuntimeError(
            "TypeError: 'callable' must return a tensor, not {}".format(
                type(
                    out_component_tensor
                    if not isinstance(out_component_tensor, Tensor)
                    else out_residual_tensor
                ).__name__
            )
        )
    return as_composition(out_component_tensor, out_residual_tensor)


def c_map(input, other: Composition, callable: Callable[..., Tensor]) -> Composition:
    # TODO: check shape, dtype and device of result
    out_component_tensor = callable(input._component_tensor, other._component_tensor)
    out_residual_tensor = callable(input._residual_tensor, other._residual_tensor)
    if not isinstance(out_component_tensor, Tensor) or not isinstance(
        out_residual_tensor, Tensor
    ):
        raise RuntimeError(
            "TypeError: 'callable' must return a tensor, not {}".format(
                type(
                    out_component_tensor
                    if not isinstance(out_component_tensor, Tensor)
                    else out_residual_tensor
                ).__name__
            )
        )
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def numel(input: Composition) -> _int:
    return torch.numel(input._residual_tensor)


def c_numel(input: Composition, count_residual=False) -> _int:
    if count_residual:
        return input._component_tensor.numel() + input._residual_tensor.numel()
    else:
        return input._component_tensor.numel()


def numc(input: Composition) -> _int:
    return len(input.components)


@_auto_registration
def clone(
    input: Composition,
    *,
    memory_format: Optional[memory_format] = None,
) -> Composition:
    out_component_tensor = torch.clone(
        input._component_tensor, memory_format=memory_format
    )
    out_residual_tensor = torch.clone(
        input._residual_tensor, memory_format=memory_format
    )
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def detach(input: Composition) -> Composition:
    out_component_tensor = torch.detach(input._component_tensor)
    out_residual_tensor = torch.detach(input._residual_tensor)
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def detach_(input: Composition) -> Composition:
    torch.detach_(input._component_tensor)
    torch.detach_(input._residual_tensor)
    return input


@overload
def add(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def add(
    input: Union[Composition, Tensor, Number],
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def add(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    # TODO: fix bug: broadcasting errors, trigger when composition + composition
    # TODO: fix bug: shape errors, trigger when composition(residual broadcasting) + tensor
    # TODO: fix bug: also fix `sub`
    # TODO: add broadcasting examples in docs after bug fixing
    if not isinstance(input, Composition) and not isinstance(other, Composition):
        raise unsupported_operand_error("add", type(input), type(other))
    if isinstance(input, Composition):
        if isinstance(other, Composition):
            return core.decBLAS.cc_add(input, other, alpha=alpha, out=out)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_add(input, other, alpha=alpha, out=out)
        else:
            raise unsupported_operand_error("add", type(input), type(other))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.tc_add(input, other, alpha=alpha, out=out)
    else:
        raise unsupported_operand_error("add", type(input), type(other))


@overload
def sub(
    input: Composition,
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def sub(
    input: Union[Composition, Tensor, Number],
    other: Composition,
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def sub(
    input: Union[Composition, Tensor, Number],
    other: Union[Composition, Tensor, Number],
    *,
    alpha: Optional[Number] = 1,
    out: Optional[Composition] = None,
) -> Composition:
    # TODO: fix bug: see TODOs in `add`
    if not isinstance(input, Composition) and not isinstance(other, Composition):
        raise unsupported_operand_error("sub", type(input), type(other))
    if isinstance(input, Composition):
        if isinstance(other, Composition):
            return core.decBLAS.cc_sub(input, other, alpha=alpha, out=out)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_sub(input, other, alpha=alpha, out=out)
        else:
            raise unsupported_operand_error("sub", type(input), type(other))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.tc_sub(input, other, alpha=alpha, out=out)
    else:
        raise unsupported_operand_error("sub", type(input), type(other))


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
def subtract(*args, **kwargs) -> Composition:
    return sub(*args, **kwargs)


@overload
def mul(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def mul(
    input: Union[Tensor, Number],
    other: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def mul(
    input: Composition,
    other: Composition,
    *,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    ...


@_auto_registration
def mul(
    input: Union[Composition, Tensor, Number],
    other: Union[Composition, Tensor, Number],
    *,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(other, Composition):
        raise unsupported_operand_error("mul", type(input), type(other))
    if isinstance(input, Composition):
        if isinstance(other, Composition):
            return core.decMVF.cc_mul(
                input,
                other,
                out=out,
                ref_input=ref_input,
                ref_other=ref_other,
            )
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_mul(input, other, out=out)
        else:
            raise unsupported_operand_error("mul", type(input), type(other))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.tc_mul(input, other, out=out)
    else:
        raise unsupported_operand_error("mul", type(input), type(other))


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


@overload
def div(
    input: Composition,
    other: Union[Tensor, Number],
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def div(
    input: Union[Tensor, Number],
    other: Composition,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def div(
    input: Composition,
    other: Composition,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    ...


@_auto_registration
def div(
    input,
    other,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[Composition] = None,
    ref_input: Optional[Tensor] = None,
    ref_other: Optional[Tensor] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(other, Composition):
        raise unsupported_operand_error("div", type(input), type(other))
    if isinstance(input, Composition):
        if isinstance(other, Composition):
            return core.decMVF.cc_div(
                input,
                other,
                rounding_mode=rounding_mode,
                ref_input=ref_input,
                ref_other=ref_other,
            )
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_div(
                input, other, rounding_mode=rounding_mode, out=out
            )
        else:
            raise unsupported_operand_error("div", type(input), type(other))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        # TODO: not implement yet
        raise unsupported_operand_error("div", type(input), type(other))
    else:
        raise unsupported_operand_error("div", type(input), type(other))


@overload
def divide(
    input: Composition,
    other: Tensor,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def divide(
    input: Composition,
    other: Tensor,
    *,
    rounding_mode: Optional[str],
    out: Optional[Composition] = None,
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


@_auto_registration
def divide(*args, **kwargs):
    return div(*args, **kwargs)


@overload
def mv(
    input: Composition,
    vec: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def mv(
    input: Union[Composition, Tensor],
    vec: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def mv(
    input,
    vec,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(vec, Composition):
        raise unsupported_operand_error("mv", type(input), type(vec))
    if isinstance(input, Composition):
        if isinstance(vec, Composition):
            # TODO: not implement yet
            raise unsupported_operand_error("mv", type(input), type(vec))
        elif isinstance(vec, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_mv(input, vec, out=out)
        else:
            raise unsupported_operand_error("mv", type(input), type(vec))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.ct_mv(input, vec, out=out)
    else:
        raise unsupported_operand_error("mv", type(input), type(vec))


@overload
def mm(
    input: Composition,
    mat2: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def mm(
    input: Union[Composition, Tensor],
    mat2: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def mm(
    input,
    mat2,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(mat2, Composition):
        raise unsupported_operand_error("mm", type(input), type(mat2))
    if isinstance(input, Composition):
        if isinstance(mat2, Composition):
            # TODO: not implement yet
            raise unsupported_operand_error("mm", type(input), type(mat2))
        elif isinstance(mat2, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_mm(input, mat2, out=out)
        else:
            raise unsupported_operand_error("mm", type(input), type(mat2))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.tc_mm(input, mat2, out=out)
    else:
        raise unsupported_operand_error("mm", type(input), type(mat2))


@overload
def bmm(
    input: Composition,
    mat2: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def bmm(
    input: Union[Composition, Tensor],
    mat2: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def bmm(
    input: Union[Composition, Tensor],
    mat2: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(mat2, Composition):
        raise unsupported_operand_error("bmm", type(input), type(mat2))
    if isinstance(input, Composition):
        if isinstance(mat2, Composition):
            return core.decMVF.cc_bmm(input, mat2, out=out)
        elif isinstance(mat2, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_bmm(input, mat2, out=out)
        else:
            raise unsupported_operand_error("bmm", type(input), type(mat2))
    elif isinstance(input, Tensor):
        assert isinstance(mat2, Composition)  # narrow type
        return core.decBLAS.tc_bmm(input, mat2, out=out)
    else:
        raise unsupported_operand_error("bmm", type(input), type(mat2))


@overload
def matmul(
    input: Composition,
    other: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@overload
def matmul(
    input: Union[Composition, Tensor],
    other: Composition,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def matmul(
    input: Union[Composition, Tensor],
    other: Union[Composition, Tensor],
    *,
    out: Optional[Composition] = None,
) -> Composition:
    if not isinstance(input, Composition) and not isinstance(other, Composition):
        raise unsupported_operand_error("matmul", type(input), type(other))
    if isinstance(input, Composition):
        if isinstance(other, Composition):
            return core.decMVF.cc_matmul(input, other)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_matmul(input, other, out=out)
        else:
            raise unsupported_operand_error("matmul", type(input), type(other))
    elif isinstance(input, (_int, _float, _bool, Tensor)):
        return core.decBLAS.tc_matmul(input, other, out=out)
    else:
        raise unsupported_operand_error("matmul", type(input), type(other))


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
def isinf(input: Composition) -> Composition:
    out_residual = torch.isinf(input.residual)
    out_components = torch.isinf(input.components)
    return as_composition(out_components, out_residual)


@_auto_registration
def isnan(input: Composition) -> Composition:
    out_residual = torch.isnan(input.residual)
    out_components = torch.isnan(input.components)
    return as_composition(out_components, out_residual)


@_auto_registration
def unsqueeze(input: Composition, dim: _int) -> Composition:
    out_residual_tensor = input._residual_tensor.unsqueeze(dim)
    out_component_tensor = input._component_tensor.unsqueeze(_shift_dim(dim))
    return as_composition(out_component_tensor, out_residual_tensor)


@overload
def squeeze(input: Composition) -> Composition:
    ...


@overload
def squeeze(input: Composition, dim: _int) -> Composition:
    ...


@_auto_registration
def squeeze(input: Composition, dim=None) -> Composition:
    if dim is None:
        out_residual_tensor = input._residual_tensor.squeeze()
        out_component_tensor = input._component_tensor.squeeze()
        if input.numc() == 1:
            out_component_tensor = out_component_tensor.unsqueeze(0)
    else:
        out_residual_tensor = input._residual_tensor.squeeze(dim)
        out_component_tensor = input._component_tensor.squeeze(_shift_dim(dim))
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def transpose(input: Composition, dim0: _int, dim1: _int) -> Composition:
    out_residual_tensor = input._residual_tensor.transpose(dim0, dim1)
    out_component_tensor = input._component_tensor.transpose(
        _shift_dim(dim0), _shift_dim(dim1)
    )
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def permute(input: Composition, dims: _size) -> Composition:
    out_residual_tensor = input._residual_tensor.permute(dims)
    out_component_tensor = input._component_tensor.permute((0,) + _shift_dims(dims))
    return as_composition(out_component_tensor, out_residual_tensor)


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
    out_components = {}
    out_residual = {}
    if out is not None:
        out_components["out"] = out.components
        out_residual["out"] = out.residual
    if dim is None:
        if input.ndim == 0:
            return input
        dim = tuple(range(1, input._component_tensor.dim()))
        out_component_tensor = torch.sum(
            input._component_tensor,
            dim=dim,
            dtype=dtype,
            **out_components,
        )
        out_residual_tensor = torch.sum(
            input._residual_tensor, dtype=dtype, **out_residual
        )
    else:
        out_residual_tensor = torch.sum(
            input._residual_tensor,
            dim=dim,
            keepdim=keepdim,
            dtype=dtype,
            **out_residual,
        )
        if isinstance(dim, _int):
            dim = (dim,)
        out_component_tensor = torch.sum(
            input._component_tensor,
            dim=_shift_dims(dim),
            keepdim=keepdim,
            dtype=dtype,
            **out_components,
        )
    return as_composition(out_component_tensor, out_residual_tensor)


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
    dim: Union[None, _int, _size] = None,
    keepdim: _bool = False,
    *,
    dtype: Optional[_dtype] = None,
    out: Optional[Composition] = None,
) -> Composition:
    # TODO: move to BLAS
    if out is not None:
        raise arg_value_error(
            f"{mean.__name__}() dees not support keyword 'out' currently"
        )
    if dim is None:
        dim = tuple(range(1, input._component_tensor.dim()))
        if out is not None:
            out_component_tensor = torch.mean(
                input._component_tensor, dim=dim, dtype=dtype, out=out._component_tensor
            )
            out_residual_tensor = torch.mean(
                input._residual_tensor, dtype=dtype, out=out._residual_tensor
            )
        else:
            out_component_tensor = torch.mean(
                input._component_tensor, dim=dim, dtype=dtype
            )
            out_residual_tensor = torch.mean(input._residual_tensor, dtype=dtype)
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
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def reshape(input: Composition, shape: _size) -> Composition:
    out_component_tensor = input._component_tensor.reshape((input.numc(),) + shape)
    out_residual_tensor = input._residual_tensor.reshape(shape)
    return as_composition(out_component_tensor, out_residual_tensor)


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
    return as_composition(out_component_tensor, out_residual_tensor)


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
    return as_composition(out_component_tensor, out_residual_tensor)


# TODO: to support 'out: Optional[Composition] = None'
@_auto_registration
def masked_select(
    input: Composition,
    mask: Tensor,
    *,
    out: Optional[Composition] = None,
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
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def masked_scatter(input: Composition, mask: Tensor, source: Tensor) -> Composition:
    out_component_tensor = input._component_tensor.masked_scatter(mask[None], source)
    out_residual_tensor = input._residual_tensor.masked_scatter(mask, source)
    return as_composition(out_component_tensor, out_residual_tensor)


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
    return as_composition(out_component_tensor, out_residual_tensor)


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
    dim: _int,
    index: Tensor,
    src_value: Optional[Union[Tensor, Number]] = None,
    *,
    reduce: Optional[str] = None,
    out: Optional[Composition] = None,
    src: Optional[Tensor] = None,
    value: Optional[Number] = None,
) -> Composition:
    r"""
    Unsafe.
    Safe when reduce is not None.
    """
    if src_value is None:
        src_value = src if src is not None else value
    assert src_value is not None
    if reduce == "add":
        holder = torch.zeros_like(input._residual_tensor).to(input._residual_tensor)
        holder = holder.scatter(dim, index, src_value, reduce=reduce)
        c_out = input + holder
        if out is not None:
            # TODO: use the out argument of `torch.add` raises an error
            out._component_tensor[:] = c_out._component_tensor
            out._residual_tensor[:] = c_out._residual_tensor
        return c_out
    else:
        c_index = index[None].expand((input.numc(),) + (-1,) * index.dim())
        c_src: Union[Tensor, Number]
        if isinstance(src_value, Tensor):
            c_src = src_value[None].expand((input.numc(),) + (-1,) * src_value.dim())
        else:
            c_src = src_value
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
                src_value,
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
                src_value,
                reduce=reduce,
                out=out._residual_tensor if out is not None else None,
            )
        return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def diagonal_scatter(
    input: Composition,
    src: Tensor,
    offset: _int = 0,
    dim1: _int = 0,
    dim2: _int = 1,
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
    return as_composition(out_component_tensor, out_residual_tensor)


@overload
def index_select(
    input: Composition,
    dim: _int,
    index: Tensor,
    *,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def index_select(
    input: Composition,
    dim: _int,
    index: Tensor,
    *,
    out: Optional[Composition] = None,
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
    return as_composition(out_component_tensor, out_residual_tensor)


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
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def masked_select(
    input: Composition,
    mask: Tensor,
    *,
    out: Optional[Composition] = None,
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
    return as_composition(out_component_tensor, out_residual_tensor)


@overload
def select(input: Composition, dim: _int, index: _int) -> Composition:
    ...


def select(input: Composition, dim: Any, index: _int) -> Composition:
    out_component_tensor = torch.select(
        input._component_tensor, dim=_shift_dim(dim), index=index
    )
    out_residual_tensor = torch.select(input._residual_tensor, dim=dim, index=index)
    return as_composition(out_component_tensor, out_residual_tensor)


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
    return as_composition(out_component_tensor, out_residual_tensor)


@overload
def round(input: Composition, *, out: Optional[Composition] = None) -> Composition:
    ...


@overload
def round(
    input: Composition,
    *,
    decimals: _int,
    out: Optional[Composition] = None,
) -> Composition:
    ...


@_auto_registration
def round(
    input: Composition,
    *,
    decimals: _int = None,
    out: Optional[Composition] = None,
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
        out_component_tensor = torch.round(input._component_tensor)
        out_residual_tensor = torch.round(input._residual_tensor)
    return as_composition(out_component_tensor, out_residual_tensor)


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
        return as_composition(out_component_tensor, out_residual_tensor)

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
    return as_composition(out_component_tensor, out_residual_tensor)


def empty_indices(
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
    # TODO: bug: c.abs() not eqal to c.c_sum().abs(); This API has not been documented.
    out_component_tensor = torch.abs(
        input._component_tensor,
        out=out._component_tensor if out is not None else None,
    )
    out_residual_tensor = torch.abs(
        input._residual_tensor,
        out=out._residual_tensor if out is not None else None,
    )
    return as_composition(out_component_tensor, out_residual_tensor)


@_auto_registration
def abs_(input: Composition) -> Composition:
    # TODO: bug: c.abs() not eqal to c.c_sum().abs()
    torch.abs_(input._component_tensor)
    torch.abs_(input._residual_tensor)
    return input


@_auto_registration
def relu(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    return core.decOVF.relu(input, ref=ref)


@_auto_registration
def relu_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    return core.decOVF.relu_(input, ref=ref)


@_auto_registration
def tanh(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.tanh(input, out=out, ref=ref)


@_auto_registration
def tanh_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    return core.decOVF.tanh_(input, ref=ref)


@_auto_registration
def sigmoid(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.sigmoid(input, out=out, ref=ref)


@_auto_registration
def sigmoid_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    return core.decOVF.sigmoid_(input, ref=ref)


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
        (max_batch_size,),
        dtype=torch.long,
        device=packed_input.device,
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


def _rnn_cell(
    input: Composition,
    hx: Union[Tensor, Composition],
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter = None,
    bias_hh: Parameter = None,
    nonlinearity: str = "tanh",
    *,
    ref: Optional[Tensor] = None,
):
    if nonlinearity == "tanh":
        activation = tanh
    elif nonlinearity == "relu":
        activation = relu
    else:
        raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
    return activation(
        pydec.nn.functional.linear(input, weight_ih, bias_ih)
        + pydec.nn.functional.linear(hx, weight_hh, bias_hh),
        ref=ref,
    )


def _rnn_layer(
    input: Composition,
    batch_sizes: Optional[Tensor],
    hx: Tensor,
    weight_ih: Parameter,
    weight_hh: Parameter,
    bias_ih: Parameter,
    bias_hh: Parameter,
    reverse: bool = False,
    nonlinearity: str = "tanh",
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
                    hx = pydec.cat(
                        [hx, append_hx],
                        dim=0,
                    )
        if not reverse:
            batch = input[batch_start : batch_start + batch_size]
            batch_start += batch_size
        else:
            batch = input[batch_start - batch_size : batch_start]
            batch_start -= batch_size
        hx = _rnn_cell(
            batch,
            hx,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            nonlinearity,
        )
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


def _rnn_packed(
    input: Composition,
    batch_sizes: Tensor,
    hx: Tensor,
    flat_weights: List[Parameter],
    bias: bool,
    num_layers: int,
    dropout: float,
    training: bool,
    bidirectional: bool,
    nonlinearity: str = "tanh",
):
    x = input
    weight_group_len = 2 * (2 if bias else 1) * (2 if bidirectional else 1)
    out_hx_list = []

    for i in range(num_layers):
        hx_layer, hx_layer_r = None, None
        weight_ih, weight_hh, bias_ih, bias_hh = (
            None,
            None,
            None,
            None,
        )
        weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r = (
            None,
            None,
            None,
            None,
        )
        weight_group = flat_weights[weight_group_len * i : weight_group_len * (i + 1)]

        weight_ih = weight_group[0]
        weight_hh = weight_group[1]
        if bias:
            bias_ih = weight_group[2]
            bias_hh = weight_group[3]
        if not bidirectional:
            hx_layer = hx[i]
            x, out_hx = _rnn_layer(
                x,
                batch_sizes,
                hx_layer,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                nonlinearity=nonlinearity,
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

            x_, out_hx = _rnn_layer(
                x,
                batch_sizes,
                hx_layer,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                nonlinearity=nonlinearity,
            )
            x_r, out_hx_r = _rnn_layer(
                x,
                batch_sizes,
                hx_layer_r,
                weight_ih_r,
                weight_hh_r,
                bias_ih_r,
                bias_hh_r,
                True,
                nonlinearity=nonlinearity,
            )
            x = pydec.cat([x_, x_r], dim=-1)
            out_hx_list.append(out_hx)
            out_hx_list.append(out_hx_r)
        if training and dropout > 0:
            # TODO: should also apply dropout to out_hx?
            x = pydec.nn.functional.dropout(
                x, p=dropout, training=training, inplace=True
            )

    out = x
    out_hx = pydec.stack(out_hx_list, dim=0)
    return out, out_hx


@overload
def rnn_relu(
    data: Composition,
    batch_sizes: Tensor,
    hx: Tensor,
    params: Union[Tuple[Tensor, ...], List[Tensor]],
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    /,
) -> Tuple[Composition, Composition]:
    ...


@overload
def rnn_relu(
    input: Composition,
    hx: Tensor,
    params: Union[Tuple[Tensor, ...], List[Tensor]],
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
    /,
) -> Tuple[Composition, Composition]:
    ...


@_auto_registration
def rnn_relu(
    *args,
):
    """
    Only supports the positional arguments in current version
    """
    if isinstance(args[3], bool):
        (
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        ) = args
        batch_sizes = None
        if batch_first:
            input: Composition = input.transpose(0, 1)
        seq_len = input.size(0)
        batch_size = input.size(1)
        batch_sizes = torch.tensor(
            [batch_size] * input.size(0),
            dtype=torch.long,
            device="cpu",
        )
        input = input.view(seq_len * batch_size, -1)
        out, out_hx = _rnn_packed(
            input,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            nonlinearity="relu",
        )
        out = out.view(seq_len, batch_size, -1)
        if batch_first:
            out.transpose_(0, 1)
        return out, out_hx
    else:
        (
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
        ) = args
        return _rnn_packed(
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            nonlinearity="relu",
        )


@overload
def rnn_tanh(
    data: Composition,
    batch_sizes: Tensor,
    hx: Tensor,
    params: Union[Tuple[Tensor, ...], List[Tensor]],
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    /,
) -> Tuple[Composition, Composition]:
    ...


@overload
def rnn_tanh(
    input: Composition,
    hx: Tensor,
    params: Union[Tuple[Tensor, ...], List[Tensor]],
    has_biases: _bool,
    num_layers: _int,
    dropout: _float,
    train: _bool,
    bidirectional: _bool,
    batch_first: _bool,
    /,
) -> Tuple[Composition, Composition]:
    ...


@_auto_registration
def rnn_tanh(
    *args,
):
    """
    Only supports the positional arguments in current version
    """
    if isinstance(args[3], bool):
        (
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        ) = args
        batch_sizes = None
        if batch_first:
            input: Composition = input.transpose(0, 1)
        seq_len = input.size(0)
        batch_size = input.size(1)
        batch_sizes = torch.tensor(
            [batch_size] * input.size(0),
            dtype=torch.long,
            device="cpu",
        )
        input = input.view(seq_len * batch_size, -1)
        out, out_hx = _rnn_packed(
            input,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            nonlinearity="tanh",
        )
        out = out.view(seq_len, batch_size, -1)
        if batch_first:
            out.transpose_(0, 1)
        return out, out_hx
    else:
        (
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
        ) = args
        return _rnn_packed(
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            nonlinearity="tanh",
        )


@_auto_registration
def reciprocal(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.reciprocal(input, out=out, ref=ref)


@_auto_registration
def reciprocal_(
    input: Composition,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.reciprocal_(input, ref=ref)


@_auto_registration
def exp(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.exp(input, out=out, ref=ref)


@_auto_registration
def exp_(
    input: Composition,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.exp_(input, ref=ref)


@overload
def softmax(
    input: Composition,
    dim: _int,
    dtype: Optional[_dtype] = None,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    ...


@_auto_registration
def softmax(
    input: Composition,
    dim: _int,
    dtype: Optional[_dtype] = None,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    # TODO: should we disable grad here?
    # TODO: This API has not been documented.
    if dtype is not None:
        input = input.to(dtype)
        if ref is not None:
            ref = ref.to(dtype)
    input = _softmax_rescale(input)
    components_bias = torch.max(input.components, dim=0, keepdim=False)[0]
    components_bias = -torch.max(components_bias, dim=dim, keepdim=True)[0]
    residual_bias = -torch.max(input.residual, dim=dim, keepdim=True)[0]
    bias = torch.min(components_bias, residual_bias)
    input = core.decOVF.biased_exp(input, bias=bias, ref=ref)
    if ref is None:
        exp_sum = input.c_sum().sum(dim=dim, keepdim=True)
    else:
        exp_sum = torch.exp(ref + bias)
    return pydec.div(input, exp_sum, out=out)


@_auto_registration
def sqrt(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.sqrt(input, out=out, ref=ref)


@_auto_registration
def sqrt_(
    input: Composition,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.sqrt_(input, ref=ref)


@overload
def var(
    input: Composition,
    dim: Union[_int, _size],
    unbiased: _bool = True,
    keepdim: _bool = False,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    ...


@overload
def var(
    input: Composition,
    unbiased: _bool = True,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
    ...


@_auto_registration
def var(
    input: Composition,
    dim: Union[_int, _size] = None,
    unbiased: _bool = True,
    keepdim: _bool = False,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    if isinstance(dim, _bool):
        unbiased = dim
        dim = None
    if dim is None:
        dim = tuple(range(0, input.dim()))
    return core.decMVF.var(input, dim, unbiased, keepdim, out=out, ref=ref)


@_auto_registration
def square(
    input: Composition,
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.square(input, out=out, ref=ref)


@_auto_registration
def square_(
    input: Composition,
    ref: Optional[Tensor] = None,
) -> Composition:
    return core.decOVF.square_(input, ref=ref)


def _softmax_rescale(input: Composition) -> Composition:
    """
    This is not a standard api of PyTorch.
    Use to deal with numerical explosion problem in decomposition.
    """
    combine_components = torch.cat(
        (
            input.components,
            input.residual[None],
        )
    )
    softmax_components = torch.softmax(combine_components, dim=0)
    rescaled_components = softmax_components * input.c_sum()
    out_components = rescaled_components[:-1]
    out_residual = rescaled_components[-1]
    return as_composition(out_components, out_residual)


@overload
def eq(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def eq(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def eq(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def eq(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().eq(other.c_sum())
    return input.c_sum().eq(other)


@overload
def ne(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def ne(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def ne(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def ne(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().ne(other.c_sum())
    return input.c_sum().ne(other)


@overload
def gt(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def gt(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def gt(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def gt(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().gt(other.c_sum())
    return input.c_sum().gt(other)


@overload
def lt(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def lt(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def lt(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def lt(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().lt(other.c_sum())
    return input.c_sum().lt(other)


@overload
def ge(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def ge(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def ge(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def ge(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().ge(other.c_sum())
    return input.c_sum().ge(other)


@overload
def le(input: Composition, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    ...


@overload
def le(
    input: Composition, other: Composition, *, out: Optional[Tensor] = None
) -> Tensor:
    ...


@overload
def le(input: Composition, other: Number, *, out: Optional[Tensor] = None) -> Tensor:
    ...


def le(input: Composition, other: Any, *, out: Optional[Tensor] = None) -> Tensor:
    if isinstance(other, Composition):
        return input.c_sum().le(other.c_sum())
    return input.c_sum().le(other)
