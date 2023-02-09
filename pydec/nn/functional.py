r"""Functional interface"""

import torch
import pydec
from torch import Tensor
import torch.nn.functional as F
from .._composition import Composition
from ..decomposition import (
    get_decomposition_func,
    get_decomposition_name,
)
from ..overrides import _auto_registration, _register_builtin_function
from ..exception_utils import none_decomposition_func_error, arg_value_error

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

from typing import (
    List,
    Tuple,
    Optional,
    Union,
    Any,
    ContextManager,
    Callable,
    overload,
    Iterator,
    NamedTuple,
    Sequence,
    TypeVar,
)

from ..variable_functions import _from_replce
import warnings

T = TypeVar("T")


def _add_docstr(obj: T, doc_obj: str) -> T:
    obj.__doc__ = doc_obj


@_auto_registration
def relu(
    input: Composition, inplace: bool = False, *, ref: Optional[Tensor] = None
) -> Composition:
    r"""relu(input, inplace=False, *, ref=None) -> Composition

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        result = pydec.relu_(input, ref=ref)
    else:
        result = pydec.relu(input, ref=ref)
    return result


relu_ = _add_docstr(
    pydec.relu_,
    r"""
relu_(input, *, ref=None) -> Composition

In-place version of :func:`~relu`.
""",
)


@_auto_registration
def leaky_relu(
    input: Composition, inplace=False, *, ref: Optional[Tensor] = None
) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(
            input=input, func=torch.nn.functional.leaky_relu, ref=ref, inplace=inplace
        )
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def leaky_relu_(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    return leaky_relu(input=input, inplace=True, ref=ref)


@_auto_registration
def gelu(input: Composition, *, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.nn.functional.gelu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def tanh(input, *, ref: Optional[Tensor] = None):
    r"""tanh(input, *, ref=None) -> Composition

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")
    return input.tanh(ref=ref)


def sigmoid(input, *, ref: Optional[Tensor] = None):
    r"""sigmoid(input, *, ref=None) -> Composition

    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    return input.sigmoid(ref=ref)


@_auto_registration
def linear(input: Composition, weight: Tensor, bias: Tensor = None) -> Composition:
    out = input @ weight.t()
    if bias is not None:
        out._residual_tensor = out._residual_tensor + bias
        return out
    else:
        return out


@_register_builtin_function(torch.nn.functional.layer_norm)
def layer_norm_1d(
    input: Composition,
    ref: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Composition:
    r"""Applies Layer Normalization for last dimension."""
    input_mean = input.mean(dim=-1, keepdim=True)
    if ref is None:
        ref = input.c_sum()
    input_std = torch.sqrt(torch.var(ref, dim=-1, unbiased=False, keepdim=True) + eps)
    out = (input - input_mean) * weight / input_std

    if bias is not None:
        out._residual_tensor = out._residual_tensor + bias
        return out
    else:
        return out


@overload
def conv2d(
    input: Composition,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Union[_int, _size] = 0,
    dilation: Union[_int, _size] = 1,
    groups: _int = 1,
) -> Composition:
    ...


@overload
def conv2d(
    input: Composition,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: str = "valid",
    dilation: Union[_int, _size] = 1,
    groups: _int = 1,
) -> Composition:
    ...


@_auto_registration
def conv2d(
    input: Composition,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Any = 0,
    dilation: Union[_int, _size] = 1,
    groups: _int = 1,
):
    """
    Applies a 2D convolution over an input composition.
    """
    if len(input.size()) != 4 or len(input.size()) != 3:
        raise arg_value_error(
            f"Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [{len(input.size())}]"
        )
    if len(input.size()) == 3:
        out_composition_tensor = F.conv2d(
            input._composition_tensor, weight, None, stride, padding, dilation, groups,
        )
        out_residual_tensor = F.conv2d(
            input._residual_tensor, weight, None, stride, padding, dilation, groups,
        )
    else:
        out_composition_tensor = F.conv2d(
            input._composition_tensor.view((-1,) + input.size()[1:]),
            weight,
            None,
            stride,
            padding,
            dilation,
            groups,
        ).view((-1,) + input.size())
        out_residual_tensor = F.conv2d(
            input._residual_tensor, weight, None, stride, padding, dilation, groups,
        )
    out_residual_tensor += bias
    return _from_replce(out_composition_tensor, out_residual_tensor)


@_auto_registration
def max_pool2d(
    input: Composition,
    kernel_size: Union[_int, Tuple[_int, _int]],
    stride: Union[_int, Tuple[_int, _int]] = None,
    padding: Union[_int, Tuple[_int, _int]] = 0,
    dilation: Union[_int, Tuple[_int, _int]] = 1,
    ceil_mode: _bool = False,
    return_indices: _bool = False,
):
    """
    Applies a 2D max pooling over an input composition.
    """
    if len(input.size()) != 4 or len(input.size()) != 3:
        raise arg_value_error(
            f"Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [{len(input.size())}]"
        )
    recovery = input.c_sum()
    _, indices = F.max_pool2d(
        recovery,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    out = max_pool2d_with_indices(input, indices=indices)
    if return_indices:
        return out, indices
    else:
        return out


def max_pool2d_with_indices(
    input: Composition,
    indices: Tensor,
    kernel_size: Union[_int, Tuple[_int, _int]],
    stride: Union[_int, Tuple[_int, _int]] = None,
    padding: Union[_int, Tuple[_int, _int]] = 0,
    dilation: Union[_int, Tuple[_int, _int]] = 1,
    ceil_mode: _bool = False,
):
    """
    Applies a 2D max pooling over an input composition.
    TODO: need to support arguments [kernel_size, stride, padding, dilation, ceil_mode]
    """
    if len(input.size()) != 4 or len(input.size()) != 3:
        raise arg_value_error(
            f"Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [{len(input.size())}]"
        )
    flat_input = input.view(input.size()[:-2] + (-1,))
    H_out, W_out = indices.size()[-2:]

    out = flat_input.gather(dim=-1, index=indices.view(indices.size()[:-2] + (-1,)))
    out = out.view(out.size()[:-1] + (H_out, W_out))

    return out


def legacy_relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    residual_out = torch.nn.functional.relu(input._residual_tensor)
    out = input.masked_fill(zero_mask, 0.0)
    masked_residual_out = out._residual_tensor
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:

        delta_context = _from_replce(input._composition_tensor, residual_out)
        delta_out = decomposition_func(
            input=delta_context, func=lambda x: x, ref=masked_residual_out
        )
        out._composition_tensor += delta_out._composition_tensor
        out._residual_tensor = delta_out._residual_tensor
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


# relu = legacy_relu
