r"""Functional interface"""

import torch
from torch import Tensor
import torch.nn.functional as F
from ..composition import Composition
from ..decomposition import (
    get_decomposition_func,
    get_decomposition_name,
)
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


def relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.nn.functional.relu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def gelu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        out = decomposition_func(input=input, func=torch.nn.functional.gelu, ref=ref)
        assert isinstance(out, Composition)
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


def linear(input: Composition, weight: Tensor, bias: Tensor = None) -> Composition:
    out = input @ weight.t()
    if bias is not None:
        out._residual_tensor = out._residual_tensor + bias
        return out
    else:
        return out


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
    def hybrid_decomposition(
        bias, context: Composition, *, threshold=0.15, eps=1e-6,
    ) -> Composition:
        def ratio_map(ratio: Tensor):
            zero_map = ratio < threshold
            ratio[zero_map] = 0
            ratio[~zero_map] = 1

        compositions = context._composition_tensor
        sum_compositions = compositions.sum(dim=0, keepdim=True)
        abs_compositions = compositions.abs()
        abs_sum_compositions = abs_compositions.sum(dim=0, keepdim=True)
        ratio = sum_compositions.abs() / abs_sum_compositions

        sum_compositions[sum_compositions == 0] = eps
        abs_sum_compositions[abs_sum_compositions == 0] = eps

        ratio_map(ratio)

        weights = ratio * compositions / sum_compositions
        abs_weights = (1 - ratio) * abs_compositions / abs_sum_compositions

        bias_composition_tensor = weights * bias + abs_weights * bias
        from ..variable_functions import _from_replce

        out = _from_replce(bias_composition_tensor)
        return out

    # return legacy_relu(input, ref)
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    isolated_relu_out = torch.nn.functional.relu(input._residual_tensor)

    out = input.masked_fill(zero_mask, 0.0)
    relu_out = out._residual_tensor

    delta_relu_out = relu_out - isolated_relu_out
    decomposition_func = hybrid_decomposition
    bias_composition = decomposition_func(bias=delta_relu_out, context=input)
    out._residual_tensor = isolated_relu_out
    out = out + bias_composition
    return out
