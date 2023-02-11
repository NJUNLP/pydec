r"""Functional interface"""

import torch
import pydec
from torch import Tensor
import torch.nn.functional as F
from .._composition import Composition, IndexComposition
from ..decomposition import (
    get_decomposition_func,
    get_decomposition_name,
)
from ..overrides import _auto_registration, _register_builtin_function
from ..exception_utils import none_decomposition_func_error, arg_value_error

from torch.nn.functional import _no_grad_embedding_renorm_

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
        out += bias
        return out
    else:
        return out


@_auto_registration
def layer_norm(
    input: Composition,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    ref: Optional[Tensor] = None,
) -> Tensor:
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """

    normalized_dims = tuple(range(-len(normalized_shape), 0))
    input_mean = input.mean(dim=normalized_dims, keepdim=True)
    if ref is None:
        ref = input.c_sum()
    input_std = torch.sqrt(
        torch.var(ref, dim=normalized_dims, unbiased=False, keepdim=True) + eps
    )
    out = (input - input_mean) * weight / input_std

    if bias is not None:
        out += bias
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
        out_component_tensor = F.conv2d(
            input._component_tensor, weight, None, stride, padding, dilation, groups,
        )
        out_residual_tensor = F.conv2d(
            input._residual_tensor, weight, None, stride, padding, dilation, groups,
        )
    else:
        out_component_tensor = F.conv2d(
            input._component_tensor.view((-1,) + input.size()[1:]),
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
    return pydec._from_replce(out_component_tensor, out_residual_tensor)


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


@_auto_registration
def dropout(
    input: Composition, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            "dropout probability has to be between 0 and 1, " "but got {}".format(p)
        )
    if not training:
        return input

    drop_mask = torch.ones(input.size(), device=input.devce, dtype=input.dtype)
    drop_mask = torch._VF.dropout_(drop_mask, p, training)
    if inplace:
        input *= drop_mask
        return input
    else:
        return input * drop_mask


@_auto_registration
def embedding(
    input: IndexComposition,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Composition:
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    Args:
        input (IndexComposition): IndexComposition containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
          where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0,2,0,5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)

    component_tensor_mask = input._component_tensor == IndexComposition.MASK_NUM
    residual_tensor_mask = input._residual_tensor == IndexComposition.MASK_NUM
    component_tensor = input._component_tensor.masked_fill(
        component_tensor_mask, padding_idx
    )
    residual_tensor = input._residual_tensor.masked_fill(
        residual_tensor_mask, padding_idx
    )
    out_component_tensor = torch.embedding(
        weight, component_tensor, padding_idx, scale_grad_by_freq, sparse
    )
    out_residual_tensor = torch.embedding(
        weight, residual_tensor, padding_idx, scale_grad_by_freq, sparse
    )
    out_component_tensor[component_tensor_mask] = 0
    out_residual_tensor[residual_tensor_mask] = 0
    return pydec._from_replce(out_component_tensor, out_residual_tensor)


def legacy_relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    residual_out = torch.nn.functional.relu(input._residual_tensor)
    out = input.masked_fill(zero_mask, 0.0)
    masked_residual_out = out._residual_tensor
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:

        delta_context = pydec._from_replce(input._component_tensor, residual_out)
        delta_out = decomposition_func(
            input=delta_context, func=lambda x: x, ref=masked_residual_out
        )
        out._component_tensor += delta_out._component_tensor
        out._residual_tensor = delta_out._residual_tensor
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())
