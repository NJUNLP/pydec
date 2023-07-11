r"""Functional interface"""

import torch
import pydec
from torch import Tensor
import torch.nn.functional as F
from .._composition import Composition, IndexComposition
from pydec.core.decOVF import (
    get_decomposition_func,
    get_decomposition_name,
)
from pydec import core
from ..overrides import _auto_registration, _register_builtin_function
from ..exception_utils import none_decomposition_func_error, arg_value_error

from torch.nn.functional import (  # type: ignore[attr-defined]
    _no_grad_embedding_renorm_,
    _get_softmax_dim,
    _mha_shape_check,
    _in_projection_packed,
    _in_projection,
    pad,
    # _scaled_dot_product_attention,
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
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

import warnings

T = TypeVar("T")


def _add_docstr(obj: T, doc_obj: str) -> T:
    obj.__doc__ = doc_obj
    return obj


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
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
    *,
    ref: Optional[Tensor] = None,
) -> Tensor:
    r"""
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if inplace:
        result = core.decOVF.leaky_relu_(input, negative_slope)
    else:
        result = core.decOVF.leaky_relu(input, negative_slope)
    return result


@_auto_registration
def leaky_relu_(
    input: Composition, negative_slope: float = 0.01, *, ref: Optional[Tensor] = None
) -> Composition:
    return core.decOVF.leaky_relu_(input, negative_slope, ref=ref)


@_auto_registration
def gelu(
    input: Composition, approximate: str = "none", *, ref: Optional[Tensor] = None
) -> Composition:
    return core.decOVF.gelu(input, approximate, ref=ref)


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
def linear(
    input: Composition, weight: Tensor, bias: Optional[Tensor] = None
) -> Composition:
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
    *,
    ref: Optional[Tensor] = None,
    linearize: bool = True,
) -> Composition:
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    if linearize:
        return layer_norm_linearized(
            input, normalized_shape, weight, bias, eps, ref=ref
        )
    normalized_dims = tuple(range(-len(normalized_shape), 0))
    input_mean = input.mean(dim=normalized_dims, keepdim=True)
    if ref is None:
        ref = input.c_sum()
    input_std = pydec.sqrt(
        pydec.var(input, dim=normalized_dims, unbiased=False, keepdim=True, ref=ref)
        + eps,
    )
    if weight is not None:
        out = (input - input_mean) * weight / input_std
    else:
        out = (input - input_mean) / input_std

    if bias is not None:
        out += bias
        return out
    else:
        return out


def layer_norm_linearized(
    input: Composition,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
    *,
    ref: Optional[Tensor] = None,
) -> Composition:
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
    if weight is not None:
        out = (input - input_mean) * weight / input_std
    else:
        out = (input - input_mean) / input_std

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
            input._component_tensor,
            weight,
            None,
            stride,
            padding,
            dilation,
            groups,
        )
        out_residual_tensor = F.conv2d(
            input._residual_tensor,
            weight,
            None,
            stride,
            padding,
            dilation,
            groups,
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
            input._residual_tensor,
            weight,
            None,
            stride,
            padding,
            dilation,
            groups,
        )
    out_residual_tensor += bias
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


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
    # TODO: api name is duplicated with torch.nn.functional.max_pool2d_with_indices
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

    component_empty_mask, residual_empty_mask = input.empty_mask
    component_tensor = input._component_tensor.masked_fill(
        component_empty_mask, padding_idx
    )
    residual_tensor = input._residual_tensor.masked_fill(
        residual_empty_mask, padding_idx
    )
    out_component_tensor = torch.embedding(
        weight, component_tensor, padding_idx, scale_grad_by_freq, sparse
    )
    out_residual_tensor = torch.embedding(
        weight, residual_tensor, padding_idx, scale_grad_by_freq, sparse
    )
    out_component_tensor[component_empty_mask] = 0
    out_residual_tensor[residual_empty_mask] = 0
    return pydec.as_composition(out_component_tensor, out_residual_tensor)


def legacy_relu(input: Composition, ref: Optional[Tensor] = None) -> Composition:
    # TODO: maybe deprecated after we have scaling decomposition
    if ref is None:
        ref = input.c_sum()
    zero_mask = ref < 0
    residual_out = torch.nn.functional.relu(input._residual_tensor)
    out = input.masked_fill(zero_mask, 0.0)
    masked_residual_out = out._residual_tensor
    decomposition_func = get_decomposition_func()
    if decomposition_func is not None:
        delta_context = pydec.as_composition(input._component_tensor, residual_out)
        delta_out = decomposition_func(
            input=delta_context, func=lambda x: x, ref=masked_residual_out
        )
        out._component_tensor += delta_out._component_tensor
        out._residual_tensor = delta_out._residual_tensor
        return out
    else:
        raise none_decomposition_func_error(get_decomposition_name())


@_auto_registration
def softmax(
    input: Composition,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None,
) -> Composition:
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Args:
        input (Composition): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned composition.
          If specified, the input composition is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret


@_auto_registration
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    return NotImplemented

    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert (
                attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
            ), f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (
            bsz,
            src_len,
        ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )
    attn_output = (
        attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    )
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
