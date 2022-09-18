from __future__ import annotations

import torch
from typing import Any, Union, List, Tuple, Sequence, Optional, Callable, overload
from torch import Tensor
from torch._C import memory_format

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

from pydec.exception_utils import (
    args_error,
    size_error,
    composition_num_error,
    unsupported_operand_error,
    arg_value_error,
    none_bias_decomposition_func_error,
)


from pydec.utils import _shift_dim, _shift_dims


r"""
To avoid circular import, we have to initialize the following method in __init__.py.
"""


def _from_replce(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    ...

def _get_bias_decomposition_name() -> str:...

def _get_bias_decomposition_func(inplace: bool = False) -> Callable[..., Composition]:
    ...


class Composition:
    __doc__ = r"""
    Composition doc
    """

    @overload
    def __init__(self, size: _size, composition_num: _int, **kwargs: Any) -> None:
        ...

    @overload
    def __init__(
        self, composition_tensor: Tensor, residual_tensor: Tensor = None
    ) -> None:
        ...

    @overload
    def __init__(self, composition: Composition) -> None:
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        def init_from_tensor(
            composition_tensor: Tensor, residual_tensor: Tensor = None
        ):
            self._composition_tensor = torch.tensor(composition_tensor).to(
                composition_tensor
            )
            if residual_tensor is not None:
                if composition_tensor.size()[1:] != residual_tensor.size():
                    raise size_error(
                        composition_tensor.size()[1:],
                        residual_tensor.size(),
                        "composition",
                        "residual",
                    )
                self._residual_tensor = torch.tensor(residual_tensor).to(
                    residual_tensor
                )
            else:
                self._residual_tensor = torch.zeros(composition_tensor.size()[1:]).to(
                    composition_tensor
                )

        if isinstance(args[0], (torch.Size, list, Tuple)):
            if len(args) != 2:
                raise args_error(Composition.__init__.__name__, args, kwargs)
            size, composition_num = args
            self._composition_tensor = torch.zeros((composition_num,) + size, **kwargs)
            self._residual_tensor: Tensor = torch.zeros(size, **kwargs)
        elif isinstance(args[0], Tensor):
            if len(args) != 2 or len(kwargs) != 0:
                raise args_error(Composition.__init__.__name__, args, kwargs)
            composition_tensor, residual_tensor = args
            init_from_tensor(composition_tensor, residual_tensor)
        elif isinstance(args[0], Composition):
            if len(args) != 1 or len(kwargs) != 0:
                raise args_error(Composition.__init__.__name__, args, kwargs)
            c: Composition = args[0]
            init_from_tensor(c._composition_tensor, c._residual_tensor)
        else:
            raise args_error(Composition.__init__.__name__, args, kwargs)

    def __getitem__(
        self, indices: Union[None, _int, slice, Tensor, List, Tuple]
    ) -> Tensor:
        return self._composition_tensor[indices]

    def __setitem__(
        self,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
        val: Union[Tensor, Number],
    ) -> None:
        self._composition_tensor[indices] = val

    def __len__(self):
        return self._composition_tensor.__len__()

    def __iter__(self):
        return self._composition_tensor.__iter__()

    def __reversed__(self):
        return self._composition_tensor.__reversed__()

    def __contains__(self, element):
        return self._composition_tensor.__contains__(element)

    def __repr__(self, *, tensor_contents=None) -> str:
        import itertools

        composition_hint = [f"composition {i}:\n" for i in range(len(self))]
        composition_hint.append("residual:\n")
        tensor_str = [repr(t) + "\n" for t in self]
        tensor_str.append(repr(self._residual_tensor))
        zipped_str = zip(composition_hint, tensor_str)
        merged_str = list(itertools.chain(*zipped_str))
        return "".join(merged_str)

    def numel(self) -> _int:
        return self._residual_tensor.numel()

    def numc(self) -> _int:
        return len(self)

    def clone(self, *, memory_format: Optional[memory_format] = None) -> Composition:
        out = Composition(tuple(), 0)
        out._composition_tensor = self._composition_tensor.clone(
            memory_format=memory_format
        )
        out._residual_tensor = self._residual_tensor.clone(memory_format=memory_format)
        return out

    def detach(self) -> Composition:
        out = Composition(tuple(), 0)
        out._composition_tensor = self._composition_tensor.detach()
        out._residual_tensor = self._residual_tensor.detach()
        return out

    @overload
    def size(self) -> torch.Size:
        ...

    @overload
    def size(self, dim: _int) -> _int:
        ...

    def size(self, dim: Optional[_int] = None) -> Union[torch.Size, _int]:
        if dim is None:
            return self.c_size()[1:]
        else:
            return self.size()[dim]

    @overload
    def c_size(self) -> torch.Size:
        ...

    @overload
    def c_size(self, dim: _int) -> _int:
        ...

    def c_size(self, dim: Optional[_int] = None) -> Union[torch.Size, _int]:
        if dim is None:
            return self._composition_tensor.size()
        else:
            return self._composition_tensor.size(dim)

    def __neg__(self) -> Composition:
        return _from_replce(-self._composition_tensor, -self._residual_tensor)

    def __pos__(self) -> Composition:
        return _from_replce(+self._composition_tensor, +self._residual_tensor)

    def __iadd__(self, other) -> Composition:
        if isinstance(other, Composition):
            if self.composition_num() != other.composition_num():
                raise composition_num_error(
                    self.composition_num(), other.composition_num
                )
            self._composition_tensor += other.composition_tensor
            self._residual_tensor += other._residual_tensor
            return self
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            decomposition_func = _get_bias_decomposition_func(inplace=True)
            if decomposition_func is not None:
                self = decomposition_func(self, other)
            else:
                raise none_bias_decomposition_func_error(_get_bias_decomposition_name())
            return self
        else:
            raise unsupported_operand_error("+=", type(self), type(other))

    def __add__(self, other) -> Composition:
        if isinstance(other, Composition):
            if self.composition_num() != other.composition_num():
                raise composition_num_error(
                    self.composition_num(), other.composition_num
                )
            out_composition_tensor = self._composition_tensor + other.composition_tensor
            out_residual_tensor = self._residual_tensor + other._residual_tensor
            return _from_replce(out_composition_tensor, out_residual_tensor)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            decomposition_func = _get_bias_decomposition_func()
            if decomposition_func is not None:
                return decomposition_func(self, other)
            else:
                raise none_bias_decomposition_func_error(_get_bias_decomposition_name())
        else:
            raise unsupported_operand_error("+", type(self), type(other))

    def __radd__(self, other) -> Composition:
        try:
            return self + other
        except TypeError:
            raise unsupported_operand_error("+", type(other), type(self))

    def __sub__(self, other) -> Composition:
        try:
            return self + (-other)
        except TypeError:
            raise unsupported_operand_error("-", type(self), type(other))

    def __rsub__(self, other) -> Composition:
        try:
            return other + (-self)
        except TypeError:
            raise unsupported_operand_error("-", type(self), type(other))

    def __isub__(self, other) -> Composition:
        try:
            self += -other
        except TypeError:
            raise unsupported_operand_error("-=", type(self), type(other))
        return self

    def __imatmul__(self, other) -> Composition:
        if isinstance(other, Tensor):
            self._composition_tensor @= other
            self._residual_tensor @= other
            return self
        else:
            raise unsupported_operand_error("@=", type(self), type(other))

    def __matmul__(self, other) -> Composition:
        if isinstance(other, Tensor):
            out_composition_tensor = self._composition_tensor @ other
            out_residual_tensor = self._residual_tensor @ other
            return _from_replce(out_composition_tensor, out_residual_tensor)
        else:
            raise unsupported_operand_error("@", type(self), type(other))

    def __imul__(self, other) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("*=", type(self), type(other))
        self._composition_tensor *= other
        self._residual_tensor *= other
        return self

    def __mul__(self, other) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("*", type(self), type(other))
        out_composition_tensor = self._composition_tensor * other
        out_residual_tensor = self._residual_tensor * other
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def __rmul__(self, other) -> Composition:
        try:
            return self * other
        except TypeError:
            raise unsupported_operand_error("*", type(other), type(self))

    def __eq__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__eq__(self.c_sum())
        return self.c_sum().__eq__(other)

    def __ne__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__ne__(self.c_sum())
        return self.c_sum().__ne__(other)

    def __gt__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__lt__(self.c_sum())
        return self.c_sum().__gt__(other)

    def __lt__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__gt__(self.c_sum())
        return self.c_sum().__lt__(other)

    def __ge__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__le__(self.c_sum())
        return self.c_sum().__ge__(other)

    def __le__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__ge__(self.c_sum())
        return self.c_sum().__le__(other)

    @overload
    def any(self) -> Tensor:
        ...

    @overload
    def any(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    @overload
    def any(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor:
        ...

    def any(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) + len(kwargs) == 0:
            return self.c_sum().any()
        else:
            return self.c_sum().any(*args, **kwargs)

    @overload
    def all(self) -> Tensor:
        ...

    @overload
    def all(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    @overload
    def all(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor:
        ...

    def all(self, *args: Any, **kwargs: Any) -> Tensor:
        if len(args) + len(kwargs) == 0:
            return self.c_sum().all()
        else:
            return self.c_sum().all(*args, **kwargs)

    def unsqueeze(self, dim: _int) -> Composition:
        out_residual_tensor = self._residual_tensor.unsqueeze(dim)
        out_composition_tensor = self._composition_tensor.unsqueeze(_shift_dim(dim))
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def squeeze(self, dim: _int) -> Composition:
        out_residual_tensor = self._residual_tensor.squeeze(dim)
        out_composition_tensor = self._composition_tensor.squeeze(_shift_dim(dim))
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def unsqueeze_(self, dim: _int) -> Composition:
        self._residual_tensor.unsqueeze_(dim)
        self._composition_tensor.unsqueeze_(_shift_dim(dim))
        return self

    def squeeze_(self, dim: _int) -> Composition:
        self._residual_tensor.squeeze_(dim)
        self._composition_tensor.squeeze_(_shift_dim(dim))
        return self

    def transpose(self, dim0: _int, dim1: _int) -> Composition:
        out_residual_tensor = self._residual_tensor.transpose(dim0, dim1)
        out_composition_tensor = self._composition_tensor.transpose(
            _shift_dim(dim0), _shift_dim(dim1)
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def transpose_(self, dim0: _int, dim1: _int) -> Composition:
        self._residual_tensor.transpose_(dim0, dim1)
        self._composition_tensor.transpose_(_shift_dim(dim0), _shift_dim(dim1))
        return self

    @overload
    def permute(self, dims: _size) -> Composition:
        ...

    @overload
    def permute(self, *dims: _int) -> Composition:
        ...

    def permute(self, *args, **kwargs) -> Composition:
        if len(args) == 1:
            dims = args[0]
        elif len(kwargs) == 1:
            dims = kwargs["dims"]
        out_residual_tensor = self._residual_tensor.permute(dims)
        out_composition_tensor = self._composition_tensor.permute(
            (0,) + _shift_dims(dims)
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def sum(self, *, dtype: Optional[_dtype] = None) -> Composition:
        ...

    @overload
    def sum(
        self,
        dim: Union[_int, _size],
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None,
    ) -> Composition:
        ...

    def sum(
        self, dim=None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None
    ) -> Composition:
        if dim is None:
            dim = tuple(range(1, self._composition_tensor.dim()))
            out_composition_tensor = self._composition_tensor.sum(dim, dtype=dtype)
            out_residual_tensor = self._residual_tensor.sum(dtype=dtype)
        else:
            out_residual_tensor = self._residual_tensor.sum(
                dim=dim, keepdim=keepdim, dtype=dtype
            )
            if isinstance(dim, _int):
                dim = (dim,)
            out_composition_tensor = self._composition_tensor.sum(
                dim=_shift_dims(dim), keepdim=keepdim, dtype=dtype
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def c_sum(self, *, dtype: Optional[_dtype] = None) -> Tensor:
        return self._composition_tensor.sum(
            dim=0, dtype=dtype
        ) + self._residual_tensor.to(dtype=dtype)

    @overload
    def mean(self, *, dtype: Optional[_dtype] = None) -> Composition:
        ...

    @overload
    def mean(
        self,
        dim: Union[_int, _size],
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None,
    ) -> Composition:
        ...

    def mean(
        self,
        dim: Union[_int, _size] = None,
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None,
    ):
        if dim is None:
            dim = tuple(range(1, self._composition_tensor.dim()))
            out_composition_tensor = self._composition_tensor.mean(dim, dtype=dtype)
            out_residual_tensor = self._residual_tensor.mean(dtype=dtype)
        else:
            out_residual_tensor = self._residual_tensor.mean(
                dim=dim, keepdim=keepdim, dtype=dtype
            )
            if isinstance(dim, _int):
                dim = (dim,)
            out_composition_tensor = self._composition_tensor.mean(
                dim=_shift_dims(dim), keepdim=keepdim, dtype=dtype
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def view(self, dtype: _dtype) -> Composition:
        ...

    @overload
    def view(self, size: _size) -> Composition:
        ...

    @overload
    def view(self, *size: _int) -> Composition:
        ...

    def view(self, *args, dtype=None, size=None) -> Composition:
        if dtype is None and size is None:
            if isinstance(args[0], _dtype):
                dtype = args[0]
            elif isinstance(args[0], _int):
                size = torch.Size(args)
            else:
                size = args[0]

        if dtype is not None:
            out_composition_tensor = self._composition_tensor.view(dtype)
            out_residual_tensor = self._residual_tensor.view(dtype)
        else:
            out_composition_tensor = self._composition_tensor.view(
                (self.numc(),) + size
            )
            out_residual_tensor = self._residual_tensor.view(size)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def view_as(self, other: Union[Tensor, Composition]) -> Composition:
        return self.view(other.size())

    @overload
    def reshape(self, shape: _size) -> Composition:
        ...

    @overload
    def reshape(self, *shape: _int) -> Composition:
        ...

    def reshape(self, *args, shape=None) -> Composition:
        if len(args) > 1:
            out_composition_tensor = self._composition_tensor.reshape(
                self.numc(), *args
            )
            out_residual_tensor = self._residual_tensor.reshape(*args)
        else:
            if shape is None:
                shape = args[0]
            out_composition_tensor = self._composition_tensor.view(
                (self.numc(),) + shape
            )
            out_residual_tensor = self._residual_tensor.view(shape)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def reshape_as(self, other: Tensor) -> Composition:
        return self.reshape_as(other.size())

    def contiguous(self, memory_format=torch.contiguous_format) -> Composition:
        out_composition_tensor = self._composition_tensor.contiguous()
        out_residual_tensor = self._residual_tensor.contiguous()
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def is_contiguous(self, memory_format=torch.contiguous_format) -> _bool:
        return (
            self._composition_tensor.is_contiguous()
            and self._residual_tensor.is_contiguous()
        )

    @overload
    def to(
        self, dtype: _dtype, non_blocking: _bool = False, copy: _bool = False
    ) -> Composition:
        ...

    @overload
    def to(
        self,
        device: Optional[Union[_device, str]] = None,
        dtype: Optional[_dtype] = None,
        non_blocking: _bool = False,
        copy: _bool = False,
    ) -> Composition:
        ...

    @overload
    def to(
        self,
        other: Union[Tensor, Composition],
        non_blocking: _bool = False,
        copy: _bool = False,
    ) -> Composition:
        ...

    def to(self, *args, **kwargs) -> Composition:
        if isinstance(args[0], Composition):
            return self.to(args[0]._composition_tensor, *args[1:], **kwargs)
        else:
            out_composition_tensor = self._composition_tensor.to(*args, **kwargs)
            out_residual_tensor = self._residual_tensor.to(*args, **kwargs)
            return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def masked_fill(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def masked_fill(self, mask: Tensor, value: Number) -> Composition:
        ...

    def masked_fill(self, mask: Tensor, value: Any) -> Composition:
        """
        Unsafe.
        """
        out_composition_tensor = self._composition_tensor.masked_fill(mask[None], value)
        out_residual_tensor = self._residual_tensor.masked_fill(mask, value)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def c_masked_fill(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def c_masked_fill(self, mask: Tensor, value: Number) -> Composition:
        ...

    def c_masked_fill(self, mask: Tensor, value: Any) -> Composition:
        out_composition_tensor = self._composition_tensor.masked_fill(mask, value)
        out_residual_tensor = self._residual_tensor.clone()
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def c_masked_fill_(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def c_masked_fill_(self, mask: Tensor, value: Number) -> Composition:
        ...

    def c_masked_fill_(self, mask: Tensor, value: Any) -> Composition:
        self._composition_tensor.masked_fill_(mask, value)
        return self

    @overload
    def masked_fill_(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def masked_fill_(self, mask: Tensor, value: Number) -> Composition:
        ...

    def masked_fill_(self, mask: Tensor, value: Any) -> Composition:
        """
        Unsafe.
        """
        self._composition_tensor.masked_fill_(mask[None], value)
        self._residual_tensor.masked_fill_(mask, value)
        return self

    def masked_select(self, mask: Tensor) -> Composition:
        out_composition_tensor = self._composition_tensor.masked_select(
            mask[None]
        ).reshape(self.numc(), -1)
        out_residual_tensor = self._residual_tensor.masked_select(mask)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def masked_scatter(self, mask: Tensor, source: Tensor) -> Composition:
        """
        Unsafe.
        """
        out_composition_tensor = self._composition_tensor.masked_scatter(
            mask[None], source
        )
        out_residual_tensor = self._residual_tensor.masked_scatter(mask, source)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Composition:
        """
        Unsafe.
        """
        self._composition_tensor.masked_scatter_(mask[None], source)
        self._residual_tensor.masked_scatter_(mask, source)
        return self

    @overload
    def gather(
        self, dim: _int, index: Tensor, *, sparse_grad: _bool = False
    ) -> Composition:
        ...

    @overload
    def gather(
        self,
        dim: Union[str, ellipsis, None],
        index: Tensor,
        *,
        sparse_grad: _bool = False,
    ) -> Composition:
        ...

    def gather(
        self, dim: Any, index: Tensor, *, sparse_grad: _bool = False
    ) -> Composition:
        c_index = index[None].expand((self.numc(),) + (-1,) * index.dim())
        out_composition_tensor = self._composition_tensor.gather(
            _shift_dim(dim), c_index, sparse_grad=sparse_grad
        )
        out_residual_tensor = self._residual_tensor.gather(
            dim, index, sparse_grad=sparse_grad
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def scatter(self, dim: _int, index: Tensor, src: Tensor) -> Composition:
        ...

    @overload
    def scatter(
        self, dim: _int, index: Tensor, src: Tensor, *, reduce: str
    ) -> Composition:
        ...

    @overload
    def scatter(
        self, dim: _int, index: Tensor, value: Number, *, reduce: str
    ) -> Composition:
        ...

    @overload
    def scatter(
        self, dim: Union[str, ellipsis, None], index: Tensor, src: Tensor
    ) -> Composition:
        ...

    @overload
    def scatter(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    @overload
    def scatter(
        self, dim: Union[str, ellipsis, None], index: Tensor, value: Number
    ) -> Composition:
        ...

    def scatter(
        self, dim: Any, index: Tensor, src: Any, *, reduce: str = None
    ) -> Composition:
        """
        Unsafe.
        Safe when reduce is not None.
        """
        if reduce == "add":
            holder = torch.zeros_like(self._residual_tensor).to(self._residual_tensor)
            holder = holder.scatter(dim, index, src, reduce=reduce)
            return self + holder
        else:
            c_index = index[None].expand((self.numc(),) + (-1,) * index.dim())
            if isinstance(src, Tensor):
                c_src = src[None].expand((self.numc(),) + (-1,) * src.dim())
            else:
                c_src = src
            if reduce is None:
                out_composition_tensor = self._composition_tensor.scatter(
                    _shift_dim(dim), c_index, c_src
                )
                out_residual_tensor = self._residual_tensor.scatter(dim, index, src)
            else:
                out_composition_tensor = self._composition_tensor.scatter(
                    _shift_dim(dim), c_index, c_src, reduce=reduce
                )
                out_residual_tensor = self._residual_tensor.scatter(
                    dim, index, src, reduce=reduce
                )
            return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor) -> Tensor:
        ...

    @overload
    def scatter_(self, dim: _int, index: Tensor, src: Tensor, *, reduce: str) -> Tensor:
        ...

    @overload
    def scatter_(
        self, dim: _int, index: Tensor, value: Number, *, reduce: str
    ) -> Tensor:
        ...

    @overload
    def scatter_(self, dim: _int, index: Tensor, value: Number) -> Tensor:
        ...

    def scatter_(
        self, dim: Any, index: Tensor, src: Any, *, reduce: str = None
    ) -> Composition:
        """
        Unsafe.
        """
        if reduce == "add":
            holder = torch.zeros_like(self._residual_tensor).to(self._residual_tensor)
            holder = holder.scatter(dim, index, src, reduce=reduce)
            self += holder
            return self
        else:
            c_index = index[None].expand((self.numc(),) + (-1,) * index.dim())
            if isinstance(src, Tensor):
                c_src = src[None].expand((self.numc(),) + (-1,) * src.dim())
            else:
                c_src = src
            if reduce is None:
                self._composition_tensor.scatter_(_shift_dim(dim), c_index, c_src)
                self._residual_tensor.scatter_(dim, index, src)
            else:
                self._composition_tensor.scatter_(
                    _shift_dim(dim), c_index, c_src, reduce=reduce
                )
                self._residual_tensor.scatter_(dim, index, src, reduce=reduce)
            return self

    def diagonal_scatter(
        self, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1
    ) -> Composition:
        c_src = src[None].expand((self.numc(),) + (-1,) * src.dim())
        out_composition_tensor = self._composition_tensor.diagonal_scatter(
            c_src, offset, _shift_dim(dim1), _shift_dim(dim2)
        )
        out_residual_tensor = self._residual_tensor.diagonal_scatter(
            src, offset, dim1, dim2
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def cuda(
        self,
        device: Optional[Union[_device, _int, str]] = None,
        non_blocking: _bool = False,
    ) -> Composition:
        out_composition_tensor = self._composition_tensor.cuda(
            device=device, non_blocking=non_blocking
        )
        out_residual_tensor = self._residual_tensor.cuda(
            device=device, non_blocking=non_blocking
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def cpu(self) -> Composition:
        out_composition_tensor = self._composition_tensor.cpu()
        out_residual_tensor = self._residual_tensor.cpu()
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def is_cuda(self):
        return self._composition_tensor.is_cuda and self._residual_tensor.is_cuda

    @overload
    def index_select(self, dim: _int, index: Tensor) -> Tensor:
        ...

    @overload
    def index_select(self, dim: Union[str, ellipsis, None], index: Tensor) -> Tensor:
        ...

    def index_select(self):
        ...

    def masked_select(self, mask: Tensor) -> Tensor:
        ...


# original
# bsz * self.num_heads x tgt_num x src_num
# bsz * self.num_heads x src_num x head_dim

# composition
# bsz * self.num_heads x tgt_num x src_num
# all_len x bsz * self.num_heads x src_num x head_dim


# connection
# bsz * self.num_heads x tgt_num x src_num
# src_num x all_len x bsz * self.num_heads x src_num x head_dim


# bsz * self.num_heads x 1       x tgt_num x src_num x 1
# bsz * self.num_heads x src_num x 1       x src_num x head_dim

# bsz * self.num_heads x src_num x tgt_num x src_num x head_dim
# or bsz * self.num_heads x all_value_len x tgt_num x value_len x head_dim
