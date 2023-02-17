from __future__ import annotations

import torch
from . import core

from typing import Any, Dict, Union, List, Tuple, Sequence, Optional, Callable, overload
from torch import Tensor
from torch._C import memory_format
import pydec
from ._composition_str import _c_str
import types
import sys
from .overrides import (
    _register_builtin_function,
    _auto_registration,
    is_registered,
    dispatch_torch_function,
    get_customized_composition_methods,
)

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
    component_num_error,
    unsupported_operand_error,
    arg_value_error,
    none_decomposition_func_error,
)


from pydec.utils import _shift_dim, _shift_dims, parse_args

r"""
To avoid circular import, we have to initialize the following method in __init__.py.
"""


class Composition:
    r"""
    TODO: Composition doc
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not is_registered(func) or not all(
            issubclass(t, (torch.Tensor, pydec.Composition)) for t in types
        ):
            return NotImplemented
        return dispatch_torch_function(func)(*args, **kwargs)

    @property
    def requires_grad(self) -> _bool:
        return self._residual_tensor.requires_grad

    @property
    def shape(self) -> torch.Size:
        return self._residual_tensor.shape

    @property
    def device(self) -> _device:
        return self._residual_tensor.device

    @property
    def dtype(self) -> _dtype:
        return self._residual_tensor.dtype

    @property
    def T(self) -> Composition:
        return self.permute(*torch.arange(self.ndim - 1, -1, -1))

    @property
    def mT(self) -> Composition:
        return self.transpose(-2, -1)

    @property
    def ndim(self) -> _int:
        return self.dim()

    @property
    def components(self) -> Tensor:
        return self._component_tensor

    @property
    def residual(self) -> Tensor:
        return self._residual_tensor

    @overload
    def __init__(
        self, component_tensor: Tensor, residual_tensor: Tensor = None
    ) -> None:
        ...

    @overload
    def __init__(self, composition: Composition) -> None:
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        def init_from_tensor(component_tensor: Tensor, residual_tensor: Tensor = None):
            if residual_tensor is not None:
                if component_tensor.dtype != residual_tensor.dtype:
                    raise arg_value_error(
                        f"the dtype of component_tensor ({component_tensor.dtype}) should be the same as that of residual_tensor ({residual_tensor.dtype})"
                    )
                if component_tensor.device != residual_tensor.device:
                    raise arg_value_error(
                        f"the device of component_tensor ({component_tensor.device}) should be the same as that of residual_tensor ({residual_tensor.device})"
                    )
                if component_tensor.requires_grad != residual_tensor.requires_grad:
                    raise arg_value_error(
                        f"component_tensor.requires_grad ({component_tensor.requires_grad}) should be the same as residual_tensor.requires_grad ({residual_tensor.requires_grad})"
                    )

            # formal way but may cause problems, see https://github.com/pytorch/pytorch/issues/85094
            self._component_tensor = component_tensor.clone().detach()
            if residual_tensor is not None:
                if component_tensor.size()[1:] != residual_tensor.size():
                    raise size_error(
                        component_tensor.size()[1:],
                        residual_tensor.size(),
                        "composition",
                        "residual",
                    )
                self._residual_tensor = residual_tensor.clone().detach()
            else:
                self._residual_tensor = torch.zeros(component_tensor.size()[1:]).to(
                    component_tensor
                )

        def bind_customized_methods():
            method_dict = get_customized_composition_methods()
            for name, func in method_dict.items():
                setattr(self, name, types.MethodType(func, self))

        self._component_tensor: Tensor = None
        self._residual_tensor: Tensor = None
        input_kwargs = kwargs.copy()  # for error hint

        if len(args) == 0 and len(kwargs) == 0:
            # a private constructor to create a void composition with no data,
            # which is usually assigned in `_from_replace`.
            bind_customized_methods()
            return
        elif len(args) > 0:
            if isinstance(args[0], Composition):
                parse_args(args, ["composition"], kwargs)
            else:
                parse_args(args, ["component_tensor", "residual_tensor"], kwargs)
        elif len(args) > 2:
            raise args_error("Composition.__init__", args, input_kwargs)

        if "composition" in kwargs:
            c = kwargs["composition"]
            if isinstance(c, Composition):
                init_from_tensor(c._component_tensor, c._residual_tensor)
            else:
                raise args_error("Composition.__init__", args, input_kwargs)
        elif "component_tensor" in kwargs:
            if "residual_tensor" in kwargs and not isinstance(
                kwargs["residual_tensor"], Tensor
            ):
                raise args_error("Composition.__init__", args, input_kwargs)
            if isinstance(kwargs["component_tensor"], Tensor):
                init_from_tensor(**kwargs)
            else:
                raise args_error("Composition.__init__", args, input_kwargs)
        else:
            raise args_error("Composition.__init__", args, input_kwargs)

        bind_customized_methods()

    @_auto_registration
    def __getitem__(
        self, indices: Union[None, _int, slice, Tensor, List, Tuple]
    ) -> Composition:
        # support autotracing
        if pydec.autotracing.is_tracing_enabled():  # TODO
            if isinstance(indices, Tuple):
                indices = (slice(None, None, None),) + indices
            else:
                indices = (
                    slice(None, None, None),
                    indices,
                )
        return self.__c_getitem__(indices)

    @_auto_registration
    def __setitem__(
        self,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
        val: Union[Composition, Tensor, Number],
    ) -> None:
        # support autotracing
        if pydec.autotracing.is_tracing_enabled():  # TODO
            if isinstance(indices, Tuple):
                indices = (slice(None, None, None),) + indices
            else:
                indices = (
                    slice(None, None, None),
                    indices,
                )
        return self.__c_setitem__(indices, val)

    def __c_getitem__(
        self, indices: Union[None, _int, slice, Tensor, List, Tuple]
    ) -> Union[Composition, Tensor]:
        if isinstance(indices, (type(None), _int, slice, List, Tensor)):
            indices = (indices,)
        if indices[0] is None:
            raise arg_value_error(
                "The first dimension of indices should not be NoneType"
            )
        if isinstance(indices[0], _int):
            return self._component_tensor[indices]
        else:
            out_component_tensor = self._component_tensor[indices]
            out_residual_tensor = self._residual_tensor[indices[1:]]
            return pydec._from_replce(out_component_tensor, out_residual_tensor)

    def __c_setitem__(
        self,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
        val: Union[Composition, Tensor, Number],
    ) -> None:
        if isinstance(indices, (type(None), _int, slice, List, Tensor)):
            indices = (indices,)
        if indices[0] is None:
            raise arg_value_error(
                "The first dimension of indices should not be NoneType"
            )

        if isinstance(val, (Tensor, _int, _float, _bool)):
            self._component_tensor[indices] = val
            return
        if isinstance(val, (Composition)):
            if isinstance(indices[0], _int):
                raise arg_value_error(
                    f"Expected the assignment value to be (Tensor) or (Number), not ({type(val).__name__}) for single component assignment"
                )
            else:
                self._component_tensor[indices] = val._component_tensor
                self._residual_tensor[indices[1:]] = val._residual_tensor

    def __call__(self) -> _SliceComposition:
        """
        Shortcut for `__c_getitem__` and `__c_setitem__`, e.g., `c()[2]` equals to `c.__c_getitem__(2)`.
        """
        return _SliceComposition(self)

    @_auto_registration
    def __len__(self):
        # TODO: maybe same as the behavior of tensor
        return self._component_tensor.__len__()

    def __iter__(self):
        return self._component_tensor.__iter__()

    @_auto_registration
    def __reversed__(self):
        return self._component_tensor.__reversed__()

    @_auto_registration
    def __contains__(self, element):
        return self._component_tensor.__contains__(element)

    @_auto_registration
    def __repr__(self, *, composition_contents: List[str] = None) -> str:
        return _c_str(self, composition_contents=composition_contents)

    @_auto_registration
    def numel(self) -> _int:
        return pydec.numel(self)

    def c_numel(self, count_residual=False) -> _int:
        return pydec.c_numel(self, count_residual=count_residual)

    def numc(self) -> _int:
        return pydec.numc(self)

    @_auto_registration
    def clone(self, *, memory_format: Optional[memory_format] = None) -> Composition:
        return pydec.clone(input, memory_format=memory_format)

    @_auto_registration
    def detach(self) -> Composition:
        return pydec.detach(self)

    @_auto_registration
    def detach_(self) -> Composition:
        return pydec.detach_(self)

    @overload
    def size(self) -> torch.Size:
        ...

    @overload
    def size(self, dim: _int) -> _int:
        ...

    @overload
    def size(self, dim: Optional[_int] = None) -> Union[torch.Size, _int]:
        ...

    @_auto_registration
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
            return self._component_tensor.size()
        else:
            return self._component_tensor.size(dim)

    @_auto_registration
    def dim(self) -> _int:
        return self._residual_tensor.dim()

    @_auto_registration
    def __neg__(self) -> Composition:
        return pydec._from_replce(-self._component_tensor, -self._residual_tensor)

    @_auto_registration
    def __pos__(self) -> Composition:
        return pydec._from_replce(+self._component_tensor, +self._residual_tensor)

    @_auto_registration
    def __iadd__(self, other) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("+=", type(self), type(other))
        if isinstance(other, Composition):
            return core.decBLAS.cc_add_(self, other)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_add_(self, other)
        else:
            raise unsupported_operand_error("+=", type(self), type(other))

    @_auto_registration
    def __add__(self, other) -> Composition:
        try:
            return pydec.add(self, other)
        except TypeError:
            raise unsupported_operand_error("+", type(self), type(other))

    @_auto_registration
    def __radd__(self, other) -> Composition:
        try:
            return pydec.add(other, self)
        except TypeError:
            raise unsupported_operand_error("+", type(other), type(self))

    @_auto_registration
    def __sub__(self, other) -> Composition:
        try:
            return pydec.sub(self, other)
        except TypeError:
            raise unsupported_operand_error("-", type(self), type(other))

    @_auto_registration
    def __rsub__(self, other) -> Composition:
        try:
            return pydec.sub(other, self)
        except TypeError:
            raise unsupported_operand_error("-", type(other), type(self))

    @_auto_registration
    def __isub__(self, other) -> Composition:
        if not isinstance(self, Composition):
            raise unsupported_operand_error("-=", type(self), type(other))
        if isinstance(other, Composition):
            return core.decBLAS.cc_sub_(self, other)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_sub_(self, other)
        else:
            raise unsupported_operand_error("-=", type(self), type(other))

    @_auto_registration
    def __matmul__(self, other) -> Composition:
        if isinstance(other, Tensor):
            return core.decBLAS.ct_matmul(self, other)
        else:
            raise unsupported_operand_error("@", type(self), type(other))

    @_auto_registration
    def __rmatmul__(self, other) -> Composition:
        if isinstance(other, Tensor):
            return core.decBLAS.tc_matmul(other, self)
        else:
            raise unsupported_operand_error("@=", type(other), type(other))

    @_auto_registration
    def __imul__(self, other) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("*=", type(self), type(other))
        if isinstance(other, Composition):
            return core.decBLAS.cc_mul_(self, other)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_mul_(self, other)
        else:
            raise unsupported_operand_error("*=", type(self), type(other))

    @_auto_registration
    def __mul__(self, other) -> Composition:
        try:
            return pydec.mul(self, other)
        except TypeError:
            raise unsupported_operand_error("*", type(self), type(other))

    @_auto_registration
    def __rmul__(self, other) -> Composition:
        try:
            return pydec.mul(other, self)
        except TypeError:
            raise unsupported_operand_error("*", type(other), type(self))

    @_auto_registration
    def __itruediv__(self, other: Any) -> Composition:
        if not isinstance(self, Composition):
            raise unsupported_operand_error("/=", type(self), type(other))
        if isinstance(other, Composition):
            # TODO: not implement yet
            return unsupported_operand_error("/=", type(self), type(other))
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_div_(self, other)
        else:
            raise unsupported_operand_error("/=", type(self), type(other))

    @_auto_registration
    def __truediv__(self, other: Any) -> Composition:
        try:
            return pydec.div(self, other)
        except TypeError:
            raise unsupported_operand_error("/", type(self), type(other))

    @_auto_registration
    def __rtruediv__(self, other: Any) -> Composition:
        try:
            return pydec.div(other, self)
        except TypeError:
            raise unsupported_operand_error("/", type(other), type(self))

    @_auto_registration
    def __eq__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__eq__(self.c_sum())
        return self.c_sum().__eq__(other)

    @_auto_registration
    def __ne__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__ne__(self.c_sum())
        return self.c_sum().__ne__(other)

    @_auto_registration
    def __gt__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__lt__(self.c_sum())
        return self.c_sum().__gt__(other)

    @_auto_registration
    def __lt__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__gt__(self.c_sum())
        return self.c_sum().__lt__(other)

    @_auto_registration
    def __ge__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__le__(self.c_sum())
        return self.c_sum().__ge__(other)

    @_auto_registration
    def __le__(self, other: Any) -> Tensor:
        if isinstance(other, Composition):
            return other.__ge__(self.c_sum())
        return self.c_sum().__le__(other)

    @_auto_registration
    def add(
        self, other: Union[Composition, Tensor, Number], *, alpha: Optional[Number] = 1,
    ) -> Composition:
        """
        Note: although the function signature of `torch.Tensor.add` has parameter `out`, it is not working
        """
        return pydec.add(self, other, alpha=alpha)

    @_auto_registration
    def add_(
        self, other: Union[Composition, Tensor, Number], *, alpha: Optional[Number] = 1,
    ) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("add_", type(self), type(other))
        if isinstance(other, Composition):
            return core.decBLAS.cc_add_(self, other, alpha=alpha)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_add_(self, other, alpha=alpha)
        else:
            raise unsupported_operand_error("add_", type(self), type(other))

    @_auto_registration
    def sub(
        self, other: Union[Composition, Tensor, Number], *, alpha: Optional[Number] = 1,
    ) -> Composition:
        return pydec.sub(other, alpha=alpha)

    @_auto_registration
    def sub_(
        self, other: Union[Composition, Tensor, Number], *, alpha: Optional[Number] = 1
    ) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("sub_", type(self), type(other))
        if isinstance(other, Composition):
            return core.decBLAS.cc_sub_(self, other)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_sub_(self, other)
        else:
            raise unsupported_operand_error("sub_", type(self), type(other))

    @_auto_registration
    def mul(self, other: Union[Tensor, Number]) -> Composition:
        return pydec.mul(self, other)

    @_auto_registration
    def mul_(self, other: Union[Tensor, Number]) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("mul_", type(self), type(other))
        if isinstance(other, Composition):
            # TODO: not implement yet
            raise unsupported_operand_error("mul_", type(self), type(other))
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_mul_(self, other)
        else:
            raise unsupported_operand_error("mul_", type(self), type(other))

    @_auto_registration
    def div(
        self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None
    ) -> Composition:
        return pydec.div(self, other, rounding_mode=rounding_mode)

    @_auto_registration
    def div_(
        self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None
    ) -> Composition:
        if not isinstance(self, Composition):
            # TODO: maybe return a composition is a good choice
            raise unsupported_operand_error("div_", type(self), type(other))
        if isinstance(other, Composition):
            # TODO: not implement yet
            raise unsupported_operand_error("div_", type(self), type(other))
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            return core.decBLAS.ct_div_(self, other, rounding_mode=rounding_mode)
        else:
            raise unsupported_operand_error("div_", type(self), type(other))

    @_auto_registration
    def mv(self, vec: Tensor) -> Composition:
        return pydec.mv(self, vec)

    @_auto_registration
    def mm(self, mat2: Tensor) -> Composition:
        return pydec.mm(self, mat2)

    @overload
    def any(self) -> Tensor:
        ...

    @overload
    def any(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    @_auto_registration
    def any(self, *args: Any, **kwargs: Any) -> Tensor:
        return pydec.any(self, *args, **kwargs)

    @overload
    def all(self) -> Tensor:
        ...

    @overload
    def all(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    @_auto_registration
    def all(self, *args: Any, **kwargs: Any) -> Tensor:
        return pydec.all(self, *args, **kwargs)

    @_auto_registration
    def unsqueeze(self, dim: _int) -> Composition:
        return pydec.unsqueeze(self, dim)

    @overload
    def squeeze(self) -> Composition:
        ...

    @overload
    def squeeze(self, dim: _int) -> Composition:
        ...

    @_auto_registration
    def squeeze(self, dim: _int = None) -> Composition:
        return pydec.squeeze(self, dim)

    @_auto_registration
    def unsqueeze_(self, dim: _int) -> Composition:
        self._residual_tensor.unsqueeze_(dim)
        self._component_tensor.unsqueeze_(_shift_dim(dim))
        return self

    @overload
    def squeeze_(self) -> Composition:
        ...

    @overload
    def squeeze_(self, dim: _int) -> Composition:
        ...

    @_auto_registration
    def squeeze_(self, dim: _int = None) -> Composition:
        if dim is None:
            self._residual_tensor.squeeze_()
            if self.numc() == 1:
                self._component_tensor.squeeze_().unsqueeze_(0)
            else:
                self._component_tensor.squeeze_()
        else:
            self._residual_tensor.squeeze_(dim)
            self._component_tensor.squeeze_(_shift_dim(dim))
        return self

    @_auto_registration
    def transpose(self, dim0: _int, dim1: _int) -> Composition:
        return pydec.transpose(self, dim0, dim1)

    @_auto_registration
    def transpose_(self, dim0: _int, dim1: _int) -> Composition:
        self._residual_tensor.transpose_(dim0, dim1)
        self._component_tensor.transpose_(_shift_dim(dim0), _shift_dim(dim1))
        return self

    @overload
    def permute(self, dims: _size) -> Composition:
        ...

    @overload
    def permute(self, *dims: _int) -> Composition:
        ...

    @_auto_registration
    def permute(self, *args, **kwargs) -> Composition:
        if len(kwargs) == 1:
            dims = kwargs["dims"]
        elif isinstance(args[0], _int):
            dims = torch.Size(args)
        else:
            dims = args[0]
        return pydec.permute(self, dims=dims)

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

    @_auto_registration
    def sum(
        self, dim=None, keepdim: _bool = False, *, dtype: Optional[_dtype] = None
    ) -> Composition:
        return pydec.sum(dim=dim, keepdim=keepdim, dtype=dtype)

    def c_sum(self, *, dtype: Optional[_dtype] = None) -> Tensor:
        return pydec.c_sum(self, dtype=dtype)

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

    @_auto_registration
    def mean(
        self,
        dim: Union[_int, _size] = None,
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None,
    ):
        return pydec.mean(self, dim, keepdim, dtype=dtype)

    @overload
    def view(self, dtype: _dtype) -> Composition:
        ...

    @overload
    def view(self, size: _size) -> Composition:
        ...

    @overload
    def view(self, *size: _int) -> Composition:
        ...

    @_auto_registration
    def view(self, *args, dtype=None, size=None) -> Composition:
        if dtype is None and size is None:
            if isinstance(args[0], _dtype):
                dtype = args[0]
            elif isinstance(args[0], _int):
                size = torch.Size(args)
            else:
                size = args[0]

        if dtype is not None:
            out_component_tensor = self._component_tensor.view(dtype)
            out_residual_tensor = self._residual_tensor.view(dtype)
        else:
            out_component_tensor = self._component_tensor.view((self.numc(),) + size)
            out_residual_tensor = self._residual_tensor.view(size)
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @_auto_registration
    def view_as(self, other: Union[Tensor, Composition]) -> Composition:
        return self.view(other.size())

    @overload
    def reshape(self, shape: _size) -> Composition:
        ...

    @overload
    def reshape(self, *shape: _int) -> Composition:
        ...

    @_auto_registration
    def reshape(self, *args, shape=None) -> Composition:
        if shape is None:
            if isinstance(args[0], _int):
                shape = torch.Size(args)
            else:
                shape = args[0]
        return pydec.reshape(self, shape=shape)

    @_auto_registration
    def reshape_as(self, other: Tensor) -> Composition:
        return self.reshape_as(other.size())

    @_auto_registration
    def contiguous(self, memory_format=torch.contiguous_format) -> Composition:
        out_component_tensor = self._component_tensor.contiguous()
        out_residual_tensor = self._residual_tensor.contiguous()
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @_auto_registration
    def is_contiguous(self, memory_format=torch.contiguous_format) -> _bool:
        return (
            self._component_tensor.is_contiguous()
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

    @_auto_registration
    def to(self, *args, **kwargs) -> Composition:
        if isinstance(args[0], Composition):
            return self.to(args[0]._component_tensor, *args[1:], **kwargs)
        else:
            out_component_tensor = self._component_tensor.to(*args, **kwargs)
            out_residual_tensor = self._residual_tensor.to(*args, **kwargs)
            return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @overload
    def masked_fill(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def masked_fill(self, mask: Tensor, value: Number) -> Composition:
        ...

    @_auto_registration
    def masked_fill(self, mask: Tensor, value: Any) -> Composition:
        return pydec.masked_fill(self, mask, value)

    @overload
    def masked_fill_(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def masked_fill_(self, mask: Tensor, value: Number) -> Composition:
        ...

    @_auto_registration
    def masked_fill_(self, mask: Tensor, value: Any) -> Composition:
        r"""
        Unsafe.
        """
        self._component_tensor.masked_fill_(mask[None], value)
        self._residual_tensor.masked_fill_(mask, value)
        return self

    @overload
    def c_masked_fill(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def c_masked_fill(self, mask: Tensor, value: Number) -> Composition:
        ...

    def c_masked_fill(self, mask: Tensor, value: Any) -> Composition:
        return pydec.c_masked_fill(self, mask, value)

    @overload
    def c_masked_fill_(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def c_masked_fill_(self, mask: Tensor, value: Number) -> Composition:
        ...

    def c_masked_fill_(self, mask: Tensor, value: Any) -> Composition:
        if mask.dim() == 1:
            if len(mask) != self.numc():
                raise arg_value_error(
                    f"the length of mask ({len(mask)}) should match component number ({self.numc()})"
                )
            mask_size = (self.numc(),) + (1,) * self.dim()
            mask = mask.view(mask_size)
        self._component_tensor.masked_fill_(mask, value)
        return self

    @_auto_registration
    def masked_select(self, mask: Tensor) -> Composition:
        return pydec.masked_select(self, mask)

    @_auto_registration
    def masked_scatter(self, mask: Tensor, source: Tensor) -> Composition:
        return pydec.masked_scatter(self, mask, source)

    @_auto_registration
    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Composition:
        r"""
        Unsafe.
        """
        self._component_tensor.masked_scatter_(mask[None], source)
        self._residual_tensor.masked_scatter_(mask, source)
        return self

    @overload
    def gather(
        self, dim: _int, index: Tensor, *, sparse_grad: _bool = False
    ) -> Composition:
        ...

    @_auto_registration
    def gather(
        self, dim: Any, index: Tensor, *, sparse_grad: _bool = False
    ) -> Composition:
        return pydec.gather(self, dim, index, sparse_grad=sparse_grad)

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
    def scatter(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    @_auto_registration
    def scatter(
        self,
        dim: _int,
        index: Tensor,
        src: Any = None,
        value: Any = None,
        *,
        reduce: str = None,
    ) -> Composition:
        return pydec.scatter(self, dim, index, src, value, reduce=reduce)

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

    @_auto_registration
    def scatter_(
        self, dim: Any, index: Tensor, src: Any, *, reduce: str = None
    ) -> Composition:
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
                self._component_tensor.scatter_(_shift_dim(dim), c_index, c_src)
                self._residual_tensor.scatter_(dim, index, src)
            else:
                self._component_tensor.scatter_(
                    _shift_dim(dim), c_index, c_src, reduce=reduce
                )
                self._residual_tensor.scatter_(dim, index, src, reduce=reduce)
            return self

    @_auto_registration
    def diagonal_scatter(
        self, src: Tensor, offset: _int = 0, dim1: _int = 0, dim2: _int = 1
    ) -> Composition:
        return pydec.diagonal_scatter(self, src, offset, dim1, dim2)

    @_auto_registration
    def cuda(
        self,
        device: Optional[Union[_device, _int, str]] = None,
        non_blocking: _bool = False,
    ) -> Composition:
        out_component_tensor = self._component_tensor.cuda(
            device=device, non_blocking=non_blocking
        )
        out_residual_tensor = self._residual_tensor.cuda(
            device=device, non_blocking=non_blocking
        )
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @_auto_registration
    def cpu(self) -> Composition:
        out_component_tensor = self._component_tensor.cpu()
        out_residual_tensor = self._residual_tensor.cpu()
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    def is_cuda(self):
        return self._residual_tensor.is_cuda

    @overload
    def index_select(self, dim: _int, index: Tensor) -> Composition:
        ...

    @_auto_registration
    def index_select(self, dim: _int, index: Tensor) -> Composition:
        return pydec.index_select(self, dim=dim, index=index)

    @overload
    def c_index_select(self, index: Tensor, with_residual: _bool = True) -> Composition:
        ...

    def c_index_select(self, index: Tensor, with_residual: _bool = True) -> Composition:
        return pydec.c_index_select(self, index=index, with_residual=with_residual)

    @_auto_registration
    def masked_select(self, mask: Tensor) -> Composition:
        return pydec.masked_select(self, mask)

    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    @_auto_registration
    def index_fill(self, dim: _int, index: Tensor, value: Any) -> Composition:
        return pydec.index_fill(self, dim=dim, index=index, value=value)

    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    @_auto_registration
    def index_fill_(self, dim: _int, index: Tensor, value: Number) -> Composition:
        self._component_tensor.index_fill_(
            dim=_shift_dim(dim), index=index, value=value
        )
        self._residual_tensor.index_fill_(dim=dim, index=index, value=value)
        return self

    @overload
    def select(self, dim: _int, index: _int) -> Composition:
        ...

    @_auto_registration
    def select(self, dim: _int, index: _int) -> Composition:
        out_component_tensor = self._component_tensor.select(
            dim=_shift_dim(dim), index=index
        )
        out_residual_tensor = self._residual_tensor.select(dim=dim, index=index)
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @overload
    def type(self, dtype: None = None, non_blocking: _bool = False) -> str:
        ...

    @overload
    def type(self, dtype: Union[str, _dtype], non_blocking: _bool = False) -> Tensor:
        ...

    @_auto_registration
    def type(self, dtype=None, non_blocking: _bool = False):
        if dtype is None:
            return self._residual_tensor.type()
        else:
            out_component_tensor = self._component_tensor.type(
                dtype=dtype, non_blocking=non_blocking
            )
            out_residual_tensor = self._residual_tensor.type(
                dtype=dtype, non_blocking=non_blocking
            )
            return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @_auto_registration
    def type_as(self, other: Union[Tensor, Composition]) -> Composition:
        if isinstance(other, Composition):
            out_component_tensor = self._component_tensor.type_as(
                other._component_tensor
            )
            out_residual_tensor = self._residual_tensor.type_as(other._component_tensor)
        else:
            out_component_tensor = self._component_tensor.type_as(other)
            out_residual_tensor = self._residual_tensor.type_as(other)
        return pydec._from_replce(out_component_tensor, out_residual_tensor)

    @overload
    def round(self) -> Composition:
        ...

    @overload
    def round(self, *, decimals: _int) -> Composition:
        ...

    @_auto_registration
    def round(self, *, decimals: _int = None):
        return pydec.round(self, decimals=decimals)

    @overload
    def round_(self) -> Composition:
        ...

    @overload
    def round_(self, *, decimals: _int) -> Composition:
        ...

    @_auto_registration
    def round_(self, *, decimals: _int = None) -> Composition:
        return pydec.round_(self, decimals=decimals)

    @_auto_registration
    def abs(self) -> Composition:
        return pydec.abs(self)

    @_auto_registration
    def abs_(self) -> Composition:
        return pydec.abs_(self)

    @_auto_registration
    def requires_grad_(self, mode: _bool = True) -> Composition:
        self._component_tensor.requires_grad_(mode)
        self._residual_tensor.requires_grad_(mode)
        return self

    @_auto_registration
    def apply_(self, callable: Callable) -> Composition:
        self._component_tensor.apply_(callable)
        self._residual_tensor.apply_(callable)
        return self

    @_auto_registration
    def map_(self, composition: Composition, callable: Callable) -> Composition:
        self._residual_tensor.map_(composition._residual_tensor, callable)
        # permute to be broadcastable
        p_dims = [i for i in range(1, self._component_tensor.dim())] + [0]
        p_component_tensor = self._component_tensor.permute(*p_dims)
        p_dims = [i for i in range(1, composition._component_tensor.dim())] + [0]
        p_other_component_tensor = composition.permute(*p_dims)
        p_component_tensor.map_(p_other_component_tensor, callable)

        # recover
        p_dims = [-1] + [i for i in range(0, p_component_tensor.dim() - 1)]
        self._component_tensor = p_component_tensor.permute(p_dims)
        return self

    @_auto_registration
    def relu(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.relu(self, ref=ref)

    @_auto_registration
    def relu_(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.relu_(self, ref=ref)

    @_auto_registration
    def tanh(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.tanh(self, ref=ref)

    @_auto_registration
    def tanh_(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.tanh_(self, ref=ref)

    @_auto_registration
    def sigmoid(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.sigmoid(self, ref=ref)

    @_auto_registration
    def sigmoid_(self, *, ref: Optional[Tensor] = None) -> Composition:
        return pydec.sigmoid_(self, ref=ref)


class IndexComposition(Composition):
    """
    TODO
    """

    # we use this number to represent empty value
    # all indices should avoid this value
    MASK_NUM = -sys.maxsize

    @property
    def component_indices_mask(self) -> Tensor:
        return self.components == IndexComposition.MASK_NUM

    @property
    def residual_indices_mask(self) -> Tensor:
        return self.residual == IndexComposition.MASK_NUM

    @overload
    def __init__(
        self, component_tensor: Tensor, residual_tensor: Tensor = None,
    ) -> None:
        ...

    @overload
    def __init__(self, composition: IndexComposition) -> None:
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        if len(args) + len(kwargs) == 0:
            # a private constructor to create a void composition with no data,
            # which is usually assigned in `_from_replace`.
            return

        # Type checking
        accept_dtypes = [torch.int, torch.long]
        msg = "Expected arguments to have one of the following scalar types: {}; but got {} instead (while checking arguments for IndexComposition)"

        if self.dtype not in accept_dtypes:
            raise RuntimeError(msg.format(accept_dtypes, self.dtype))

    def __repr__(self, *, composition_contents: List[str] = None) -> str:
        # TODO
        return super().__repr__(composition_contents=composition_contents)


class _SliceComposition(Composition):
    r"""
    TODO: A temporary type for implementation of component subscript accessing for Composition.
    """

    def __init__(self, composition: Composition):
        super().__init__()  # void initialization
        self._component_tensor = composition._component_tensor
        self._residual_tensor = composition._residual_tensor

    def __getitem__(
        self, indices: Union[None, _int, slice, Tensor, List, Tuple]
    ) -> Union[Composition, Tensor]:
        return super().__c_getitem__(indices)

    def __setitem__(
        self,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
        val: Union[Composition, Tensor, Number],
    ) -> None:
        return super().__c_setitem__(indices, val)
