from __future__ import annotations

import torch
from typing import Any, Dict, Union, List, Tuple, Sequence, Optional, Callable, overload
from torch import Tensor
from torch._C import memory_format
import pydec
from ._composition_str import _c_str
import types
import warnings

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
)


from pydec.utils import _shift_dim, _shift_dims


r"""
To avoid circular import, we have to initialize the following method in __init__.py.
"""


def _from_replce(
    composition_tensor: Tensor, residual_tensor: Tensor = None
) -> Composition:
    ...


_registered_method_dict = {}


def _c_register_method(name: str, func: Callable):
    # this function is not public, invoked by pydec.autotracing.library (stacklevel=3)
    if name in _registered_method_dict:
        warnings.warn("override registered method ({})".format(name), stacklevel=3)
    _registered_method_dict[name] = func


class _CompositionMeta(type):
    r"""
    This metaclass is used to support customized method registration for Composition.
    """

    def __new__(cls, clsname, bases, attrs):
        assert len(bases) == 0
        c_init = attrs["__init__"] if "__init__" in attrs else None

        def meta_init(self, *args, **kwargs):
            if c_init is not None:
                c_init(self, *args, **kwargs)

            # do registeration
            for name, func in _registered_method_dict.items():
                setattr(self, name, types.MethodType(func, self))

        attrs["__init__"] = meta_init

        return super().__new__(cls, clsname, bases, attrs)


class Composition(metaclass=_CompositionMeta):
    __doc__ = r"""
    TODO: Composition doc
    """

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
            if residual_tensor is not None:
                if composition_tensor.dtype != residual_tensor.dtype:
                    raise arg_value_error(
                        f"the dtype of composition_tensor ({composition_tensor.dtype}) should be the same as that of residual_tensor ({residual_tensor.dtype})."
                    )
                if composition_tensor.device != residual_tensor.device:
                    raise arg_value_error(
                        f"the device of composition_tensor ({composition_tensor.device}) should be the same as that of residual_tensor ({residual_tensor.device})."
                    )
                if composition_tensor.requires_grad != residual_tensor.requires_grad:
                    raise arg_value_error(
                        f"composition_tensor.requires_grad ({composition_tensor.requires_grad}) should be the same as residual_tensor.requires_grad ({residual_tensor.requires_grad})."
                    )

            # formal way but may cause problems, see https://github.com/pytorch/pytorch/issues/85094
            self._composition_tensor = composition_tensor.clone().detach()
            if residual_tensor is not None:
                if composition_tensor.size()[1:] != residual_tensor.size():
                    raise size_error(
                        composition_tensor.size()[1:],
                        residual_tensor.size(),
                        "composition",
                        "residual",
                    )
                self._residual_tensor = residual_tensor.clone().detach()
            else:
                self._residual_tensor = torch.zeros(composition_tensor.size()[1:]).to(
                    composition_tensor
                )

        def parse_args(args: list, key_list: List):
            for i in range(len(args)):
                kwargs[key_list[i]] = args[i]

        self._composition_tensor: Tensor = None
        self._residual_tensor: Tensor = None

        if len(args) == 1:
            if isinstance(args[0], Composition):
                parse_args(args, ["composition"])
            elif isinstance(args[0], Tensor):
                parse_args(args, ["composition_tensor"])
            else:
                raise args_error("Composition.__init__", args, kwargs)
        elif len(args) == 2:
            if isinstance(args[0], Tensor) and isinstance(args[1], Tensor):
                parse_args(args, ["composition_tensor", "residual_tensor"])
            else:
                raise args_error("Composition.__init__", args, kwargs)
        elif len(args) > 2:
            raise args_error("Composition.__init__", args, kwargs)

        if "composition" in kwargs:
            c: Composition = kwargs["composition"]
            init_from_tensor(c._composition_tensor, c._residual_tensor)
        elif "composition_tensor" in kwargs:
            init_from_tensor(**kwargs)
        else:
            raise args_error("Composition.__init__", args, kwargs)

    def __getitem__(
        self, indices: Union[None, _int, slice, Tensor, List, Tuple]
    ) -> Union[Composition, Tensor]:
        # support autotracing
        if pydec.autotracing.is_tracing_enabled():
            if isinstance(indices, Tuple):
                indices = (slice(None, None, None),) + indices
            else:
                indices = (
                    slice(None, None, None),
                    indices,
                )

        if isinstance(indices, (type(None), _int, slice, List, Tensor)):
            indices = (indices,)
        if indices[0] is None:
            raise arg_value_error(
                "The first dimension of indices should not be NoneType."
            )
        if isinstance(indices[0], _int):
            return self._composition_tensor[indices]
        else:
            out_composition_tensor = self._composition_tensor[indices]
            out_residual_tensor = self._residual_tensor[indices[1:]]
            return _from_replce(out_composition_tensor, out_residual_tensor)

    def __setitem__(
        self,
        indices: Union[None, _int, slice, Tensor, List, Tuple],
        val: Union[Composition, Tensor, Number],
    ) -> None:
        # support autotracing
        if pydec.autotracing.is_tracing_enabled():
            if isinstance(indices, Tuple):
                indices = (slice(None, None, None),) + indices
            else:
                indices = (
                    slice(None, None, None),
                    indices,
                )

        if isinstance(indices, (type(None), _int, slice, List, Tensor)):
            indices = (indices,)
        if indices[0] is None:
            raise arg_value_error(
                "The first dimension of indices should not be NoneType."
            )

        if isinstance(val, (Tensor, _int, _float, _bool)):
            self._composition_tensor[indices] = val
            return
        if isinstance(val, (Composition)):
            if isinstance(indices[0], _int):
                raise arg_value_error(
                    f"A single component assignment is being made, and the right part of the equal sign should be (Tensor) or (Number), not ({type(val).__name__})."
                )
            else:
                self._composition_tensor[indices] = val._composition_tensor
                self._residual_tensor[indices[1:]] = val._residual_tensor

    def __len__(self):
        return self._composition_tensor.__len__()

    def __iter__(self):
        return self._composition_tensor.__iter__()

    def __reversed__(self):
        return self._composition_tensor.__reversed__()

    def __contains__(self, element):
        return self._composition_tensor.__contains__(element)

    def __repr__(self, *, composition_contents: List[str] = None) -> str:
        return _c_str(self, composition_contents=composition_contents)

    def numel(self) -> _int:
        return self._residual_tensor.numel()

    def c_numel(self, count_residual=False) -> _int:
        if count_residual:
            return self._composition_tensor.numel() + self._residual_tensor.numel()
        else:
            return self._composition_tensor.numel()

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

    def detach_(self) -> Composition:
        self._composition_tensor.detach_()
        self._residual_tensor.detach_()
        return self

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

    def dim(self) -> _int:
        return self._residual_tensor.dim()

    def __neg__(self) -> Composition:
        return _from_replce(-self._composition_tensor, -self._residual_tensor)

    def __pos__(self) -> Composition:
        return _from_replce(+self._composition_tensor, +self._residual_tensor)

    def __iadd__(self, other) -> Composition:
        if isinstance(other, Composition):
            if self.numc() != other.numc():
                raise component_num_error(self.numc(), other.numc())
            self._composition_tensor += other._composition_tensor
            self._residual_tensor += other._residual_tensor
            return self
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            self._residual_tensor += other
            return self
        else:
            raise unsupported_operand_error("+=", type(self), type(other))

    def __add__(self, other) -> Composition:
        if isinstance(other, Composition):
            if self.numc() != other.numc():
                raise component_num_error(self.numc(), other.numc())
            out_composition_tensor = (
                self._composition_tensor + other._composition_tensor
            )
            out_residual_tensor = self._residual_tensor + other._residual_tensor
            return _from_replce(out_composition_tensor, out_residual_tensor)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            out_composition_tensor = self._composition_tensor.clone()
            out_residual_tensor = self._residual_tensor + other
            return _from_replce(out_composition_tensor, out_residual_tensor)
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
            raise unsupported_operand_error("-", type(other), type(self))

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

    def __rmatmul__(self, other) -> Composition:
        if isinstance(other, Tensor):
            if self.dim() == 1:
                # if the composition_tensor's ndim is 2, the component dim
                # will be incorrectly included in the multiplication
                out_composition_tensor = other @ self._composition_tensor.unsqueeze(-1)
                out_composition_tensor.squeeze_(-1)
                out_residual_tensor = other @ self._residual_tensor
            else:
                out_composition_tensor = other @ self._composition_tensor
                out_residual_tensor = other @ self._residual_tensor
            return _from_replce(out_composition_tensor, out_residual_tensor)
        else:
            raise unsupported_operand_error("@=", type(self), type(other))

    def __imul__(self, other) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("*=", type(self), type(other))
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                self._composition_tensor = self._composition_tensor.view(new_size)
            self._composition_tensor *= other
            self._residual_tensor *= other
        else:
            self._composition_tensor *= other
            self._residual_tensor *= other
        return self

    def __mul__(self, other) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("*", type(self), type(other))
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                out_composition_tensor = self._composition_tensor.view(new_size) * other
            else:
                out_composition_tensor = self._composition_tensor * other
            out_residual_tensor = self._residual_tensor * other
        else:
            out_composition_tensor = self._composition_tensor * other
            out_residual_tensor = self._residual_tensor * other
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def __rmul__(self, other) -> Composition:
        try:
            return self * other
        except TypeError:
            raise unsupported_operand_error("*", type(other), type(self))

    def __itruediv__(self, other: Any) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("/=", type(self), type(other))
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                self._composition_tensor = self._composition_tensor.view(new_size)
            self._composition_tensor /= other
            self._residual_tensor /= other
        else:
            self._composition_tensor /= other
            self._residual_tensor /= other
        return self

    def __truediv__(self, other: Any) -> Composition:
        if isinstance(other, Composition):
            raise unsupported_operand_error("/", type(self), type(other))
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                out_composition_tensor = self._composition_tensor.view(new_size) / other
            else:
                out_composition_tensor = self._composition_tensor / other
            out_residual_tensor = self._residual_tensor / other
        else:
            out_composition_tensor = self._composition_tensor / other
            out_residual_tensor = self._residual_tensor / other
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def __rtruediv__(self, other: Any) -> Composition:
        try:
            return self * other
        except TypeError:
            raise unsupported_operand_error("/", type(other), type(self))

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

    def add(
        self,
        other: Union[Composition, Tensor, Number],
        *,
        alpha: Optional[Number] = 1,
        out: Optional[Composition] = None,
        **kwargs,
    ) -> Composition:
        if isinstance(other, Composition):
            if self.numc() != other.numc():
                raise component_num_error(self.numc(), other.numc())
            if out is None:
                out_composition_tensor = self._composition_tensor.add(
                    other._composition_tensor, alpha=alpha
                )
                out_residual_tensor = self._residual_tensor.add(
                    other._residual_tensor, alpha=alpha
                )
            else:
                out_composition_tensor = self._composition_tensor.add(
                    other._composition_tensor, alpha=alpha, out=out._composition_tensor
                )
                out_residual_tensor = self._residual_tensor.add(
                    other._residual_tensor, alpha=alpha, out=out._residual_tensor
                )
            return _from_replce(out_composition_tensor, out_residual_tensor)
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            out_composition_tensor = self._composition_tensor.clone()
            out_residual_tensor = self._residual_tensor.add(
                other, alpha=alpha, out=out._residual_tensor
            )
            if out is not None:
                out._composition_tensor[:] = out_composition_tensor
            return _from_replce(out_composition_tensor, out_residual_tensor)
        else:
            raise unsupported_operand_error("add", type(self), type(other))

    def add_(
        self,
        other: Union[Composition, Tensor, Number],
        *,
        alpha: Optional[Number] = 1,
        **kwargs,
    ) -> Composition:
        if isinstance(other, Composition):
            if self.numc() != other.numc():
                raise component_num_error(self.numc(), other.numc())
            self._composition_tensor.add_(other.composition_tensor, alpha=alpha)
            self._residual_tensor.add_(other._residual_tensor, alpha=alpha)
            return self
        elif isinstance(other, (_int, _float, _bool, Tensor)):
            self._residual_tensor.add_(other, alpha=alpha)
            return self
        else:
            raise unsupported_operand_error("add_", type(self), type(other))

    def sub(
        self,
        other: Union[Composition, Tensor, Number],
        *,
        alpha: Optional[Number] = 1,
        out: Optional[Composition] = None,
        **kwargs,
    ) -> Composition:
        return self.add(-other, alpha=alpha, out=out, **kwargs)

    def sub_(
        self, other: Union[Composition, Tensor, Number], *, alpha: Optional[Number] = 1
    ) -> Composition:
        return self.add_(-other, alpha=alpha)

    def mul(
        self, other: Union[Tensor, Number], *, out: Optional[Composition] = None
    ) -> Composition:
        if isinstance(other, Composition):
            raise args_error(Composition.mul.__name__, self, other, out=out)
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                out_composition_tensor = self._composition_tensor.view(new_size).mul(
                    other, out=out._composition_tensor
                )
            else:
                out_composition_tensor = self._composition_tensor.mul(
                    other, out=out._composition_tensor
                )
            out_residual_tensor = self._residual_tensor.mul(
                other, out=out._residual_tensor
            )
        else:
            out_composition_tensor = self._composition_tensor * other
            out_residual_tensor = self._residual_tensor * other
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def mul_(self, other: Union[Tensor, Number]) -> Composition:
        self *= other
        return self

    def div(
        self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None
    ) -> Composition:
        if isinstance(other, Composition):
            raise args_error(
                Composition.div.__name__, self, other, rounding_mode=rounding_mode
            )
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                out_composition_tensor = self._composition_tensor.view(new_size).div(
                    other, rounding_mode=rounding_mode
                )
            else:
                out_composition_tensor = self._composition_tensor.div(
                    other, rounding_mode == rounding_mode
                )
            out_residual_tensor = self._residual_tensor.div(
                other, rounding_mode=rounding_mode
            )
        else:
            out_composition_tensor = self._composition_tensor.div(
                other, rounding_mode=rounding_mode
            )
            out_residual_tensor = self._residual_tensor.div(
                other, rounding_mode=rounding_mode
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def div_(
        self, other: Union[Tensor, Number], *, rounding_mode: Optional[str] = None
    ) -> Composition:
        if isinstance(other, Composition):
            raise args_error(
                Composition.div_.__name__, self, other, rounding_mode=rounding_mode
            )
        if isinstance(other, Tensor):
            if other.dim() > self.dim():
                new_size = (
                    (self.numc(),) + (1,) * (other.dim() - self.dim()) + self.size()
                )
                self._composition_tensor.view(new_size).div_(
                    other, rounding_mode=rounding_mode
                )
            else:
                self._composition_tensor.div_(other, rounding_mode == rounding_mode)
            self._residual_tensor.div_(other, rounding_mode=rounding_mode)
        else:
            self._composition_tensor.div_(other, rounding_mode=rounding_mode)
            self._residual_tensor.div_(other, rounding_mode=rounding_mode)
        return self

    def mv(self, vec: Tensor) -> Composition:
        r"""
        Note that the use of Tensor.mv(Composition) is not supported and may raise an error in autoracing.
        Use pydec.mv() instead or use torch.mv() in autotracing instead.
        """
        out_residual_tensor = self._residual_tensor.mv(vec)
        out_composition_tensor = self._composition_tensor @ vec
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def mm(self, mat2: Tensor) -> Composition:
        r"""
        Note that the use of Tensor.mm(Composition) is not supported and may raise an error in autoracing.
        Use pydec.mm() instead or use torch.mm() in autotracing instead.
        """
        out_residual_tensor = self._residual_tensor.mm(mat2)
        out_composition_tensor = self._composition_tensor @ mat2
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def any(self) -> Tensor:
        ...

    @overload
    def any(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    def any(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.c_sum().any(*args, **kwargs)

    @overload
    def all(self) -> Tensor:
        ...

    @overload
    def all(self, dim: _int, keepdim: _bool = False) -> Tensor:
        ...

    def all(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.c_sum().all(*args, **kwargs)

    def unsqueeze(self, dim: _int) -> Composition:
        out_residual_tensor = self._residual_tensor.unsqueeze(dim)
        out_composition_tensor = self._composition_tensor.unsqueeze(_shift_dim(dim))
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def squeeze(self) -> Composition:
        ...

    @overload
    def squeeze(self, dim: _int) -> Composition:
        ...

    def squeeze(self, dim: _int = None) -> Composition:
        if dim is None:
            out_residual_tensor = self._residual_tensor.squeeze()
            out_composition_tensor = self._composition_tensor.squeeze()
            if self.numc() == 1:
                out_composition_tensor = out_composition_tensor.unsqueeze(0)
        else:
            out_residual_tensor = self._residual_tensor.squeeze(dim)
            out_composition_tensor = self._composition_tensor.squeeze(_shift_dim(dim))
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def unsqueeze_(self, dim: _int) -> Composition:
        self._residual_tensor.unsqueeze_(dim)
        self._composition_tensor.unsqueeze_(_shift_dim(dim))
        return self

    @overload
    def squeeze_(self) -> Composition:
        ...

    @overload
    def squeeze_(self, dim: _int) -> Composition:
        ...

    def squeeze_(self, dim: _int = None) -> Composition:
        if dim is None:
            self._residual_tensor.squeeze_()
            if self.numc() == 1:
                self._composition_tensor.squeeze_().unsqueeze_(0)
            else:
                self._composition_tensor.squeeze_()
        else:
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
        if len(kwargs) == 1:
            dims = kwargs["dims"]
        elif isinstance(args[0], _int):
            dims = torch.Size(args)
        else:
            dims = args[0]
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
        if shape is None:
            if isinstance(args[0], _int):
                shape = torch.Size(args)
            else:
                shape = args[0]
        out_composition_tensor = self._composition_tensor.view((self.numc(),) + shape)
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
        r"""
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
        if mask.dim() == 1:
            if len(mask) != self.numc():
                raise arg_value_error(
                    f"the length of mask ({len(mask)}) should match component number ({self.numc()})"
                )
            mask_size = (self.numc(),) + (1,) * self.dim()
            mask = mask.view(mask_size)
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
        if mask.dim() == 1:
            if len(mask) != self.numc():
                raise arg_value_error(
                    f"the length of mask ({len(mask)}) should match component number ({self.numc()})"
                )
            mask_size = (self.numc(),) + (1,) * self.dim()
            mask = mask.view(mask_size)
        self._composition_tensor.masked_fill_(mask, value)
        return self

    @overload
    def masked_fill_(self, mask: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def masked_fill_(self, mask: Tensor, value: Number) -> Composition:
        ...

    def masked_fill_(self, mask: Tensor, value: Any) -> Composition:
        r"""
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
        r"""
        Unsafe.
        """
        out_composition_tensor = self._composition_tensor.masked_scatter(
            mask[None], source
        )
        out_residual_tensor = self._residual_tensor.masked_scatter(mask, source)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def masked_scatter_(self, mask: Tensor, source: Tensor) -> Composition:
        r"""
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
    def scatter(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    def scatter(
        self,
        dim: _int,
        index: Tensor,
        src: Any = None,
        value: Any = None,
        *,
        reduce: str = None,
    ) -> Composition:
        r"""
        Unsafe.
        Safe when reduce is not None.
        """
        if src is None:
            src = value
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
        r"""
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
        r"""
        Unsafe.
        """
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
    def index_select(self, dim: _int, index: Tensor) -> Composition:
        ...

    def index_select(self, dim: _int, index: Tensor) -> Composition:
        out_composition_tensor = self._composition_tensor.index_select(
            dim=_shift_dim(dim), index=index
        )
        out_residual_tensor = self._residual_tensor.index_select(dim=dim, index=index)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def c_index_select(self, index: Tensor, with_residual: _bool = True) -> Composition:
        ...

    def c_index_select(self, index: Tensor, with_residual: _bool = True) -> Composition:
        out_composition_tensor = self._composition_tensor.index_select(
            dim=0, index=index
        )
        if with_residual:
            out_residual_tensor = self._residual_tensor.clone()
        else:
            out_residual_tensor = torch.zeros_like(self._residual_tensor).to(
                self._residual_tensor
            )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def masked_select(self, mask: Tensor) -> Composition:
        out_composition_tensor = self._composition_tensor.masked_select(mask=mask[None])
        out_residual_tensor = self._residual_tensor.masked_select(mask=mask)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def index_fill(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    def index_fill(self, dim: _int, index: Tensor, value: Any) -> Composition:
        r"""
        Unsafe.
        """
        out_composition_tensor = self._composition_tensor.index_fill(
            dim=_shift_dim(dim), index=index, value=value
        )
        out_residual_tensor = self._residual_tensor.index_fill(
            dim=dim, index=index, value=value
        )
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Tensor) -> Composition:
        ...

    @overload
    def index_fill_(self, dim: _int, index: Tensor, value: Number) -> Composition:
        ...

    def index_fill_(self, dim: _int, index: Tensor, value: Number) -> Composition:
        r"""
        Unsafe.
        """
        self._composition_tensor.index_fill_(
            dim=_shift_dim(dim), index=index, value=value
        )
        self._residual_tensor.index_fill_(dim=dim, index=index, value=value)
        return self

    @overload
    def select(self, dim: _int, index: _int) -> Composition:
        ...

    def select(self, dim: _int, index: _int) -> Composition:
        out_composition_tensor = self._composition_tensor.select(
            dim=_shift_dim(dim), index=index
        )
        out_residual_tensor = self._residual_tensor.select(dim=dim, index=index)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def type(self, dtype: None = None, non_blocking: _bool = False) -> str:
        ...

    @overload
    def type(self, dtype: Union[str, _dtype], non_blocking: _bool = False) -> Tensor:
        ...

    def type(self, dtype=None, non_blocking: _bool = False):
        if dtype is None:
            return self._residual_tensor.type()
        else:
            out_composition_tensor = self._composition_tensor.type(
                dtype=dtype, non_blocking=non_blocking
            )
            out_residual_tensor = self._residual_tensor.type(
                dtype=dtype, non_blocking=non_blocking
            )
            return _from_replce(out_composition_tensor, out_residual_tensor)

    def type_as(self, other: Union[Tensor, Composition]) -> Composition:
        if isinstance(other, Composition):
            out_composition_tensor = self._composition_tensor.type_as(
                other._composition_tensor
            )
            out_residual_tensor = self._residual_tensor.type_as(
                other._composition_tensor
            )
        else:
            out_composition_tensor = self._composition_tensor.type_as(other)
            out_residual_tensor = self._residual_tensor.type_as(other)
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def round(self) -> Composition:
        ...

    @overload
    def round(self, *, decimals: _int) -> Composition:
        ...

    def round(self, *, decimals: _int = None):
        if decimals is not None:
            out_composition_tensor = self._composition_tensor.round(decimals=decimals)
            out_residual_tensor = self._residual_tensor.round(decimals=decimals)
        else:
            out_composition_tensor = self._composition_tensor.round()
            out_residual_tensor = self._residual_tensor.round()
        return _from_replce(out_composition_tensor, out_residual_tensor)

    @overload
    def round_(self) -> Composition:
        ...

    @overload
    def round_(self, *, decimals: _int) -> Composition:
        ...

    def round_(self, *, decimals: _int = None) -> Composition:
        if decimals is not None:
            self._composition_tensor.round_(decimals=decimals)
            self._residual_tensor.round_(decimals=decimals)
        else:
            self._composition_tensor.round_()
            self._residual_tensor.round_()
        return self

    def abs(self) -> Composition:
        out_composition_tensor = self._composition_tensor.abs()
        out_residual_tensor = self._residual_tensor.abs()
        return _from_replce(out_composition_tensor, out_residual_tensor)

    def abs_(self) -> Composition:
        self._composition_tensor.abs_()
        self._residual_tensor.abs_()
        return self

    def requires_grad_(self, mode: _bool = True) -> Composition:
        self._composition_tensor.requires_grad_(mode)
        self._residual_tensor.requires_grad_(mode)
        return self

    def apply_(self, callable: Callable) -> Composition:
        self._composition_tensor.apply_(callable)
        self._residual_tensor.apply_(callable)
        return self

    def map_(self, composition: Composition, callable: Callable) -> Composition:
        self._residual_tensor.map_(composition._residual_tensor, callable)
        # permute to be broadcastable
        p_dims = [i for i in range(1, self._composition_tensor.dim())] + [0]
        p_composition_tensor = self._composition_tensor.permute(*p_dims)
        p_dims = [i for i in range(1, composition._composition_tensor.dim())] + [0]
        p_other_composition_tensor = composition.permute(*p_dims)
        p_composition_tensor.map_(p_other_composition_tensor, callable)

        # recover
        p_dims = [-1] + [i for i in range(0, p_composition_tensor.dim() - 1)]
        self._composition_tensor = p_composition_tensor.permute(p_dims)
        return self
