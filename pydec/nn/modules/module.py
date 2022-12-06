import pydec
from functools import wraps
from ...autotracing.compiler import _is_input_composition
from ...exception_utils import args_error_not_in_tracing
from torch.nn import Module
import types

from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)

T = TypeVar("T", bound="DecModule")


class DecModule:
    r"""This base class is mainly used for type hints.
    Most of the processing happens in metaclass `ProxyModule`.
    """

    def pydec_forward(self, *args, **kwargs):
        r"""Defines the computation performed in tracing.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            f'Module [{type(self).__name__}] is missing the required "pydec_forward" function'
        )

    def convert_from(cls: T, obj) -> T:
        r"""Converting a torch Module to pydec Module.

        Warning: The `__init__` of the pydec Module will not be invoked.
        Just use it in auto tracing.
        """
        raise NotImplementedError(
            f'Module [{cls.__name__}] is missing the required "convert_from" function'
        )


class ProxyModule(type):
    r"""This metaclass ..."""

    def __new__(cls, clsname, bases, attrs):
        base = bases[0]

        def forward_wrapper(obj, torch_forward, pydec_forward):
            @wraps(torch_forward)
            def meta_forward(self, *args, **kwargs):
                c_mode = _is_input_composition(*args, **kwargs)
                if c_mode and not pydec.is_tracing_enabled():
                    # Maybe just warning.
                    raise args_error_not_in_tracing(*args, **kwargs)
                if c_mode:
                    return pydec_forward(*args, **kwargs)
                else:
                    return torch_forward(*args, **kwargs)

            return types.MethodType(meta_forward, obj)

        user_init = attrs["__init__"] if "__init__" in attrs else None

        def meta_init(self, *args, **kwargs):
            assert isinstance(self, base)
            base.__init__(self, *args, **kwargs)
            if user_init is not None:
                user_init(self, *args, **kwargs)

            torch_forward = self.forward
            pydec_forward = self.pydec_forward
            setattr(
                self, "forward", forward_wrapper(self, torch_forward, pydec_forward)
            )

        def convert_from(cls, obj):
            assert isinstance(obj, base)
            torch_forward = obj.forward
            pydec_forward = types.MethodType(cls.pydec_forward, obj)

            setattr(obj, "forward", forward_wrapper(obj, torch_forward, pydec_forward))
            obj.__class__ = cls
            return obj

        attrs["__init__"] = meta_init

        attrs["convert_from"] = classmethod(convert_from)

        return super().__new__(cls, clsname, bases, attrs)


class CopyModule(type):
    def __new__(cls, clsname, bases, attrs):
        base = bases[0]
        user_init = attrs["__init__"] if "__init__" in attrs else None

        def meta_init(self, *args, **kwargs):
            assert isinstance(self, base)
            base.__init__(self, *args, **kwargs)
            if user_init is not None:
                user_init(self, *args, **kwargs)

        def convert_from(cls, obj):
            assert isinstance(obj, base)
            obj.__class__ = cls
            return obj

        attrs["__init__"] = meta_init

        attrs["convert_from"] = classmethod(convert_from)

        return super().__new__(cls, clsname, bases, attrs)


class Module(Module, DecModule, metaclass=CopyModule):
    __doc__ = Module.__doc__
