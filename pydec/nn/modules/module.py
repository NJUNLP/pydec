from torch.nn import Module
from typing import TypeVar
from .meta import CopyModule

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


class Module(Module, DecModule, metaclass=CopyModule):
    __doc__ = Module.__doc__
