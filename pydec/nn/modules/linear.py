from .meta import ProxyModule, CopyModule
from .module import DecModule
from pydec import Composition
import torch.nn as nn
from .. import functional as F
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

__all__ = [
    "Identity",
    "Linear",
]


class Identity(nn.Identity, DecModule, metaclass=CopyModule):
    __doc__ = nn.Identity.__doc__


class Linear(nn.Linear, DecModule, metaclass=ProxyModule):
    def pydec_forward(self, input: Composition) -> Composition:
        return F.linear(input, weight=self.weight, bias=self.bias)
