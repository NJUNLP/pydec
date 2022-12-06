import pydec
import torch.nn as nn
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

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Tracer`.
T = TypeVar("T", bound="Tracer")


class Tracer(nn.Module):
    def __init__(self, model: nn.Module = None, runtime_compiler_context=None) -> None:
        super().__init__()

        self.model = model
        self.runtime_compiler_context = runtime_compiler_context
        self.tracing = True

    def forward(self, *args, **kwargs):
        assert self.model is not None
        assert self.runtime_compiler_context is not None
        with pydec.set_tracing_enabled(self.tracing):
            # TODO: Consider always performing runtime compilation to support type error warnings.
            # See compiler.py `_api_wrapper`.
            if self.tracing:
                with self.runtime_compiler_context:
                    return self.model(*args, **kwargs)
            else:
                return self.model(*args, **kwargs)

    def trace(self: T, mode: bool = True) -> T:
        r"""Sets the module in forward tracing mode.
        """
        if not isinstance(mode, bool):
            raise ValueError("tracing mode is expected to be boolean")
        self.tracing = mode
        return self
