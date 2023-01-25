from torch.autograd.grad_mode import _DecoratorContextManager
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

from typing import Any

__all__ = ["no_grad", "enable_grad", "set_grad_enabled", "inference_mode"]


class _TracingMode:
    is_enabled = False

    @classmethod
    def set_enabled(cls, enabled: _bool):
        cls.is_enabled = enabled


def is_tracing_enabled() -> _bool:
    return _TracingMode.is_enabled


class no_tracing(_DecoratorContextManager):
    r"""Context-manager that disabled forward tracing."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = is_tracing_enabled()
        _TracingMode.set_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _TracingMode.set_enabled(self.prev)


class enable_tracing(_DecoratorContextManager):
    r"""Context-manager that enables forward tracing."""

    def __enter__(self) -> None:
        self.prev = is_tracing_enabled()
        _TracingMode.set_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _TracingMode.set_enabled(self.prev)


class set_tracing_enabled(_DecoratorContextManager):
    r"""Context-manager that sets forward tracing to on or off."""

    def __init__(self, mode: bool) -> None:
        self.prev = is_tracing_enabled()
        _TracingMode.set_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _TracingMode.set_enabled(self.prev)

    def clone(self):
        return self.__class__(self.mode)
