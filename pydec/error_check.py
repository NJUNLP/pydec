import torch
from torch import Tensor

from typing import Union, Any, Callable, ContextManager, TYPE_CHECKING

from .composition import Composition

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

from .exception_utils import size_error, overflow_error, arg_value_error


class _ErrorCheckState:
    bypass_check: bool = False


class no_error_check(ContextManager):
    def __init__(self) -> None:
        self.prev = None

    def __enter__(self):
        self.prev = _ErrorCheckState.bypass_check
        _ErrorCheckState.bypass_check = True

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        _ErrorCheckState.bypass_check = self.prev


class error_check(ContextManager):
    def __init__(self) -> None:
        self.prev = None

    def __enter__(self):
        self.prev = _ErrorCheckState.bypass_check
        _ErrorCheckState.bypass_check = False

    def __exit__(
        self,
        __exc_type,
        __exc_value,
        __traceback,
    ):
        _ErrorCheckState.bypass_check = self.prev


def is_error_checking_enabled():
    return not _ErrorCheckState.bypass_check


def check_error(
    c: Composition,
    ref: Tensor,
    composition_mask: Tensor = None,
    error_tensor_mask: Tensor = None,
    error_bound=1e-2,
    reduce: str = "sum",
) -> None:
    if _ErrorCheckState.bypass_check:
        return

    if c.size() != ref.size():
        raise size_error(
            c.size(),
            ref.size(),
            "composition",
            "reference",
        )

    if reduce not in ["sum", "max"]:
        raise arg_value_error("reduce argument must be either sum or max.")

    if composition_mask is not None:
        composition_mask = composition_mask.to(torch.bool)
        c = c.c_masked_fill(composition_mask, 0.0)

    errors = c.c_sum() - ref

    if error_tensor_mask is not None:
        error_tensor_mask = error_tensor_mask.to(torch.bool)
        errors = errors.masked_fill(error_tensor_mask, 0.0)

    if reduce == "sum":
        reduced_error = errors.abs().sum()
    else:
        reduced_error = errors.abs().max()

    if reduced_error > error_bound:
        raise overflow_error(reduced_error.item(), error_bound)
