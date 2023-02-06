from . import composition as _composition
from . import variable_functions as _variable_functions
from .composition import Composition
from .decomposition import (
    set_decomposition_func,
    get_decomposition_name,
    get_decomposition_func,
    using_decomposition_func,
    no_decomposition,
    set_decomposition_args,
    get_decomposition_args,
    using_decomposition_args,
)

from .error_check import (
    no_error_check,
    error_check,
    check_error,
    is_error_checking_enabled,
)
from .autotracing import (
    no_tracing,
    enable_tracing,
    set_tracing_enabled,
    is_tracing_enabled,
)


__all__ = [
    "Composition",
    "set_decomposition_func",
    "get_decomposition_name",
    "get_decomposition_func",
    "using_decomposition_func",
    "set_decomposition_args",
    "get_decomposition_args",
    "using_decomposition_args",
    "no_error_check",
    "error_check",
    "check_error",
    "is_error_checking_enabled",
    "no_tracing",
    "enable_tracing",
    "set_tracing_enabled",
    "is_tracing_enabled",
]

import typing as _typing
import torch.types as _types

PRIVATE_NAME = ["memory_format", "strided"]
PRIVATE_NAME.extend(dir(_typing))
PRIVATE_NAME.extend(dir(_types))

import pydec.nn as nn

from typing import TYPE_CHECKING

# Fake import for type checking
if TYPE_CHECKING:
    from .variable_functions import *

for name in dir(_variable_functions):
    if name.startswith("__") or name in PRIVATE_NAME:
        continue
    obj = getattr(_variable_functions, name)
    obj.__module__ = "pydec"
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)

# initialization

_composition._from_replce = _variable_functions._from_replce
