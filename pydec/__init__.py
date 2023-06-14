from . import _composition
from . import overrides
from . import variable_functions as _variable_functions
from ._composition import (
    Composition,
    IndexComposition,
    enable_c_accessing,
    set_c_accessing_enabled,
    is_c_accessing_enabled,
)
from . import core

from .core import decOVF


# TODO: need update
__all__ = [
    "Composition",
    "IndexComposition",
    "enable_c_accessing",
    "set_c_accessing_enabled",
    "is_c_accessing_enabled",
]

import typing as _typing
import torch.types as _types

PRIVATE_NAME = ["memory_format", "strided"]
PRIVATE_NAME.extend(dir(_typing))
PRIVATE_NAME.extend(dir(_types))


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

import pydec.nn as nn
