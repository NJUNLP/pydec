from . import composition as _composition
from . import variable_functions as _variable_functions
from .composition import Composition
from .bias_decomposition import (
    set_bias_decomposition_func,
    get_bias_decomposition_name,
    get_bias_decomposition_func,
    using_bias_decomposition_func,
    no_bias_decomposition,
    set_bias_decomposition_args,
    get_bias_decomposition_args,
    using_bias_decomposition_args,
)

from .error_check import (
    no_error_check,
    error_check,
    check_error,
    is_error_checking_enabled,
)

__all__ = [
    "Composition",
    "set_bias_decomposition_func",
    "get_bias_decomposition_name",
    "get_bias_decomposition_func",
    "using_bias_decomposition_func",
    "no_bias_decomposition",
    "set_bias_decomposition_args",
    "get_bias_decomposition_args",
    "using_bias_decomposition_args",
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
_composition._get_bias_decomposition_name = get_bias_decomposition_name
_composition._get_bias_decomposition_func = get_bias_decomposition_func


# import torch
# c = Composition((3, 4), 3, dtype=torch.float)


# # print(torch.tensor([0,0,0,0], dtype=torch.float) in c[0])
# # print(c[1])
# # print(c[1])
# c._composition_tensor[:] = 1
# # print(c.residual())
# c.residual()[:] = 1
# # print(c)
# # t1 = torch.randn(
# #     (
# #         3,
# #         3,
# #         4,
# #     )
# # )
# # t2 = torch.randn(
# #     (
# #         3,
# #         4,
# #         2,
# #     )
# # )
# # # print((t1 @ t2).size())
# c = 3 * c
# print(c)
# c += 3
# print(c)
# # t1 += "asd"
# # t1.__matmul__
# # CompositionModual.get_bias_decomposition_func()(c)
# print(c)
# # print(c.__repr__())
# # c.__str__
