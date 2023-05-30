from .states import (
    register_decomposition_func,
    set_decomposition_func,
    get_decomposition_name,
    get_decomposition_func,
    using_decomposition_func,
    set_decomposition_args,
    get_decomposition_args,
    using_decomposition_args,
)
from .ops import *

from .algorithms import *

"""
Initialization
"""
try:
    set_decomposition_func("affine")
except ValueError:
    from . import states

    set_decomposition_func(
        states._DECOMPOSITION_FUNC_REGISTRY.keys().__iter__().__next__()
    )
