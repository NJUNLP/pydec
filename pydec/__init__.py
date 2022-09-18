from . import composition
from . import variable_functions
from .composition import Composition
from .bias_decomposition import (
    set_bias_decomposition_func,
    get_bias_decomposition_name,
    get_bias_decomposition_func,
    using_bias_decomposition_func,
    no_bias_decomposition,
)
from .variable_functions import (
    void,
    cat,
    c_cat,
    _from_replce,
    diagonal_init,
    call_torch_function,
)
from .error_check import no_error_check, error_check, check_error

__all__ = [
    "Composition",
    "set_bias_decomposition_func",
    "get_bias_decomposition_func",
    "using_bias_decomposition_func",
    "no_bias_decomposition",
]

"""
initialization
"""
composition._from_replce = variable_functions._from_replce
composition._get_bias_decomposition_name = get_bias_decomposition_name
composition._get_bias_decomposition_func = get_bias_decomposition_func


from pydec.nn import functional

# for name in dir(_C._VariableFunctions):
#     if name.startswith('__') or name in PRIVATE_OPS:
#         continue
#     obj = getattr(_C._VariableFunctions, name)
#     obj.__module__ = 'torch'
#     globals()[name] = obj
#     if not name.startswith("_"):
#         __all__.append(name)

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
