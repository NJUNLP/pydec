from __future__ import annotations

import torch
import pydec
import math
import functools

from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Any,
    Callable,
    Optional,
    overload,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from ..._composition import Composition


@functools.lru_cache()
def get_subset(k: int) -> List[List[int]]:
    if k == 0:
        return [[]]
    ssub = get_subset(k - 1)
    sub = []
    for s in ssub:
        sub.append(s.copy())
        sub.append(s + [k - 1])
    return sub


def comb(n, m):
    return math.factorial(n) / math.factorial(m) / math.factorial(n - m)


def c_shapley(
    input: Composition,
    func: Callable,
) -> Composition:
    numc = input.numc()
    out_residual = func(input.residual)
    out_size = func(input.c_sum()).size()
    shapley_values = torch.zeros((numc,) + out_size).to(input)
    subset = get_subset(numc - 1)
    for i in range(numc):
        shapley_i = torch.zeros(out_size).to(input)
        for set in subset:
            indices = torch.LongTensor(set, device=input.device)
            indices[indices >= i] += 1
            mask_input = input.clone()
            mask_input()[indices] = 0
            include_out = func(mask_input.c_sum())
            mask_input()[i] = 0
            exclude_out = func(mask_input.c_sum())
            shapley_i += (include_out - exclude_out) / comb(numc - 1, len(indices))
        shapley_i /= numc
        shapley_values[i] = shapley_i

    return pydec.as_composition(shapley_values, out_residual)
