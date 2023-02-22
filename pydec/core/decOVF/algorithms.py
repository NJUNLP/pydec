from __future__ import annotations
import pydec
from torch import Tensor

from typing import Dict, Tuple, Union, Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..._composition import Composition

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

import pydec
from pydec.core.decOVF import register_decomposition_func, set_decomposition_func

__all__ = ["abs_decomposition", "_none_decomposition", "hybrid_decomposition"]


# TODO: all inplace operations should be re-examined, `ref` args should not be overridden
# TODO: algorithms should support `out` argument


@register_decomposition_func("abs_decomposition")
def abs_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    out: Optional[Composition] = None,
    ref: Optional[Tensor] = None,
    threshold: _float = 0.15,
    inplace: _bool = False,
) -> Composition:
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref
    recovery_out = func(recovery)
    residual_out = func(input._residual_tensor)

    decompose_out = recovery_out - residual_out

    composition = input._component_tensor
    abs_composition = composition.abs()

    multiplier = decompose_out / abs_composition.sum(dim=0)

    if inplace:
        input._component_tensor.abs_()
        input._component_tensor *= multiplier
        input._residual_tensor = residual_out
        return input
    else:
        out_component_tensor = abs_composition * multiplier
        out_residual_tensor = residual_out
        return pydec._from_replce(out_component_tensor, out_residual_tensor)


@register_decomposition_func("hybrid_decomposition")
def hybrid_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    threshold: _float = 0.15,
    inplace: _bool = False,
) -> Composition:
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref
    recovery_out = func(recovery)
    residual_out = func(input._residual_tensor)

    decompose_out = recovery_out - residual_out

    composition = input._component_tensor
    sum_composition = composition.sum(dim=0)
    abs_composition = composition.abs()
    abs_sum_composition = abs_composition.sum(dim=0, keepdim=True)
    instability_ratio = sum_composition.abs() / abs_sum_composition
    mask = (instability_ratio < threshold).expand_as(composition)

    if not inplace:
        composition = composition.clone()

    composition[mask] = composition[mask].abs()

    multiplier = decompose_out / composition.sum(dim=0)

    if inplace:
        input._component_tensor *= multiplier
        input._residual_tensor = residual_out
        return input
    else:
        out_component_tensor = composition * multiplier
        out_residual_tensor = residual_out
        return pydec._from_replce(out_component_tensor, out_residual_tensor)


@register_decomposition_func("none")
def _none_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    inplace: _bool = False,
) -> Composition:
    r"""
    Note: since PyDec 0.2.0, this algorithm is no longer the default
    decomposition algorithm for pydec, as the results it obtains do not make any sense.

    A trivial decomposition algorithm. Just add the output to residual.
    """
    if ref is None:
        recovery = input.c_sum()
    else:
        recovery = ref
    recovery_out = func(recovery)

    if inplace:
        input._component_tensor[:] = 0
        input._residual_tensor = recovery_out
        return input
    else:
        out = pydec.zeros_like(input)
        out._residual_tensor += recovery_out
        return out
