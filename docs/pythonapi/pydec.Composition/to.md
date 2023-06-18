# PYDEC.COMPOSITION.TO
> Composition.to(*args, **kwargs) →  {{{pydec_Composition}}}

Performs Composition dtype and/or device conversion. A {{{torch_dtype}}} and {{{torch_device}}} are inferred from the arguments of `self.to(*args, **kwargs)`.

?> If the `self` Composition already has the correct {{{torch_dtype}}} and {{{torch_device}}}, then `self` is returned. Otherwise, the returned composition is a copy of `self` with the desired {{{torch_dtype}}} and {{{torch_device}}}.

Here are the ways to call `to`:

> Composition.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) →  {{{pydec_Composition}}}

Returns a Composition with the specified `dtype`.

**Parameters:**

* **memory_format** (*{{{torch_memory_format}}}, optional*) – the desired memory format of returned composition. Default: **torch.preserve_format**.

> Composition.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) →  {{{pydec_Composition}}}

Returns a Composition with the specified `device` and (optional) `dtype`. If `dtype` is *None* it is inferred to be `self.dtype`. When non_blocking, tries to convert asynchronously with respect to the host if possible, e.g., converting a CPU Composition with pinned memory to a CUDA Composition. When `copy` is set, a new Composition is created even when the Composition already matches the desired conversion.

**Parameters:**

* **memory_format** (*{{{torch_memory_format}}}, optional*) – the desired memory format of returned composition. Default: **torch.preserve_format**.

> Composition.to(other, non_blocking=False, copy=False) →  {{{pydec_Composition}}}

Returns a Composition with same {{{torch_dtype}}} and {{{torch_device}}} as the `other`. `other` can be a Tensor or a Composition. When `non_blocking`, tries to convert asynchronously with respect to the host if possible, e.g., converting a CPU Composition with pinned memory to a CUDA Composition. When `copy` is set, a new Composition is created even when the Composition already matches the desired conversion.

Example:
```python
>>> composition = pydec.zeros(3, c_num=2)  # Initially dtype=float32, device=cpu
>>> composition.to(torch.float64)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.]),
  dtype=torch.float64}
"""
>>> cuda0 = torch.device("cuda:0")
>>> composition.to(cuda0)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.]),
  device='cuda:0'}
"""
>>> composition.to(cuda0, dtype=torch.float64)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.]),
  device='cuda:0', dtype=torch.float64}
"""
>>> t_other = torch.randn((), dtype=torch.float64, device=cuda0)
>>> composition.to(t_other, non_blocking=True)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.]),
  device='cuda:0', dtype=torch.float64}
"""
>>> c_other = pydec.zeros(5, c_num=6, dtype=torch.float64, device=cuda0)
>>> composition.to(c_other, non_blocking=True)
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.]),
  device='cuda:0', dtype=torch.float64}
"""
```