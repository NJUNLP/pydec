# PYDEC.MATMUL
> pydec.matmul(input, other, *, out=None) →  {{{pydec_Composition}}}

Matrix product of `input` and `other`.

The behavior depends on the dimensionality of the compositions/tensors, see [here](https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul) for details.

`input` and `other` can't both be compositions. If `input` is a composition, then `other` can only be a tensor, and vice versa.


**Parameters:**

* **input** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) – the first composition/tensor to be multiplied.
* **other** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) -  the second composition/tensor to be multiplied.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> mat1 = pydec.Composition(torch.randn(2, 10, 2, 3))
>>> mat2 = torch.randn(3, 3)
>>> out = pydec.matmul(mat1, mat2)
>>> out.size()
"""
torch.Size([10, 2, 3])
"""
>>> out.c_size()
"""
torch.Size([2, 10, 2, 3])
"""
```
